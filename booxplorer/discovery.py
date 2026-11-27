"""Natural-language book discovery using existing search and model providers."""

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Literal, Protocol
from unicodedata import normalize
import re


STORES = ("yes24", "aladin", "kyobo")
MAX_ROUNDS = 5
DEFAULT_TARGET_BOOKS = 3
RESULT_LIMIT = 5
Chat = Callable[[str, dict], dict]


class SearchUnavailable(Exception):
    """A bookstore request failed without returning a trustworthy search page."""


@dataclass(frozen=True)
class BookHit:
    title: str
    url: str
    author: str = ""
    isbn: str = ""
    format: str = ""
    description: str = ""
    toc: str = ""
    work_isbns: tuple[str, ...] = ()
    detail_error: str = ""
    publisher: str = ""
    publication_date: str = ""
    language: str = ""


@dataclass(frozen=True)
class Observation:
    store: str
    query: str
    round_number: int
    position: int
    book: BookHit
    search_url: str = ""
    applied_query: str = ""


@dataclass(frozen=True)
class SearchPage:
    url: str
    books: tuple[BookHit, ...]
    applied_query: str = ""
    notice: str = ""


class BookstoreProvider(Protocol):
    def search(self, store: str, query: str, limit: int) -> SearchPage:
        """Return the external module's results in bookstore order.

        A confirmed zero-result page has no books. Raise SearchUnavailable on
        transport or access failure. Preserve any hit whose detail lookup fails.
        The external module handles request pacing within a search and its details.
        """
        ...


@dataclass(frozen=True)
class SearchRound:
    store: str
    query: str
    number: int
    status: Literal["ok", "empty", "partial", "unavailable"]
    url: str = ""
    applied_query: str = ""
    notice: str = ""
    observations: tuple[Observation, ...] = ()
    new_store_isbns: int = 0
    new_total_isbns: int = 0


@dataclass(frozen=True)
class SearchResult:
    request: str
    conditions: tuple[str, ...]
    rounds: tuple[SearchRound, ...]
    stop_reasons: dict[str, str]
    planning_error: str = ""
    requested_books: int | None = None

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(item for page in self.rounds for item in page.observations)


PLAN_PROMPT = """사용자의 도서 탐색 요청을 국내 서점 통합검색용으로 준비한다.
짧고 그 자체로 검색할 수 있는 요청은 원문을 첫 검색어로 쓴다. 짧은 질문은 질문 표현만 걷어낸다.
긴 요청의 첫 검색어는 주제 또는 대상을 나타내는 핵심 명사나 짧은 명사구 하나다.
난이도, 사례, 독서 목적을 검색어에 이어 붙이지 말고 conditions에 넣는다.
예를 들어 '고대 그리스 역사에 처음 접근하는데 연표보다 생활상 중심의 책을 찾고 싶어요'의 첫 검색어는 '고대 그리스'다.
검색 연산자와 문장형 검색어는 피한다.
검색어의 제약과 독자의 목적·언어·난이도·설명 방식·명시 제외 조건을 원뜻대로 conditions에 보존한다.
선호를 필수 조건으로 바꾸거나 요구하지 않은 분야를 추가하지 않는다.
사용자가 원하는 권수를 숫자로 명시한 경우만 requested_books에 넣는다. 없으면 null이다.
JSON으로 query(문자열), conditions(문자열 배열), requested_books(양의 정수 또는 null)를 반환한다."""

NEXT_PROMPT = """국내 서점의 순위, 적용 검색어, 신규 ISBN, 후보와 근거를 보고 다음 행동을 결정한다.
한 서점에서 target_books만큼 잠정 적합한 서로 다른 저작을 찾았거나 의미를 유지한 새 검색어가 없으면 종료한다.
requested_books가 null이면 사용자가 요구한 권수는 없다. target_books는 내부 탐색 목표이며 사용자 요청 권수로 설명하지 않는다.
ISBN으로 구별되는 판본이라도 같은 저작은 한 권이다. 관찰된 후보 밖의 책은 판단에 쓰지 않는다.
근거가 부족해도 주제에 맞는 후보는 미확인으로 고려한다. 후보가 부족할 때만 새로운 짧은 명사구를 제안한다.
서점마다 search 또는 stop, 재시도하지 않은 query, 이유를 각각 준다. 이유에 책을 언급할 때 해당 서점에서 관찰한 후보만 인용한다.
JSON은 서점명을 키로 하고 각각 action, query, reason을 문자열로 반환한다."""

def search(request: str, provider: BookstoreProvider, chat: Chat, *, delay: float = 1.0) -> SearchResult:
    """Run independent bounded searches using a structured bookstore provider."""
    if not request.strip():
        raise ValueError("request must not be blank")
    if delay < 0:
        raise ValueError("delay must be nonnegative")

    plan = chat(PLAN_PROMPT, {"request": request})
    query = plan.get("query")
    conditions = plan.get("conditions")
    requested_books = plan.get("requested_books")
    if not isinstance(query, str) or not query.strip() or not isinstance(conditions, list):
        raise ValueError("invalid query plan")
    if any(not isinstance(condition, str) or not condition.strip() for condition in conditions):
        raise ValueError("invalid reading condition")
    if requested_books is not None and (type(requested_books) is not int or requested_books < 1):
        raise ValueError("requested_books must be a positive integer or null")

    pending = dict.fromkeys(STORES, query.strip())
    history = {store: [] for store in STORES}
    store_isbns = {store: set() for store in STORES}
    total_isbns = set()
    completed_at = {}
    rounds = []
    stops = {}
    planning_error = ""
    for number in range(1, MAX_ROUNDS + 1):
        for store, query in pending.items():
            if store in completed_at:
                sleep(max(0, delay - (monotonic() - completed_at[store])))
            try:
                page = provider.search(store, query, RESULT_LIMIT)
            except SearchUnavailable as error:
                result = SearchRound(store, query, number, "unavailable", notice=str(error))
                stops[store] = "unavailable"
            else:
                observations = tuple(
                    Observation(store, query, number, position, book, page.url, page.applied_query)
                    for position, book in enumerate(page.books[:RESULT_LIMIT], 1)
                )
                isbns = {isbn for item in observations if (isbn := isbn13(item.book.isbn))}
                status = "empty"
                if observations:
                    status = "partial" if any(item.book.detail_error for item in observations) else "ok"
                result = SearchRound(
                    store, query, number, status, page.url, page.applied_query, page.notice,
                    observations, len(isbns - store_isbns[store]), len(isbns - total_isbns),
                )
                store_isbns[store].update(isbns)
                total_isbns.update(isbns)
            completed_at[store] = monotonic()
            rounds.append(result)
            history[store].append(result)

        active = [store for store in pending if store not in stops]
        if not active:
            break
        if number == MAX_ROUNDS:
            stops.update(dict.fromkeys(active, "round_limit"))
            break
        context = {
            store: [_round_context(page) for page in history[store]]
            for store in active
        }
        try:
            decisions = chat(NEXT_PROMPT, {
                "request": request, "conditions": conditions, "remaining": MAX_ROUNDS - number,
                "requested_books": requested_books, "target_books": requested_books or DEFAULT_TARGET_BOOKS,
                "stores": context,
            })
        except (OSError, ValueError) as error:
            planning_error = str(error)
            stops.update(dict.fromkeys(active, "planning_failed"))
            break
        pending = {}
        for store in active:
            decision = decisions.get(store)
            if not isinstance(decision, dict):
                stops[store] = "invalid_decision"
                continue
            action, candidate, reason = (decision.get(field) for field in ("action", "query", "reason"))
            if action == "stop" and isinstance(reason, str) and reason.strip():
                stops[store] = reason.strip()
            elif action == "search" and isinstance(candidate, str) and candidate.strip():
                tried = {
                    _text_key(text)
                    for page in history[store] for text in (page.query, page.applied_query) if text
                }
                if _text_key(candidate) in tried:
                    stops[store] = "repeated_query"
                else:
                    pending[store] = candidate.strip()
            else:
                stops[store] = "invalid_decision"
        if not pending:
            break
    return SearchResult(request, tuple(conditions), tuple(rounds), stops, planning_error, requested_books)


def _round_context(page: SearchRound) -> dict:
    return {
        "query": page.query, "applied_query": page.applied_query, "notice": page.notice,
        "status": page.status, "shown": len(page.observations),
        "new_store_isbns": page.new_store_isbns, "new_total_isbns": page.new_total_isbns,
        "books": [
            {"position": item.position, "title": item.book.title, "isbn": item.book.isbn,
             "author": item.book.author, "format": item.book.format, "work_isbns": item.book.work_isbns,
             "description": item.book.description[:240], "detail_error": item.book.detail_error}
            for item in page.observations
        ],
    }


def isbn13(value: str) -> str:
    isbn = re.sub(r"[\s-]", "", normalize("NFKC", value))
    if not re.fullmatch(r"97[89][0-9]{10}", isbn):
        return ""
    check = sum(int(digit) * (1, 3)[i % 2] for i, digit in enumerate(isbn))
    return isbn if check % 10 == 0 else ""


def _text_key(value: str) -> str:
    return " ".join(normalize("NFKC", value).casefold().split())
