"""Regression checks for discovery and fit ordering."""

import unittest

from booxplorer.discovery import (
    GRADES, MAX_ROUNDS, NEXT_PROMPT, PLAN_PROMPT, RESULT_LIMIT, STORES,
    BookHit, Observation, SearchPage, SearchUnavailable, bookstore_order,
    normalize_books, rank_books, search,
)


class Bookstore:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def search(self, store, query, limit):
        self.calls.append((store, query, limit))
        page = self.pages[(store, query)]
        if isinstance(page, Exception):
            raise page
        return page


class SearchTests(unittest.TestCase):
    def test_partial_and_zero_are_distinct_from_unavailable(self):
        isbn = "9791187879985"
        book = BookHit("통계와 확률의 원리", "https://example.com/1", isbn=isbn)
        unavailable_detail = BookHit("다른 책", "https://example.com/2", detail_error="timeout")
        provider = Bookstore({
            ("yes24", "확률"): SearchPage("https://example.com/search", (book, unavailable_detail), "확률 이야기"),
            ("aladin", "확률"): SearchPage("https://example.com/search", ()),
            ("kyobo", "확률"): SearchUnavailable("timeout"),
            ("aladin", "확률 입문"): SearchPage("https://example.com/search", (book,)),
        })
        inputs = []

        def model(prompt, data):
            inputs.append(data)
            if prompt == PLAN_PROMPT:
                return {"query": "확률", "conditions": ["생활 사례"]}
            if len(inputs) == 2:
                return {
                    "yes24": {"action": "search", "query": "확률 이야기", "reason": "repeat"},
                    "aladin": {"action": "search", "query": "확률 입문", "reason": "broaden"},
                }
            return {"aladin": {"action": "stop", "reason": "충분함"}}

        result = search("확률을 생활 사례로 배우고 싶다", provider, model, delay=0)
        self.assertEqual([r.status for r in result.rounds], ["partial", "empty", "unavailable", "ok"])
        self.assertEqual(result.conditions, ("생활 사례",))
        self.assertEqual(result.rounds[0].applied_query, "확률 이야기")
        self.assertEqual(result.observations[0].search_url, "https://example.com/search")
        self.assertEqual(result.observations[1].book.detail_error, "timeout")
        self.assertEqual(result.stop_reasons["yes24"], "repeated_query")
        self.assertEqual(result.rounds[-1].new_store_isbns, 1)
        self.assertEqual(result.rounds[-1].new_total_isbns, 0)
        self.assertEqual(set(inputs[1]["stores"]), {"yes24", "aladin"})
        self.assertIsNone(inputs[1]["requested_books"])
        self.assertEqual(inputs[1]["target_books"], 3)

    def test_each_store_caps_rounds_and_hits(self):
        class ManyBooks:
            def search(self, store, query, limit):
                return SearchPage("https://example.com", tuple(
                    BookHit(str(n), f"https://example.com/{n}") for n in range(8)
                ))

        def model(prompt, data):
            if prompt == PLAN_PROMPT:
                return {"query": "초기", "conditions": []}
            self.assertEqual(prompt, NEXT_PROMPT)
            return {site: {"action": "search", "query": str(data["remaining"]), "reason": "다음"}
                    for site in data["stores"]}

        result = search("초기", ManyBooks(), model, delay=0)
        self.assertEqual(len(result.rounds), MAX_ROUNDS * len(STORES))
        self.assertTrue(all(len(r.observations) == RESULT_LIMIT for r in result.rounds))
        self.assertEqual(set(result.stop_reasons.values()), {"round_limit"})

    def test_followup_failure_keeps_observed_books(self):
        book = BookHit("확률 책", "https://example.com/book")
        provider = Bookstore({
            (site, "확률"): SearchPage("https://example.com", (book,)) for site in STORES
        })

        def model(prompt, data):
            if prompt == PLAN_PROMPT:
                return {"query": "확률", "conditions": []}
            raise TimeoutError("model timeout")

        result = search("확률", provider, model, delay=0)
        self.assertEqual(len(result.observations), len(STORES))
        self.assertEqual(set(result.stop_reasons.values()), {"planning_failed"})
        self.assertEqual(result.planning_error, "model timeout")

    def test_invalid_plan_fails_before_provider_call(self):
        provider = Bookstore({})
        with self.assertRaises(ValueError):
            search("요청", provider, lambda *_: {"query": " ", "conditions": [" "]})
        self.assertFalse(provider.calls)


class RankTests(unittest.TestCase):
    def setUp(self):
        print_isbn = "9788932919584"
        ebook_isbn = "9788932966427"
        self.last_section = "마지막 장 일주기 리듬과 수면·각성"
        print_book = BookHit("우리는 왜 잠을 자야 할까", "https://example.com/print",
                             isbn=print_isbn, format="종이책", work_isbns=(ebook_isbn,),
                             toc="앞선 장 " * 300 + self.last_section)
        ebook = BookHit("우리는 왜 잠을 자야 할까", "https://example.com/ebook",
                        isbn=ebook_isbn, format="전자책")
        same_print = BookHit("우리는 왜 잠을 자야 할까", "https://example.org/print",
                             isbn="978-8932919584", author="매슈 워커")
        unrelated = BookHit("우리는 왜 잠을 자야 할까", "https://example.org/other")
        self.books = normalize_books([
            Observation("yes24", "수면", 1, 4, print_book),
            Observation("aladin", "수면", 1, 3, same_print),
            Observation("kyobo", "수면", 1, 5, ebook),
            Observation("kyobo", "수면", 1, 1, unrelated),
        ])

    def test_related_editions_keep_each_observation(self):
        book, other = self.books
        self.assertEqual(book.author, "매슈 워커")
        self.assertEqual(len(book.editions), 2)
        self.assertEqual(len(book.observations), 3)
        self.assertEqual(len(book.editions[0].sources), 2)
        self.assertEqual(bookstore_order(self.books), (other, book))

    def test_late_toc_survives_and_rank_keeps_original_books(self):
        def model(_, data):
            self.assertIn(self.last_section, data["books"][0]["contents"][0])
            self.assertNotIn("observations", data["books"][0])
            return {"direct":[1], "partial":[], "uncertain":[2], "off_topic":[]}

        ranked = rank_books("수면과 생체시계", self.books, model)
        self.assertIs(ranked[0].book, self.books[0])
        self.assertEqual([(r.position, r.grade) for r in ranked], [(1, "direct"), (2, "uncertain")])

    def test_missing_unknown_duplicate_and_boolean_ids_fail(self):
        for ids in ([1], [1, 3], [1, 1, 2], [True, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                rank_books("수면", self.books, lambda *_: {
                    grade: ids if grade == "direct" else [] for grade in GRADES
                })

    def test_missing_id_gets_one_repair_with_same_candidates(self):
        calls = []

        def model(_, data):
            calls.append(data)
            if len(calls) == 1:
                return {"direct":[1], "partial":[], "uncertain":[], "off_topic":[]}
            self.assertIn("omitted", data["validation_error"])
            self.assertEqual(data["books"], calls[0]["books"])
            return {"direct":[1], "partial":[2], "uncertain":[], "off_topic":[]}

        ranked = rank_books("수면", self.books, model)
        self.assertEqual([r.book.id for r in ranked], [1, 2])
        self.assertEqual(len(calls), 2)

    def test_empty_bypasses_model(self):
        self.assertEqual(rank_books("수면", [], lambda *_: self.fail("unexpected model call")), ())


if __name__ == "__main__":
    unittest.main()
