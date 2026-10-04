import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("STORE_BACKEND", "memory")

import pymupdf

from models.schemas import Paper, TranslateAsset
from models.store import store, use_memory_store
from tools.translated_content_tool import get_translated_content


PAPER = Paper(
    id="2601.00002v1",
    title="Deterministic Translation Reading",
    authors=["A. Tester"],
    summary="A fixture paper.",
    pdf_url="https://arxiv.org/pdf/2601.00002v1.pdf",
)


class TranslatedContentToolTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)

        self.tmp = tempfile.TemporaryDirectory()
        self.mono_path = Path(self.tmp.name) / f"{PAPER.id}-mono.pdf"

        # The reader is language-agnostic; ASCII keeps the fixture free of
        # CJK font requirements.  Chinese text is covered by the replay tests.
        doc = pymupdf.open()
        for text in ("Translated page one.", "Translated page two."):
            doc.new_page().insert_text((50, 50), text, fontsize=11)
        doc.save(self.mono_path)
        doc.close()

        store.set_last_papers("s", [PAPER])
        self._set_translation("READY")

    def tearDown(self):
        self.tmp.cleanup()

    def _set_translation(self, status):
        store.upsert_translate_asset(
            TranslateAsset(
                paper_id=PAPER.id,
                output_mono_path=str(self.mono_path),
                status=status,
            )
        )

    def test_reads_one_page_of_the_translation(self):
        first = get_translated_content(session_id="s", ref=1)

        self.assertEqual(
            set(first),
            {"paper_id", "title", "page", "total_pages", "content"},
        )
        self.assertEqual(first["paper_id"], PAPER.id)
        self.assertEqual(first["page"], 1)
        self.assertEqual(first["total_pages"], 2)
        self.assertIn("page one", first["content"])
        self.assertNotIn("page two", first["content"])

        second = get_translated_content(session_id="s", ref=1, page=2)

        self.assertEqual(second["page"], 2)
        self.assertIn("page two", second["content"])

    def test_requires_ready_translation(self):
        self._set_translation("TRANSLATING")

        with self.assertRaisesRegex(ValueError, "call translate_arxiv_pdf first"):
            get_translated_content(session_id="s", ref=1)

        use_memory_store(reset=True)
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(ValueError, "call translate_arxiv_pdf first"):
            get_translated_content(session_id="s", ref=1)

    def test_rejects_invalid_pages(self):
        for page in (0, True):
            with self.subTest(page=page):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    get_translated_content(session_id="s", ref=1, page=page)

        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            get_translated_content(session_id="s", ref=1, page=3)


if __name__ == "__main__":
    unittest.main()
