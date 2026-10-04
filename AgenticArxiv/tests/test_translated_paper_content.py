"""Real-PDF coverage for translated text, continuation, and readiness."""

import hashlib
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import Paper, PdfAsset, TranslateAsset
from models.store_memory import MemoryStore
from tools import paper_content_tool as reader


def write_pdf(path, text):
    with pymupdf.open() as doc:
        page = doc.new_page()
        assert page.insert_textbox(
            pymupdf.Rect(40, 40, 550, 780), text,
            fontname="china-s", fontsize=11,
        ) >= 0
        doc.save(path)


@pytest.fixture
def translated_store(tmp_path, monkeypatch):
    backend = MemoryStore(make_dirs=False)
    monkeypatch.setattr("models.store._backend", backend)
    papers = [Paper(id="2601.00001v1", title="First paper"),
              Paper(id="2601.00002v1", title="Second paper")]
    backend.set_last_papers("s", papers)
    backend.set_last_active_paper_id("s", papers[1].id)
    paths = []
    for index, paper in enumerate(papers, 1):
        raw = tmp_path / f"raw{index}.pdf"
        mono = tmp_path / f"mono{index}.pdf"
        write_pdf(raw, "Abstract\nSOURCE_ONLY original abstract.")
        write_pdf(mono, f"TRANSLATED_{index}\n中文译文第{index}篇。\n" +
                  "\n".join(f"第{line}行：正文内容用于检查连续读取。" for line in range(25)))
        backend.upsert_pdf_asset(PdfAsset(
            paper_id=paper.id, local_path=str(raw), status="READY"))
        backend.upsert_translate_asset(TranslateAsset(
            paper_id=paper.id, output_mono_path=str(mono), status="READY"))
        paths.append(mono)
    return backend, papers, paths


def test_reads_translation_and_continues_without_gaps(translated_store):
    assert hasattr(reader, "get_translated_paper_content"), "translated reader is missing"
    backend, papers, paths = translated_store
    chunks, offset = [], 0
    while True:
        result = reader.get_translated_paper_content("s", 1, offset, 37)
        assert result["paper_id"] == papers[0].id
        assert result["source"] == "translated"
        assert result["source_sha256"] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
        assert result["offset"] == offset
        assert 0 < len(result["content"]) <= 37
        assert result["next_offset"] == offset + len(result["content"])
        chunks.append(result["content"])
        offset = result["next_offset"]
        if not result["has_more"]:
            break
    text = "".join(chunks)
    assert text == reader._pdf_to_text(str(paths[0]))
    assert "TRANSLATED_1" in text and "中文译文" in text
    assert "SOURCE_ONLY" not in text and "TRANSLATED_2" not in text
    assert backend.get_last_active_paper_id("s") == papers[0].id


@pytest.mark.parametrize("ref", [2, "第2篇", "2601.00002v1", "Second", None])
def test_reference_forms_select_the_same_translation(translated_store, ref):
    _, papers, _ = translated_store
    result = reader.get_translated_paper_content("s", ref)
    assert result["paper_id"] == papers[1].id
    assert "TRANSLATED_2" in result["content"]


@pytest.mark.parametrize("state", ["missing", "TRANSLATING", "FAILED", "missing_file", "image_only"])
def test_unavailable_translation_fails_without_changing_reference(translated_store, state):
    backend, papers, paths = translated_store
    if state == "missing":
        backend.delete_translate_asset(papers[0].id)
    elif state == "missing_file":
        paths[0].unlink()
    elif state == "image_only":
        paths[0].unlink()
        with pymupdf.open() as doc:
            doc.new_page()
            doc.save(paths[0])
    else:
        backend.update_translate_asset(papers[0].id, status=state)
    with pytest.raises(ValueError):
        reader.get_translated_paper_content("s", 1)
    assert backend.get_last_active_paper_id("s") == papers[1].id


@pytest.mark.parametrize("args", [dict(ref=" "), dict(offset=-1), dict(offset=10_000),
                                  dict(max_chars=0), dict(max_chars=4001)])
def test_invalid_read_parameters_fail(translated_store, args):
    with pytest.raises(ValueError):
        reader.get_translated_paper_content(session_id="s", **args)
