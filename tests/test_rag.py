from src.rag import SpecIngestion, SpecRetriever


def test_ingest_and_retrieve(tmp_path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "demo.txt").write_text(
        "Scratches longer than 5mm are Critical. "
        "Dents are Major when deeper than 0.5mm. "
        "Voids exceeding 2mm^2 are Major.",
        encoding="utf-8",
    )
    fallback = tmp_path / "corpus.json"
    n = SpecIngestion(specs_dir=str(specs_dir), fallback_path=str(fallback)).ingest()
    assert n >= 1

    retr = SpecRetriever(fallback_path=str(fallback))
    ctx = retr.retrieve("scratch")
    assert ctx.snippets, "expected retrieval to surface at least one snippet"
    assert any("scratch" in s.lower() for s in ctx.snippets)
