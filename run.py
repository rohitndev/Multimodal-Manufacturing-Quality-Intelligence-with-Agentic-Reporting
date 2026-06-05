"""CLI entrypoints.

Usage:
    python run.py serve                                # starts the FastAPI server
    python run.py ingest                               # re-indexes spec PDFs
    python run.py inspect path/to/image.jpg            # one-shot inspection
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("run")


def _serve(args):
    import uvicorn
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


def _ingest(args):
    from src.rag import SpecIngestion
    n = SpecIngestion(specs_dir=args.specs_dir).ingest()
    log.info("Ingested %d documents.", n)


def _inspect(args):
    import cv2
    from src.vision import VisionPipeline
    from src.rag import SpecRetriever
    from src.agent import ERPClient, ISO9001ReportGenerator, QualityAgent

    path = Path(args.image)
    if not path.exists():
        sys.exit(f"Image not found: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not decode image: {path}")

    vision = VisionPipeline()
    agent = QualityAgent(
        retriever=SpecRetriever(),
        report_generator=ISO9001ReportGenerator(),
        erp_client=ERPClient(),
    )

    overlay_dir = Path("data/overlays")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay = str(overlay_dir / f"{path.stem}.png")

    result = vision.run(frame, image_id=path.stem, overlay_output=overlay)
    decision = agent.run(
        inspection_id=path.stem,
        findings=[f.to_dict() for f in result.findings],
        product=args.product,
    )
    payload = {
        "image": str(path),
        "findings": [f.to_dict() for f in result.findings],
        "decision": decision.decision,
        "rationale": decision.rationale,
        "report_paths": {
            "markdown": decision.report.get("markdown_path"),
            "pdf": decision.report.get("pdf_path"),
        },
        "overlay": result.overlay_path,
        "erp": decision.erp_response,
    }
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality inspection CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    ing = sub.add_parser("ingest", help="Ingest product spec PDFs")
    ing.add_argument("--specs-dir", default="data/sample_specs")
    ing.set_defaults(func=_ingest)

    insp = sub.add_parser("inspect", help="Run a one-shot inspection")
    insp.add_argument("image")
    insp.add_argument("--product", default="Generic Component")
    insp.set_defaults(func=_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
