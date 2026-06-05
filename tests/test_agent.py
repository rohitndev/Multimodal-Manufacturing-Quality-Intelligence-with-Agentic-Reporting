from src.agent import ERPClient, ISO9001ReportGenerator, QualityAgent
from src.rag import SpecRetriever


def _findings(severity="Critical", score=0.9):
    return [
        {
            "defect_id": "test-D001",
            "defect_type": "scratch",
            "confidence": 0.81,
            "bbox": [10, 20, 110, 60],
            "severity": severity,
            "severity_score": score,
            "severity_probs": {"Critical": 0.7, "Major": 0.2, "Minor": 0.1},
        }
    ]


def test_agent_classifies_and_reports(tmp_path):
    reports = ISO9001ReportGenerator(output_dir=str(tmp_path / "reports"))
    erp = ERPClient(outbox=str(tmp_path / "erp.jsonl"))
    agent = QualityAgent(
        retriever=SpecRetriever(fallback_path=str(tmp_path / "missing.json")),
        report_generator=reports,
        erp_client=erp,
    )

    decision = agent.run("INSP-1", _findings(severity="Critical", score=0.92))
    assert decision.decision in {"FAIL", "QUARANTINE", "REWORK", "PASS"}
    assert decision.report["report_id"].startswith("NCR-")
    assert decision.erp_response


def test_agent_pass_when_no_findings(tmp_path):
    reports = ISO9001ReportGenerator(output_dir=str(tmp_path / "reports"))
    erp = ERPClient(outbox=str(tmp_path / "erp.jsonl"))
    agent = QualityAgent(
        retriever=SpecRetriever(fallback_path=str(tmp_path / "missing.json")),
        report_generator=reports,
        erp_client=erp,
    )
    decision = agent.run("INSP-CLEAN", [])
    assert decision.decision == "PASS"
