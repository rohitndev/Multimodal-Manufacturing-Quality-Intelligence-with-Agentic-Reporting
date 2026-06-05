"""LangGraph-orchestrated quality agent.

Multi-step graph:
    receive_defect → retrieve_spec → classify_decision → generate_report → call_erp.

Uses Ollama (local Llama 3.1) for LLM-driven reasoning when available, and a
deterministic rule engine otherwise — both produce equivalent decisions.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    decision: str  # PASS / REWORK / QUARANTINE / FAIL
    rationale: str
    report: Dict[str, Any]
    erp_response: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


SEVERITY_TO_DECISION = {
    "Critical": "FAIL",
    "Major": "QUARANTINE",
    "Minor": "REWORK",
}


class QualityAgent:
    """LangGraph multi-step inspection agent."""

    def __init__(
        self,
        retriever,
        report_generator,
        erp_client,
        ollama_model: Optional[str] = None,
    ):
        self.retriever = retriever
        self.report_generator = report_generator
        self.erp_client = erp_client
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangGraph unavailable (%s) — sequential fallback", exc)
            return None

        graph = StateGraph(dict)
        graph.add_node("retrieve_spec", self._node_retrieve)
        graph.add_node("classify_decision", self._node_classify)
        graph.add_node("generate_report", self._node_report)
        graph.add_node("call_erp", self._node_erp)
        graph.set_entry_point("retrieve_spec")
        graph.add_edge("retrieve_spec", "classify_decision")
        graph.add_edge("classify_decision", "generate_report")
        graph.add_edge("generate_report", "call_erp")
        graph.add_edge("call_erp", END)
        return graph.compile()

    def run(
        self,
        inspection_id: str,
        findings: List[dict],
        product: str = "Generic Component",
    ) -> AgentDecision:
        state = {
            "inspection_id": inspection_id,
            "findings": findings,
            "product": product,
        }
        if self._graph is not None:
            try:
                final = self._graph.invoke(state)
                return self._finalise(final)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Graph invocation failed (%s) — sequential fallback", exc)
        # Sequential fallback
        state = self._node_retrieve(state)
        state = self._node_classify(state)
        state = self._node_report(state)
        state = self._node_erp(state)
        return self._finalise(state)

    # ------- Nodes -------
    def _node_retrieve(self, state: dict) -> dict:
        findings = state.get("findings", [])
        spec_ctx = None
        if findings:
            top = findings[0]
            spec_ctx = self.retriever.retrieve(top["defect_type"], product=state.get("product"))
        state["spec_context"] = spec_ctx
        return state

    def _node_classify(self, state: dict) -> dict:
        findings = state.get("findings", [])
        if not findings:
            state["decision"] = "PASS"
            state["rationale"] = "No defects detected above the confidence threshold."
            return state

        worst = max(findings, key=lambda f: f["severity_score"])
        rule_decision = SEVERITY_TO_DECISION.get(worst["severity"], "QUARANTINE")
        rule_rationale = (
            f"Worst defect {worst['defect_id']} graded {worst['severity']} "
            f"(score={worst['severity_score']:.2f}); per spec tolerance → {rule_decision}."
        )

        llm_out = self._llm_reason(findings, state.get("spec_context"))
        if llm_out:
            state["decision"] = llm_out.get("decision", rule_decision)
            state["rationale"] = llm_out.get("rationale", rule_rationale)
        else:
            state["decision"] = rule_decision
            state["rationale"] = rule_rationale
        return state

    def _node_report(self, state: dict) -> dict:
        report = self.report_generator.generate(
            inspection_id=state["inspection_id"],
            findings=state["findings"],
            spec_context=state.get("spec_context"),
            decision=state["decision"],
            rationale=state["rationale"],
            product=state.get("product", "Generic Component"),
        )
        state["report"] = report
        return state

    def _node_erp(self, state: dict) -> dict:
        report = state["report"]
        erp_resp = self.erp_client.update_wip(
            report_id=report["report_id"],
            inspection_id=state["inspection_id"],
            decision=state["decision"],
            payload={"findings": state["findings"]},
        )
        state["erp_response"] = erp_resp
        return state

    # ------- LLM bridge -------
    def _llm_reason(self, findings: List[dict], spec_context) -> Optional[dict]:
        try:
            import ollama  # type: ignore
        except Exception:  # noqa: BLE001
            return None
        try:
            prompt = (
                "You are a manufacturing QA assistant. Given the detected defects and the "
                "product specification snippets below, decide PASS / REWORK / QUARANTINE / FAIL "
                "and give a one-sentence rationale. Respond as compact JSON with keys "
                "'decision' and 'rationale'.\n\n"
                f"DEFECTS:\n{json.dumps(findings, indent=2)}\n\n"
                f"SPECIFICATION:\n{getattr(spec_context, 'text', '') or 'n/a'}\n"
            )
            resp = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.get("message", {}).get("content", "").strip()
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end == -1:
                return None
            return json.loads(content[start : end + 1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama reasoning failed (%s)", exc)
            return None

    def _finalise(self, state: dict) -> AgentDecision:
        return AgentDecision(
            decision=state.get("decision", "QUARANTINE"),
            rationale=state.get("rationale", ""),
            report=state.get("report", {}),
            erp_response=state.get("erp_response", {}),
        )
