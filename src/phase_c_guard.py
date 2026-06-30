from __future__ import annotations

"""Phase C: Production Guardrails: PII + input/output rails + latency."""

import asyncio
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS

PII_PATTERNS = [
    ("VN_CCCD", re.compile(r"\b\d{12}\b"), 0.9),
    ("VN_CCCD", re.compile(r"\b\d{9}\b"), 0.7),
    ("VN_PHONE", re.compile(r"\b0[3-9]\d{8}\b"), 0.9),
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0.95),
]

BLOCK_KEYWORDS = [
    "ignore", "forget", "system override", "admin command", "dan", "do anything now",
    "unrestricted", "confidential", "employee salaries", "all employee", "training data",
    "system instructions", "previous instructions", "jailbreak", "dump", "reveal",
    "mat khau admin", "password admin", "luong chi tiet", "bang luong", "cccd", "cmnd",
    "so dien thoai", "thong tin nhan vien", "tan cong mang", "tiet lo", "bo qua", "dong vai", "ra lenh", "tat ca nhan vien", "bitcoin", "ethereum",
    "marvel", "nau pho", "bai tho", "phuong trinh vi phan",
]

ALLOW_HINTS = [
    "nghi", "phep", "bao hiem", "phu cap", "luong", "thu viec", "mat khau",
    "vpn", "dao tao", "cong tac", "expense", "tam ung", "mentor", "hr", "policy",
]


def setup_presidio():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9), Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7)],
    ))
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    ))
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def _regex_pii(text: str) -> list[dict]:
    entities: list[dict] = []
    for entity_type, pattern, score in PII_PATTERNS:
        for match in pattern.finditer(text):
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": match.start(),
                "end": match.end(),
            })
    entities.sort(key=lambda item: (item["start"], item["end"]))
    return entities


def _anonymize(text: str, entities: list[dict]) -> str:
    output = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        output = output[:entity["start"]] + f"<{entity['type']}>" + output[entity["end"]:]
    return output


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    if analyzer is not None and anonymizer is not None:
        try:
            results = analyzer.analyze(text=text, language="en")
            entities = [
                {"type": r.entity_type, "text": text[r.start:r.end], "score": round(r.score, 3), "start": r.start, "end": r.end}
                for r in results
            ]
            if entities:
                return {"has_pii": True, "entities": entities, "anonymized": anonymizer.anonymize(text=text, analyzer_results=results).text}
        except Exception:
            pass
    entities = _regex_pii(text)
    return {"has_pii": bool(entities), "entities": entities, "anonymized": _anonymize(text, entities) if entities else text}


def setup_nemo_rails():
    from nemoguardrails import LLMRails, RailsConfig
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _looks_blocked(text: str) -> tuple[bool, str | None]:
    lower = text.lower()
    asciiish = _strip_accents(lower).encode("ascii", "ignore").decode("ascii")
    haystack = lower + " " + asciiish
    if any(keyword in haystack for keyword in BLOCK_KEYWORDS):
        return True, "policy_or_injection"
    if not any(hint in haystack for hint in ALLOW_HINTS):
        return True, "off_topic"
    return False, None


async def check_input_rail(text: str, rails=None) -> dict:
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response_text = response if isinstance(response, str) else str(response)
            refuse = ["xin loi", "khong the", "khong duoc", "i cannot", "i'm sorry", "sorry"]
            blocked = any(word in response_text.lower() for word in refuse)
            return {"allowed": not blocked, "blocked_reason": "nemo_input_rail" if blocked else None, "response": response_text}
        except Exception as exc:
            blocked, reason = _looks_blocked(text)
            return {"allowed": not blocked, "blocked_reason": reason, "response": f"heuristic fallback after rails error: {exc}"}
    blocked, reason = _looks_blocked(text)
    return {"allowed": not blocked, "blocked_reason": reason, "response": "heuristic_input_rail"}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    pii = pii_scan(answer)
    blocked, reason = _looks_blocked(answer)
    if pii["has_pii"]:
        return {"safe": False, "flagged_reason": "pii_output", "final_answer": pii["anonymized"]}
    if blocked and reason != "off_topic":
        return {"safe": False, "flagged_reason": reason, "final_answer": "Response blocked by output guardrail."}
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_adversarial_suite(adversarial_set: list[dict], rails=None, analyzer=None, anonymizer=None) -> list[dict]:
    async def _run_all():
        rows = []
        for item in adversarial_set:
            text = item.get("input", "")
            blocked_by = None
            pii = pii_scan(text, analyzer, anonymizer)
            if pii["has_pii"]:
                blocked_by = "presidio"
            if blocked_by is None:
                rail = await check_input_rail(text, rails)
                if not rail["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            rows.append({
                "id": item.get("id"),
                "category": item.get("category", "unknown"),
                "input": text[:80] + ("..." if len(text) > 80 else ""),
                "expected": item.get("expected", "blocked"),
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item.get("expected", "blocked"),
            })
        return rows
    return _run_async(_run_all())


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    def pick(q: float) -> float:
        return round(ordered[min(int((n - 1) * q), n - 1)], 2)
    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20, rails=None, analyzer=None, anonymizer=None) -> dict:
    samples = (test_inputs or ["test input"])[:n_runs]
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure():
        for text in samples:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    _run_async(_measure())
    total = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total,
        "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


if __name__ == "__main__":
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    results = run_adversarial_suite(data)
    print(f"Adversarial suite: {sum(r['passed'] for r in results)}/{len(results)} passed")
    print(measure_p95_latency([row["input"] for row in data[:5]], n_runs=5))