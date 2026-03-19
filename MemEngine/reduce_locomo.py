#!/usr/bin/env python3
"""
Reduce LOCOMO dataset to 1/10 of conversations (sessions) and keep only QAs
whose evidence falls entirely within the kept sessions (answerable).
"""
import json
import re
from pathlib import Path


def _session_keys_with_content(conv: dict) -> list[int]:
    """Return sorted list of session indices that have dialogue content (session_K is a list)."""
    indices = []
    for k, v in conv.items():
        m = re.match(r"^session_(\d+)$", k)
        if m and isinstance(v, list):
            indices.append(int(m.group(1)))
    return sorted(indices)


def _parse_evidence_sessions(evidence_list: list) -> set[int]:
    """
    From QA evidence list like ["D1:3", "D2:8"] or ["D8:6; D9:17"], extract session indices.
    Returns set of session numbers (Dn -> n).
    """
    sessions = set()
    for item in evidence_list:
        if not isinstance(item, str):
            continue
        # One item may be "D8:6; D9:17"
        for part in item.split(";"):
            part = part.strip()
            m = re.match(r"^D(\d+)\s*:", part) or re.match(r"^D(\d+):", part)
            if m:
                sessions.add(int(m.group(1)))
    return sessions


def _keep_conversation_subset(conv: dict, kept_sessions: set[int]) -> dict:
    """Build new conversation dict with only session_1..session_K and related keys."""
    out = {}
    for k, v in conv.items():
        if k in ("speaker_a", "speaker_b"):
            out[k] = v
            continue
        m = re.match(r"^session_(\d+)(_date_time|_observation)?$", k)
        if m:
            idx = int(m.group(1))
            if idx in kept_sessions:
                out[k] = v
            continue
        if k == "event_summary" and isinstance(v, dict):
            new_es = {}
            for k2, v2 in v.items():
                m2 = re.match(r"^(events_session_|session_)(\d+)(_observation)?$", k2)
                if m2:
                    if int(m2.group(2)) in kept_sessions:
                        new_es[k2] = v2
                else:
                    new_es[k2] = v2
            out["event_summary"] = new_es
            continue
        out[k] = v
    return out


def reduce_sample(sample: dict, fraction: float = 0.1) -> dict | None:
    """
    Reduce one sample: keep 1/fraction of sessions (at least 1), then keep only QAs
    whose evidence lies entirely in kept sessions. Returns new sample or None if no QA left.
    """
    conv = sample.get("conversation") or {}
    qa_list = sample.get("qa") or []

    session_indices = _session_keys_with_content(conv)
    if not session_indices:
        return None

    n_total = len(session_indices)
    n_keep = max(1, int(n_total * fraction))  # keep first 1/10, at least 1 session
    kept_sessions = set(session_indices[:n_keep])

    new_conv = _keep_conversation_subset(conv, kept_sessions)

    new_qa = []
    for qa in qa_list:
        evidence = qa.get("evidence") or []
        if not evidence:
            # No evidence -> unanswerable from conversation -> drop
            continue
        ev_sessions = _parse_evidence_sessions(evidence)
        if not ev_sessions:
            continue
        if ev_sessions <= kept_sessions:
            new_qa.append(qa)

    if not new_qa:
        return None

    out = {k: v for k, v in sample.items() if k not in ("conversation", "qa")}
    out["conversation"] = new_conv
    out["qa"] = new_qa
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reduce LOCOMO to 1/10 sessions and filter QA by evidence.")
    parser.add_argument("--input", "-i", type=Path, default=None, help="Input locomo10.json path")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path (default: <input_dir>/locomo10_simplified.json)")
    parser.add_argument("--fraction", "-f", type=float, default=0.1, help="Fraction of sessions to keep (default: 0.1)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    if args.input is not None:
        input_path = Path(args.input)
    else:
        for path in [base / "locomo10.json", base.parent / "A-mem" / "data" / "locomo10.json"]:
            if path.exists():
                input_path = path
                break
        else:
            input_path = base / "locomo10.json"

    output_path = args.output if args.output is not None else input_path.parent / (input_path.stem + "_simplified.json")

    print(f"Reading {input_path} ...")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise SystemExit("Expected JSON array of samples")

    reduced = []
    total_qa_orig = 0
    total_qa_new = 0
    for sample in data:
        total_qa_orig += len(sample.get("qa") or [])
        new_sample = reduce_sample(sample, fraction=args.fraction)
        if new_sample is not None:
            reduced.append(new_sample)
            total_qa_new += len(new_sample.get("qa") or [])

    output_path.write_text(json.dumps(reduced, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(reduced)} samples to {output_path}")
    print(f"Samples: {len(data)} -> {len(reduced)}  |  QA: {total_qa_orig} -> {total_qa_new}")


if __name__ == "__main__":
    main()
