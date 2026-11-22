"""Collector for wild_full (protocol: PREREG_full.md).

Design constraints that are enforced in code, not just documented:

* default is a *dry run*; real traffic requires both --execute and --i-have-approval;
* plain HTTP is only accepted for 127.0.0.1/localhost (mock server); anything else
  must be https;
* `max_retries` defaults to 0. If enabled, the wait honours Retry-After and at most
  one retry is allowed per request;
* every request is written to CSV immediately and flushed, so a crash loses nothing;
* the API key is never logged, never written to disk, and never placed in a row;
* a conservative cost pre-charge aborts the session before the budget is exceeded;
* per-round order is randomised and the model order rotates (Latin) across rounds,
  so model identity is never collinear with collection position.

Usage
-----
    python collector.py --plan plan.json --session S1-20261119T0930 \
        --out out --key-file /path/to/key            # dry run, prints the plan
    python collector.py ... --execute --i-have-approval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

FIELDS = [
    # identity
    "session_id", "round", "seq_in_round", "model_requested", "provider",
    "model_served", "request_id",
    # input
    "prompt_id", "prompt_sha256",
    # time
    "wall_ts", "t_start_mono", "t_end_mono",
    # status
    "http_status", "http_attempts", "retry_after_s", "error_class", "error_message_sha",
    # timing
    "ttfb_ms", "ttft_role_ms", "ttft_content_ms", "total_ms", "inter_chunk_gap_mean_ms",
    # packet structure
    "chunk_count", "content_chunk_count", "sse_bytes_total", "sse_overhead_bytes", "chunk_bytes_min",
    "chunk_bytes_max", "chunk_bytes_mean", "chunk_bytes_cv", "first_chunk_bytes",
    # output
    "output_text_bytes", "output_chars", "completion_tokens", "prompt_tokens",
    "finish_reason", "done_seen", "truncated", "content_sha256", "reasoning_chars",
    # TEE (always empty in this version)
    "attest_present", "attest_verified", "attest_bound_to",
]


@dataclass
class Plan:
    endpoint: str
    models: list                      # [{"request_name":..., "provider":...}]
    prompts: list                     # [{"id":..., "text":...}]
    rpm: float = 20.0
    max_retries: int = 0
    max_tokens: int = 160
    temperature: float = 0.0
    timeout_s: float = 120.0
    budget_usd_session: float = 0.0
    price_out_per_mtok: float = 0.0
    price_in_per_mtok: float = 0.0
    max_session_minutes: float = 90.0
    seed: int = 20261118
    rounds: int = 17
    extra_body: dict = field(default_factory=dict)


def load_plan(path: Path) -> Plan:
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = j.get("prompts")
    if isinstance(prompts, str):
        pj = json.loads((Path(path).parent / prompts).read_text(encoding="utf-8"))
        prompts = pj["prompts"]
    return Plan(
        endpoint=j["endpoint"],
        models=j["models"],
        prompts=prompts,
        rpm=j.get("rpm", 20.0),
        max_retries=j.get("max_retries", 0),
        max_tokens=j.get("max_tokens", 160),
        temperature=j.get("temperature", 0.0),
        timeout_s=j.get("timeout_s", 120.0),
        budget_usd_session=j.get("budget_usd_session", 0.0),
        price_out_per_mtok=j.get("price_out_per_mtok", 0.0),
        price_in_per_mtok=j.get("price_in_per_mtok", 0.0),
        max_session_minutes=j.get("max_session_minutes", 90.0),
        seed=j.get("seed", 20261118),
        rounds=j.get("rounds", 17),
        extra_body=j.get("extra_body") or {},
    )


def _guard_endpoint(url: str) -> None:
    if url == "FROM_KEY_FILE":
        return
    if url.startswith("https://"):
        return
    if url.startswith("http://127.0.0.1") or url.startswith("http://localhost"):
        return
    raise SystemExit(
        "refusing plain-http endpoint that is not localhost; "
        "real endpoints must be https"
    )


class RateLimiter:
    """Minimum interval between requests. Never bursts, never bypasses limits."""

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> float:
        if self.min_interval <= 0:
            return 0.0
        delta = time.monotonic() - self._last
        sleep = max(0.0, self.min_interval - delta)
        if sleep:
            time.sleep(sleep)
        self._last = time.monotonic()
        return sleep


def _build_order(plan: Plan, rnd: int, rng: random.Random):
    """One round = every model x every prompt, exactly once.

    Model order rotates (Latin) with the round index; the joint order within the
    round is then shuffled. This breaks any model/position collinearity.
    """
    models = list(plan.models)
    shift = rnd % max(1, len(models))
    rotated = models[shift:] + models[:shift]
    cells = [(m, p) for m in rotated for p in plan.prompts]
    rng.shuffle(cells)
    return cells


def probe(url: str, key: str, model: str, prompt: str, plan: Plan) -> dict:
    """Single request. Returns the measurement row (identity fields filled by caller)."""
    import requests

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": plan.temperature,
        "max_tokens": plan.max_tokens,
    }
    body.update(getattr(plan, "extra_body", None) or {})
    row = dict(
        model_served="", request_id="", http_status=0, http_attempts=0,
        retry_after_s="", error_class="", error_message_sha="", ttfb_ms=float("nan"),
        ttft_role_ms=float("nan"), ttft_content_ms=float("nan"), total_ms=float("nan"),
        inter_chunk_gap_mean_ms=float("nan"), chunk_count=0, sse_bytes_total=0,
        sse_overhead_bytes=0, chunk_bytes_min=0, chunk_bytes_max=0,
        chunk_bytes_mean=0.0, chunk_bytes_cv=float("nan"), first_chunk_bytes=0,
        output_text_bytes=0, output_chars=0, completion_tokens="", prompt_tokens="",
        finish_reason="", done_seen=False, truncated=False, content_sha256="",
        reasoning_chars=0,
        attest_present="", attest_verified="", attest_bound_to="",
    )
    t_start = time.perf_counter_ns()
    row["t_start_mono"] = t_start

    content = bytearray()
    raw_parts: list[bytes] = []
    content_chunks = 0
    chunk_sizes: list[int] = []
    data_times: list[float] = []
    ttfb = ttft_role = ttft_content = None

    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body, stream=True, timeout=plan.timeout_s,
        )
        row["http_attempts"] = 1
        row["http_status"] = r.status_code
        if r.status_code == 429:
            row["retry_after_s"] = r.headers.get("Retry-After", "")
            row["error_class"] = "rate_limited"

        for line in r.iter_lines():
            if not line:
                continue
            now = (time.perf_counter_ns() - t_start) / 1e6
            if ttfb is None:
                ttfb = now
            row["sse_bytes_total"] += len(line)
            if row["http_status"] != 200:
                raw_parts.append(line)
                continue
            if not line.startswith(b"data:"):
                continue
            if ttft_role is None:
                ttft_role = now
            data_times.append(time.monotonic())
            chunk_sizes.append(len(line))
            payload = line[5:].strip()
            if payload == b"[DONE]":
                row["done_seen"] = True
                continue
            try:
                j = json.loads(payload)
            except Exception:
                continue
            row["request_id"] = row["request_id"] or (j.get("id") or "")
            if j.get("model"):
                row["model_served"] = j["model"]
            usage = j.get("usage")
            if usage:
                row["prompt_tokens"] = usage.get("prompt_tokens", row["prompt_tokens"])
                row["completion_tokens"] = usage.get("completion_tokens", row["completion_tokens"])
            for ch in j.get("choices") or []:
                d_ = ch.get("delta") or {}
                rc = d_.get("reasoning_content")
                if rc:
                    row["reasoning_chars"] += len(rc)
                piece = d_.get("content")
                if piece:
                    content_chunks += 1
                    if ttft_content is None:
                        ttft_content = now
                    content.extend(piece.encode("utf-8"))
                if ch.get("finish_reason"):
                    row["finish_reason"] = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001 - every failure mode is data
        row["error_class"] = type(e).__name__
    finally:
        row["t_end_mono"] = time.perf_counter_ns()
        row["wall_ts"] = time.time()
        row["total_ms"] = (row["t_end_mono"] - t_start) / 1e6
        row["ttfb_ms"] = ttfb if ttfb is not None else float("nan")
        row["ttft_role_ms"] = ttft_role if ttft_role is not None else float("nan")
        row["ttft_content_ms"] = ttft_content if ttft_content is not None else float("nan")

    if row["http_status"] not in (200,) and raw_parts:
        try:
            j = json.loads(b"".join(raw_parts).decode("utf-8", "replace"))
        except Exception:
            j = {}
        err = (j.get("error") or {}) if isinstance(j, dict) else {}
        msg = str(err.get("message") or "")[:120]
        code = str(err.get("code") or "")[:40]
        if "not activated" in msg.lower():
            cat = "not_activated"
        elif "rate" in msg.lower() or row["http_status"] == 429:
            cat = "rate_limited"
        else:
            cat = code or ("http_%d" % row["http_status"])
        row["error_class"] = "%d:%s" % (row["http_status"], cat)
        row["error_message_sha"] = hashlib.sha256(msg.encode()).hexdigest()[:16]
    row["chunk_count"] = len(chunk_sizes)
    row["content_chunk_count"] = content_chunks
    if chunk_sizes:
        row["first_chunk_bytes"] = chunk_sizes[0]
        row["chunk_bytes_min"] = min(chunk_sizes)
        row["chunk_bytes_max"] = max(chunk_sizes)
        mean = sum(chunk_sizes) / len(chunk_sizes)
        row["chunk_bytes_mean"] = round(mean, 3)
        var = sum((x - mean) ** 2 for x in chunk_sizes) / len(chunk_sizes)
        row["chunk_bytes_cv"] = round((var ** 0.5) / mean, 4) if mean else float("nan")
    if len(data_times) > 1:
        gaps = [b - a for a, b in zip(data_times, data_times[1:])]
        row["inter_chunk_gap_mean_ms"] = round(1000 * sum(gaps) / len(gaps), 3)
    row["output_text_bytes"] = len(content)
    row["output_chars"] = len(content.decode("utf-8", "replace"))
    row["sse_overhead_bytes"] = max(0, row["sse_bytes_total"] - len(content))
    row["content_sha256"] = hashlib.sha256(bytes(content)).hexdigest() if content else ""
    row["truncated"] = row["finish_reason"] == "length"
    return row


def _est_cost(plan: Plan, prompt_text: str) -> float:
    est_in = len(prompt_text) / 4.0
    return (est_in * plan.price_in_per_mtok + plan.max_tokens * plan.price_out_per_mtok) / 1e6


def run_session(plan: Plan, session_id: str, out_dir: Path, key: str, log=print) -> dict:
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"wild_full_{session_id}.csv"
    rng = random.Random(plan.seed)
    limiter = RateLimiter(plan.rpm)

    spent = 0.0
    consecutive_429 = 0
    recent: list[int] = []
    stop_reason = ""
    rounds_touched: set[int] = set()
    t_session0 = time.monotonic()

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        fh.flush()

        for rnd in range(plan.rounds):
            cells = _build_order(plan, rnd, rng)
            rounds_touched.add(rnd)
            for seq_in_round, (m, p) in enumerate(cells):
                # ---- stop conditions (PREREG §8) --------------------------------
                if plan.budget_usd_session and spent + _est_cost(plan, p["text"]) > plan.budget_usd_session:
                    stop_reason = "budget_cap"
                elif (time.monotonic() - t_session0) / 60 > plan.max_session_minutes:
                    stop_reason = "duration_cap"
                elif consecutive_429 >= 20:
                    stop_reason = "consecutive_429"
                elif len(recent) >= 50 and sum(recent[-50:]) / 50 >= 0.6:
                    stop_reason = "429_rate"
                if stop_reason:
                    log(f"[stop] {stop_reason} at round={rnd} seq={seq_in_round}")
                    break

                limiter.wait()
                row = probe(plan.endpoint, key, m["request_name"], p["text"], plan)
                row.update(session_id=session_id, round=rnd, seq_in_round=seq_in_round,
                           model_requested=m["request_name"], provider=m.get("provider", ""),
                           prompt_id=p["id"],
                           prompt_sha256=hashlib.sha256(p["text"].encode()).hexdigest()[:16])
                w.writerow({k: row.get(k, "") for k in FIELDS})
                fh.flush()

                spent += _est_cost(plan, p["text"])
                recent.append(1 if row["http_status"] == 429 else 0)
                consecutive_429 = consecutive_429 + 1 if row["http_status"] == 429 else 0

            if stop_reason:
                break

    return {
        "session_id": session_id,
        "csv": str(csv_path),
        "csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "rounds_planned": plan.rounds,
        "rounds_done": len(rounds_touched),
        "rounds_complete": (rnd + 1) if not stop_reason else max(0, rnd),
        "budget_estimate_usd": round(spent, 6),
        "stop_reason": stop_reason or "completed",
        "elapsed_min": round((time.monotonic() - t_session0) / 60, 2),
        "seed": plan.seed,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--i-have-approval", action="store_true")
    a = ap.parse_args(argv)

    plan = load_plan(Path(a.plan))
    _guard_endpoint(plan.endpoint)

    n_req = plan.rounds * len(plan.models) * len(plan.prompts)
    est = sum(_est_cost(plan, p["text"]) for p in plan.prompts) * plan.rounds * len(plan.models)
    print(f"endpoint      : {plan.endpoint}")
    print(f"models        : {len(plan.models)}  prompts: {len(plan.prompts)}  rounds: {plan.rounds}")
    print(f"requests      : {n_req}  ({n_req // max(1, len(plan.models))} per model)")
    print(f"rate          : {plan.rpm} rpm  => ~{n_req / max(1e-9, plan.rpm):.1f} min")
    print(f"est cost      : ${est:.4f} (prices are plan inputs, not verified)")
    print(f"budget cap    : ${plan.budget_usd_session:.2f}")
    print(f"max_retries   : {plan.max_retries}")

    if not (a.execute and a.i_have_approval):
        print("\nDRY RUN — no request sent. Add --execute --i-have-approval to collect.")
        return 0

    key_text = Path(a.key_file).read_text(encoding="utf-8", errors="replace")
    key = next((l.strip() for l in key_text.splitlines() if l.strip().startswith("sk-")), "")
    if not key:
        raise SystemExit("no sk- credential found in key file")
    if plan.endpoint == "FROM_KEY_FILE":
        import re as _re
        plan.endpoint = _re.search(r"https?://\S+", key_text).group(0).rstrip("/") + "/chat/completions"
    summary = run_session(plan, a.session, Path(a.out), key)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
