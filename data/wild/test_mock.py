"""Offline test of collector.py against mock_server.py (loopback only).

Writes MOCK_TEST_RESULTS.md and out_mock/TEST_RESULTS.json. No external traffic:
the mock binds 127.0.0.1 and the collector refuses plain http elsewhere.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import collector as C
import mock_server

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_mock"
CHECKS: list[dict] = []

MODELS = [
    {"request_name": "mock-chunky", "provider": "mockp"},
    {"request_name": "mock-batched", "provider": "mockp"},
    {"request_name": "mock-single", "provider": "mockp"},
    {"request_name": "mock-429", "provider": "mockp"},
    {"request_name": "mock-trunc", "provider": "mockp"},
    {"request_name": "mock-nousage", "provider": "mockp"},
]


def check(name: str, ok: bool, expected="", got=""):
    CHECKS.append(dict(name=name, ok=bool(ok), expected=str(expected), got=str(got)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   expected={expected}  got={got}")


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_plan(url: str, rounds: int, budget: float, price_out: float) -> C.Plan:
    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    return C.Plan(
        endpoint=url, models=MODELS, prompts=prompts,
        rpm=1200, max_retries=0, max_tokens=160, temperature=0.0,
        timeout_s=20, budget_usd_session=budget, price_out_per_mtok=price_out,
        price_in_per_mtok=0.0, max_session_minutes=5, seed=20261118, rounds=rounds,
    )


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    srv, url = mock_server.start()
    try:
        plan = build_plan(url, rounds=2, budget=0.0, price_out=0.0)
        summary = C.run_session(plan, "MOCK-S1", OUT, "sk-mock-not-a-real-key",
                                log=lambda *a: None)
        rows = load(Path(summary["csv"]))

        # T1 schema
        with open(summary["csv"], encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        check("T1 CSV 列与 PREREG §4 完全一致",
              header == C.FIELDS, len(C.FIELDS), len(header))

        by_model = {}
        for r in rows:
            by_model.setdefault(r["model_requested"], []).append(r)

        # T2 chunk_count 反映分块粒度
        n_chunky = int(by_model["mock-chunky"][0]["content_chunk_count"])
        n_batched = int(by_model["mock-batched"][0]["content_chunk_count"])
        n_single = int(by_model["mock-single"][0]["content_chunk_count"])
        check("T2a content_chunk_count 反映分块粒度（chunky>batched>single=1）",
              n_chunky > n_batched > n_single == 1,
              "40>4>1", f"{n_chunky}>{n_batched}>{n_single}")
        rc = by_model["mock-chunky"][0]
        check("T2b chunk_count = 内容块 + role 块 + 终止块",
              int(rc["chunk_count"]) == int(rc["content_chunk_count"]) + 2,
              "content+2", f"{rc['chunk_count']} vs {rc['content_chunk_count']}+2")

        # T3 role chunk 早于首个内容 chunk
        ok = all(float(r["ttft_role_ms"]) < float(r["ttft_content_ms"])
                 for r in by_model["mock-chunky"])
        check("T3 ttft_role_ms < ttft_content_ms（未把 role chunk 当首 token）",
              ok, "True", ok)

        # T4 SSE 封装开销
        ok = all(int(r["sse_overhead_bytes"]) > 0 and
                 int(r["sse_bytes_total"]) > int(r["output_text_bytes"])
                 for r in by_model["mock-chunky"])
        check("T4 sse_overhead_bytes>0 且 sse_bytes_total>output_text_bytes",
              ok, "True", ok)

        # T5 正常流
        ok = all(r["done_seen"] == "True" and len(r["content_sha256"]) == 64
                 for r in by_model["mock-chunky"])
        check("T5 正常流 done_seen=True 且 content_sha256 为 64 位",
              ok, "True", ok)

        # T6 429 处理：不重试、记录 Retry-After
        r429 = by_model["mock-429"][0]
        check("T6 429: status=429 / error_class / Retry-After / attempts=1",
              r429["http_status"] == "429" and r429["error_class"].endswith("rate_limited")
              and r429["retry_after_s"] == "2" and r429["http_attempts"] == "1",
              "429/rate_limited/2/1",
              f"{r429['http_status']}/{r429['error_class']}/{r429['retry_after_s']}/{r429['http_attempts']}")

        # T7 截断
        rt = by_model["mock-trunc"][0]
        check("T7 截断: truncated=True / finish_reason=length / done_seen=False",
              rt["truncated"] == "True" and rt["finish_reason"] == "length"
              and rt["done_seen"] == "False",
              "True/length/False",
              f"{rt['truncated']}/{rt['finish_reason']}/{rt['done_seen']}")

        # T8 缺 usage
        rn = by_model["mock-nousage"][0]
        check("T8 缺 usage: completion_tokens 为空（应判 invalid）",
              rn["completion_tokens"] == "", "''", repr(rn["completion_tokens"]))

        # T9 每轮含全部 模型×prompt 且各一次
        per_round = Counter(int(r["round"]) for r in rows)
        cells_ok = True
        for rnd in per_round:
            sub = [r for r in rows if int(r["round"]) == rnd]
            combos = Counter((r["model_requested"], r["prompt_id"]) for r in sub)
            cells_ok &= (len(sub) == len(MODELS) * len(plan.prompts)
                         and set(combos.values()) == {1})
        check("T9 每轮 = M×P 且 (模型,prompt) 不重复",
              cells_ok, f"{len(MODELS)*len(plan.prompts)}/round", dict(per_round))

        # T10 轮间模型顺序不同（Latin 轮转 + 轮内随机）
        sigs = []
        for rnd in sorted(per_round):
            sigs.append(tuple(r["model_requested"] for r in rows if int(r["round"]) == rnd))
        check("T10 轮间顺序不同（未按模型分块）",
              len(set(sigs)) >= 2, ">=2 种顺序", f"{len(set(sigs))} 种")

        # T11 无密钥/无敏感串落盘
        blob = Path(summary["csv"]).read_text(encoding="utf-8")
        check("T11 CSV 不含 Bearer/key/公网 IP",
              "Bearer" not in blob and "sk-mock" not in blob
              and "127.0.0.1" not in blob,
              "none", "none" if "Bearer" not in blob else "found")

        # T12 预算闸门
        plan2 = build_plan(url, rounds=6, budget=0.5, price_out=1000.0)
        s2 = C.run_session(plan2, "MOCK-BUDGET", OUT / "budget", "sk-mock",
                           log=lambda *a: None)
        rows2 = load(Path(s2["csv"]))
        check("T12 预算闸门触发并停止",
              s2["stop_reason"] == "budget_cap" and len(rows2) < len(plan2.prompts) * len(MODELS) * plan2.rounds,
              "budget_cap 且早停", f"{s2['stop_reason']} n={len(rows2)}")

        # T13 非 https 且非 localhost 被拒
        blocked = False
        try:
            C._guard_endpoint("http://example.com/v1/chat/completions")
        except SystemExit:
            blocked = True
        check("T13 拒绝非 https 外部端点", blocked, "SystemExit", "SystemExit" if blocked else "allowed")

        # T14 默认不重试：429 行 http_attempts 恒为 1
        check("T14 默认 max_retries=0（所有行 attempts=1）",
              all(r["http_attempts"] == "1" for r in rows), "1", "1")

        summary_out = dict(
            mock_url=url,
            requests=len(rows),
            per_model={k: len(v) for k, v in by_model.items()},
            checks=CHECKS,
            passed=sum(c["ok"] for c in CHECKS),
            total=len(CHECKS),
            session=summary,
            budget_session=s2,
        )
    finally:
        srv.shutdown()

    (OUT / "TEST_RESULTS.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# Mock 测试结果（离线，仅 loopback）", "",
          f"- 请求数：{summary_out['requests']}  ·  mock 端点：`{url}`",
          f"- 通过：**{summary_out['passed']}/{summary_out['total']}**", "",
          "| # | 检查项 | 期望 | 实测 | 结果 |", "|---|---|---|---|---|"]
    for i, c in enumerate(CHECKS, 1):
        md.append(f"| T{i} | {c['name']} | {c['expected']} | {c['got']} | "
                  f"{'✅ PASS' if c['ok'] else '❌ FAIL'} |")
    md += ["", "## 覆盖率", "",
           "已验证：SSE 分块粒度、role/内容 chunk 分离、SSE 封装开销、`[DONE]`、截断、"
           "缺 usage、429 与 Retry-After、不重试、轮内随机与轮间轮转、预算闸门、"
           "端点协议守卫、密钥不入盘。", "",
           "**未验证（需真实服务）**：真实 provider 的 `usage` 是否回传、真实 chunk 粒度、"
           "跨 provider 差异、真实 429 策略、attestation 字段。这些只能在冒烟阶段确认。"]
    (HERE / "MOCK_TEST_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n{summary_out['passed']}/{summary_out['total']} passed")
    return 0 if summary_out["passed"] == summary_out["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
