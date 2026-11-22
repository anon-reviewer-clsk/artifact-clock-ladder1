#!/usr/bin/env python3
"""b_tiny_moe.py — B 步：验证真实 MoE 的路由是否随输入变化、以及能否导出 ONNX。

用 hf-internal-testing 的 tiny-random Mixtral（权重随机但是真实 MoE 结构）。
只做两件事：
  1) 同一模型、不同 token 输入 → 打印每层 router 的 top-k 专家 id；
  2) 尝试 torch.onnx.export，报告成功/失败与原因。
不做时延结论（tiny 模型全在 cache，注定测不出）。
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

NAME = os.environ.get("TINY_MOE", "hf-internal-testing/tiny-random-MixtralForCausalLM")


def main():
    print("loading", NAME, flush=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, attn_implementation="eager").eval()
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    cfg = AutoConfig.from_pretrained(NAME)
    ne = getattr(cfg, "num_local_experts", None)
    tk = getattr(cfg, "num_experts_per_tok", None)
    print(f"config: experts={ne} top_k={tk} hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}")

    # 用 output_router_logits 直接拿 router logits，不 monkey-patch
    texts = ["the quick brown fox", "SELECT * FROM users WHERE id = 1"]
    for t in texts:
        ids = tok(t, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids, output_router_logits=True)
        rl = getattr(out, "router_logits", None)
        print(f"\ninput={t!r} tokens={ids['input_ids'].shape[1]} layers={0 if rl is None else len(rl)}")
        if rl:
            for li in (0, len(rl) - 1):
                g = rl[li]
                idx = torch.topk(g, tk, dim=-1).indices
                print(f"  layer{li}: 前 3 个 token 的专家 {idx[:3].tolist()}")

    # 尝试导出
    print("\n--- ONNX export ---", flush=True)
    ids = tok("hello", return_tensors="pt")
    os.makedirs("/root/a3/moe_onnx", exist_ok=True)
    try:
        torch.onnx.export(
            model, (ids["input_ids"],), "/root/a3/moe_onnx/tiny_mixtral.onnx",
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"}, "logits": {0: "b", 1: "s"}},
            opset_version=17, do_constant_folding=False, dynamo=False)
        sz = os.path.getsize("/root/a3/moe_onnx/tiny_mixtral.onnx")
        print(f"export OK  {sz/1e6:.1f} MB")
        import onnxruntime as ort
        s = ort.InferenceSession("/root/a3/moe_onnx/tiny_mixtral.onnx",
                                 providers=["CPUExecutionProvider"])
        a = s.run(None, {"input_ids": ids["input_ids"].numpy()})[0]
        ids2 = tok("SELECT 1", return_tensors="pt")
        b = s.run(None, {"input_ids": ids2["input_ids"].numpy()})[0]
        same = bool((a.shape == b.shape) and (a == b).all()) if a.shape == b.shape else False
        print(f"ORT 可跑，两个不同输入的输出相同? {same}（应为 False）")
    except Exception as e:
        print(f"export FAILED: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
