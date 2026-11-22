#!/usr/bin/env python3
"""build_moe.py — 合成 MoE：路由由输入决定，专家总集大于 L3。

结构：
  logits = x @ R                    (R: [D,E])
  idx    = TopK(logits, k)
  y      = Σ_{j∈idx} x @ W[j]       (W: [E,D,D]，只算被选中的 k 个)

模型暴露两个输出：y [1,D] 与 idx [1,k]（后者作为路由的 ground truth）。
D=2048, E=32, k=2 → 专家总集 32×16 MiB = 512 MiB（L3 为 320 MiB，故意超出）。
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

D = 2048
E = 32
K = 2


def main():
    rng = np.random.default_rng(20261122)
    R = (rng.standard_normal((D, E)).astype(np.float32) / np.float32(np.sqrt(D))).astype(np.float32)
    W = np.stack([(rng.standard_normal((D, D)).astype(np.float32)
                   / np.float32(np.sqrt(D))).astype(np.float32) for _ in range(E)])

    nodes = [
        helper.make_node("MatMul", ["x", "R"], ["logits"]),
        helper.make_node("TopK", ["logits", "kk"], ["vals", "idx"], axis=1,
                         largest=1, sorted=1),
        helper.make_node("Gather", ["idx", "zero"], ["i0"], axis=1),   # [1]
        helper.make_node("Gather", ["idx", "one"], ["i1"], axis=1),
        helper.make_node("Gather", ["W", "i0"], ["w0"], axis=0),       # [1,D,D]
        helper.make_node("Gather", ["W", "i1"], ["w1"], axis=0),
        helper.make_node("Squeeze", ["w0", "ax0"], ["w0s"]),           # [D,D]
        helper.make_node("Squeeze", ["w1", "ax0"], ["w1s"]),
        helper.make_node("MatMul", ["x", "w0s"], ["y0"]),
        helper.make_node("MatMul", ["x", "w1s"], ["y1"]),
        helper.make_node("Add", ["y0", "y1"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "moe_synth",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, D]),
         helper.make_tensor_value_info("idx", TensorProto.INT64, [1, K])],
        initializer=[numpy_helper.from_array(R, "R"),
                     numpy_helper.from_array(W, "W"),
                     numpy_helper.from_array(np.array([K], np.int64), "kk"),
                     numpy_helper.from_array(np.array(0, np.int64), "zero"),
                     numpy_helper.from_array(np.array(1, np.int64), "one"),
                     numpy_helper.from_array(np.array([0], np.int64), "ax0")])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    p = "/root/models/moe_synth.onnx"
    onnx.save(model, p)
    import os
    print(f"saved {p}  {os.path.getsize(p)/1e6:.0f} MB  experts={E} k={K} D={D}")

    # 自检：不同输入是否路由到不同专家
    import time
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])
    for name, g in [("rand", np.random.default_rng(1).standard_normal((1, D))),
                    ("checker", (np.arange(D).reshape(1, D) % 2 * 2 - 1).astype(np.float32)),
                    ("ones", np.ones((1, D)))]:
        x = g.astype(np.float32)
        s.run(None, {"x": x})
        t = time.perf_counter()
        for _ in range(5):
            y, idx = s.run(None, {"x": x})
        dt = (time.perf_counter() - t) / 5 * 1e3
        print(f"  {name:8s} idx={idx.ravel().tolist()}  {dt:6.2f} ms  finite={np.isfinite(y).all()}")


if __name__ == "__main__":
    main()
