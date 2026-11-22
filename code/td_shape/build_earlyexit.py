#!/usr/bin/env python3
"""build_earlyexit.py — 构造一个"计算量依赖输入"的 ONNX 模型。

机制（等价于 early-exit 的难度决定深度）：
  depth = clip(ceil(max(|x|) * 10), 1, 60)
  y = x
  重复 depth 次: y = y @ W      (W: [1024,1024]，共享)
输出 y。

于是：输入幅度越大 → 迭代次数越多 → 计算量越大。
这不是"读取静态数组"，而是真实的、随输入变化的前向传播。
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

D = 2048
MAXIT = 60


def main():
    rng = np.random.default_rng(20261121)
    W = (rng.standard_normal((D, D)).astype(np.float32) / np.float32(np.sqrt(D))).astype(np.float32)

    body_in = [helper.make_tensor_value_info("iter", TensorProto.INT64, []),
               helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
               helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [1, D])]
    body_out = [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
                helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [1, D])]
    body = helper.make_graph(
        [helper.make_node("MatMul", ["v_in", "W"], ["v_out"]),
         helper.make_node("Identity", ["cond_in"], ["cond_out"])],
        "body", body_in, body_out,
        initializer=[numpy_helper.from_array(W, "W")])

    nodes = [
        helper.make_node("Abs", ["x"], ["ax"]),
        helper.make_node("ReduceMax", ["ax"], ["mx"], keepdims=0),
        helper.make_node("Mul", ["mx", "ten"], ["m10"]),
        helper.make_node("Ceil", ["m10"], ["mc"]),
        helper.make_node("Max", ["mc", "one"], ["m1"]),
        helper.make_node("Min", ["m1", "cap"], ["mn"]),
        helper.make_node("Cast", ["mn"], ["trip"], to=TensorProto.INT64),
        helper.make_node("Constant", [], ["cond0"],
                         value=helper.make_tensor("c", TensorProto.BOOL, [], [True])),
        helper.make_node("Loop", ["trip", "cond0", "x"], ["y"], body=body),
    ]
    graph = helper.make_graph(
        nodes, "earlyexit_dynamic_depth",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, D])],
        initializer=[numpy_helper.from_array(np.array(10.0, np.float32), "ten"),
                     numpy_helper.from_array(np.array(1.0, np.float32), "one"),
                     numpy_helper.from_array(np.array(float(MAXIT), np.float32), "cap")])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, "/root/models/earlyexit.onnx")
    print("saved /root/models/earlyexit.onnx")

    # 本地自检：不同幅度 → 不同耗时
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession("/root/models/earlyexit.onnx", sess_options=so,
                             providers=["CPUExecutionProvider"])
    import time
    for mag in (0.02, 0.1, 0.3, 0.6):
        x = (np.random.default_rng(1).standard_normal((1, D)) * mag).astype(np.float32)
        s.run(None, {"x": x})
        t = time.perf_counter()
        for _ in range(5):
            y = s.run(None, {"x": x})[0]
        dt = (time.perf_counter() - t) / 5 * 1e3
        trip = min(MAXIT, max(1, int(np.ceil(np.abs(x).max() * 10))))
        print(f"  max|x|={np.abs(x).max():.3f} trip={trip:2d}  {dt:7.2f} ms  finite={np.isfinite(y).all()}")


if __name__ == "__main__":
    main()
