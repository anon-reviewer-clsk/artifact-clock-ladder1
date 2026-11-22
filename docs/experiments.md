# Experiment-to-file map

Each row: what the paper reports, the script that produces it, the data that
feeds it, and where it appears. Nothing here is a re-analysis: the scripts are
the ones used for the submission.

---

## E1 Clock-position ladder (9-model calibration fleet)

| | |
|---|---|
| Paper | §3, Table 1 |
| Script | `code/collection/collect_interleaved.py`, `code/analysis/review_stats.py` |
| Data | `data/fleet_cal/timing_multi_model.csv` (9 models x 3 architectures x hidden 16/32/64) |
| Units | request; 990 requests per model per leg, layout-balanced Latin slots |
| Chance | 1/9 = 0.1111 |
| Note | `enclave_us` is the P2 position, `roundtrip_us` the P3 position; the WAN row is stored in the same file with its own scenario label |

## E2 Real-model legs (7 endpoints)

| | |
|---|---|
| Paper | §5, Tables 3, 5, 8 |
| Script | `code/collection/cross_session.py`, `code/analysis/review_stats.py` |
| Data | `data/legs/*.csv` (loopback plain, loopback SGX, WAN, TDX WAN, TDX loopback, WAN replication) |
| Units | request; two collection sessions per leg, separated by ~1 day |
| Chance | 1/7 = 0.1429 |

## E3 TVLA and mutual information

| | |
|---|---|
| Paper | §3 (WAN row), §5 |
| Script | `code/analysis/tvla_suite.py` |
| Data | `data/legs/` |
| Method | Welch t (orders 1-4) + two-sample KS, Bonferroni within family; Fano lower bound on MI |
| Note | the KS family, not the moment tests, is what surfaces the left-shoulder signature |

## E4 REMOTEID

| | |
|---|---|
| Paper | §7, Tables 6-8 |
| Script | `code/analysis/framework_core.py`, `framework_open_defense.py`, `summarize_results.py` |
| Data | `data/legs/` |
| Method | B = 7 bounded statistics; empirical-Bernstein confidence sequences on geometric checkpoints; disjoint threshold and CI rounds; abstention on non-separation |

## E5 In-the-wild packet structure (12 endpoints)

| | |
|---|---|
| Paper | §6, Table 9 |
| Script | `code/collection/collect_interleaved.py` (gateway mode) |
| Data | `data/wild/` (two collections ~1 h apart, 1,224 requests each) |
| Chance | 1/12 = 0.0833 |
| Note | first-chunk size dropped after a smoke test showed it encodes the returned model name |

## E6 Bare-metal uncore (stages 0-2)

| | |
|---|---|
| Paper | §10, Table 14 |
| Script | `code/uncore/mid_run_dual.sh`, `ort_victim.py`, `stage1_driver.py`, `stage2_driver.py`, `analyze_phase2.py` |
| Data | `data/uncore/stage0_measurements.csv` (address-filter selectivity), `stage1_all.csv` (24 inference runs), `stage2.tsv` (32 runs, four models) |
| Hardware | Ice Lake-SP; M2M `PKT_MATCH` programmed at 64 B granularity; victim bound to NUMA node 1 |
| Units | run (inference process), not counter sample |

## E7 TD shape controls (six arms)

| | |
|---|---|
| Paper | §8, Table 16; per-block records in Appendix Tables 17-21 |
| Scripts | `code/td_shape/a3_run.py` (quantization, loopback), `wan_run.sh` + `wan_client.py` (WAN legs), `wan_pair.sh` (near-null), `a3_input_run.py` (input families), `build_earlyexit.py` + `a3_ee_run.py`, `build_moe.py` + `a3_moe_run.py`, `b_tiny_moe.py` (real tiny MoE, routing only) |
| Data | `data/td_shape/raw/a3_raw.json` (loopback), `data/td_shape/wan_results_quant.jsonl` (WAN), `data/td_shape/wan_results_nearnull.jsonl`, `raw/input_raw.json`, `raw/ee_raw.json`, `raw/moe_raw.json` |
| Units | block of 60 requests (30 for the shape arms); blocks are the clustered unit, requests are not IID |
| Regenerate tables | `python3 code/td_shape/make_appendix_tables.py` |

### Parameters worth copying

* serving: ONNX Runtime, `intra_op_num_threads = 1`, `inter_op = 1`, CPU EP;
* WAN leg: SSH tunnel, ping baseline 66-69 ms, measured from the same client;
* early-exit model: `trip = clip(ceil(10*max|x|), 1, 60)` over a `Loop` of 2048x2048 MatMuls;
* MoE model: 32 experts of 2048x2048 (537 MB), top-2 routing from `x @ R`, all four M2M boxes programmed per run;
* quote arm: fixed 64-byte nonce through ConfigFS TSM; quotes are 5,010 bytes and byte-identical for an unchanged configuration.

## E8 Artefacts not included

| Item | Reason |
|---|---|
| model checkpoints | size; see README licence note |
| `resp_body` column | response bodies are not released |
| raw server-side diagnostic logs | discarded when the cloud instance was released; the request-level records above are complete |
| ONNX export of a real MoE | the export path failed for the evaluator we used; the routing check is reproducible from `b_tiny_moe.py` |
