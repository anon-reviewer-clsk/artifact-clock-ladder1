# Artifact: Where the Clock Is Read

Anonymous artifact accompanying the submission *"Where the Clock Is Read: A
Timing-Channel Ladder from SGX Enclaves to Production Trust Domains"*
(USENIX Security '27 Cycle 2).

This package contains the collection harnesses, the analysis scripts, and the
per-request records from which every number in the paper is computed. It is
intended to let a reader recompute the tables and to re-run the measurement
arms on comparable hardware.

---

## 1 Layout

```
code/
  collection/   interleaved collectors, session splitting, input freezing
  analysis/     TVLA suite, MI/Fano, SPRT, criterion and cross-session figures
  td_shape/     TD-side serving harness, quantization/serving drivers,
                constructed early-exit and MoE model builders,
                make_appendix_tables.py  (regenerates Appendix Tables 17-21)
  uncore/       bare-metal collector (M2M PKT_MATCH programming), victim,
                stage-1/stage-2 drivers, stage-2 analysis
data/
  fleet_cal/    nine-model calibration fleet (CinC2017 ECG classifiers)
  legs/         seven-endpoint real-model legs (loopback / WAN / TDX)
  td_shape/     request-level JSON/JSONL for the TD shape experiments
  uncore/       bare-metal PMC records (stage 0-2)
  wild/         production-gateway collections and protocol amendments
prereg/         pre-registration documents and branch outcomes
docs/           experiment-to-file map (docs/experiments.md)
MANIFEST.sha256 sha256 of every file in this package
```

## 2 What was removed before publication

| Removed | Why |
|---|---|
| `resp_body` column in every leg CSV | response bodies are not released (see the paper's Open Science appendix); the self-reported value the paper analyses is retained as `enclave_us` |
| credentials, API keys, SSH keys | never included |
| model checkpoints | large; see the licence note below |
| vendor SDK binaries | third-party licensed |

Local absolute paths in scripts and logs were replaced with `<home>` and
`<data-root>` placeholders. Scripts expect input paths to be supplied on the
command line or edited to your own layout.

## 3 Environment

Measured and analysed on Linux (Ubuntu 20.04 / 24.04) with Python 3.8-3.12.

| Component | Requirement |
|---|---|
| `code/collection`, `code/analysis` | Python 3, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` |
| `code/td_shape` serving | `onnxruntime` (CPU EP), `numpy`; the MoE builder needs `onnx` |
| SGX arms | Intel SGX SDK + PSW/DCAP runtime on an SGX-capable Xeon |
| TD arms | a genuine TD (`tdx_guest` in `lscpu`); quotes via `/sys/kernel/config/tsm/report` |
| `code/uncore` | Ice Lake-SP class host, `perf` with uncore events, `setpci`, root |

The uncore collector programs the undocumented M2M `PKT_MATCH` filter through
PCI configuration space (registers `0x490`-`0x4a0` on devices `fe:0c.0`-`fe:0f.0`)
and binds its victim with `numactl --membind=1`; the exposed counter units sit on
the second NUMA node on our machine. Reproducing it requires the same topology.

## 4 Reproducing the tables

| Paper item | Command |
|---|---|
| Table 1 (clock ladder, 9 classes) | `python3 code/analysis/review_stats.py --in data/fleet_cal/timing_multi_model.csv` |
| Tables 3, 5, 8 (real-model legs) | `python3 code/analysis/cross_session.py --in data/legs/` |
| TVLA / MI results in S3 | `python3 code/analysis/tvla_suite.py --in data/legs/` |
| Criterion and cross-session figures | `python3 code/analysis/make_fig_criterion.py`, `make_fig_cross_session.py` |
| Appendix Tables 17-21 (TD shape) | `python3 code/td_shape/make_appendix_tables.py > tables.tex` |
| S10 uncore identification | `python3 code/uncore/analyze_phase2.py` (reads `data/uncore/stage2.tsv`) |

Scripts accept `--help`; the TOML/JSON schema notes for each leg are in
`docs/experiments.md`.

## 5 Data schema (leg CSVs)

`scenario, model_id, rep, sample_id, true_class, enclave_us, roundtrip_us, wall_ts, resp_bytes`

* `enclave_us` - time read at the enclave/service boundary (P1/P2 position);
* `roundtrip_us` - client-observed round trip (P3 position);
* `resp_bytes` - response size in bytes (the length channel);
* `true_class` - served-model label used for supervised evaluation only.

No timestamps resolve to individual users, and no request content is retained.

## 6 Licence

Source code in this package: MIT. Self-trained checkpoints referenced by the
paper (DSV1 tokenizers and downstream, TriDet, ActionFormer) were trained on
ACOUSLIC-AI (CC-BY-NC-SA 4.0) and are released for non-commercial research use
only; they are not part of this archive. Third-party models (ResNet-18,
BioMedCLIP, Mixtral) keep their original licences.

## 7 Ethics

All measurements were taken on infrastructure we controlled, plus an authorised
account on one production aggregation gateway. Only the authors' own requests
were collected; response bodies are not included. No human subjects are involved.
