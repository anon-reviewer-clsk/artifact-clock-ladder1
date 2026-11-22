"""Order-balanced (interleaved) Block-4 collector.

WHY THIS EXISTS
---------------
The original collectors (collect_common.collect) iterate models in contiguous
blocks: all 4000 samples of cfc_h16, then all of cfc_h32, ... Any drift in host or
transport state is therefore perfectly aliased with model_id, and a classifier can
"identify" a model from transport-only features that contain no enclave compute.
results/network_attack/nshot_confound_probe.csv shows this is not hypothetical:
for S4, an overhead-only feature (roundtrip - enclave, uncorrelated with enclave
time) reproduces the published n-shot curve (0.481 at n=50 vs 0.440 for roundtrip).

This collector breaks the block/time collinearity: in each round every model is
sampled once, in a freshly shuffled order, so at any point in wall-clock time all
nine models are equally likely to be observed. Anything that survives this
randomisation is a property of the model, not of the collection slot.

USAGE
-----
S2 (loopback, same host as App.exe; start App.exe and pick menu 'W'):
    python collect_interleaved.py --scenario S2_interleaved \
        --base-url http://localhost:8000 \
        --out <data-root>/Results/timing_s2_interleaved.csv

S3 (simulated LAN). --jitter-inside puts the injected sleep INSIDE the measured
window; the original S3 slept before t_start, so its jitter never appeared in the
observable at all (see FINDINGS_order_confound.md).
    python collect_interleaved.py --scenario S3_interleaved \
        --base-url http://localhost:8000 \
        --out <data-root>/Results/timing_s3_interleaved.csv \
        --jitter-us 200 500 --jitter-inside

S4 (real WAN) -- run on the remote box, scp this single file plus optionally the
small test-meta CSV for labels:
    NGROK_URL=https://<hash>.ngrok-free.app python3 collect_interleaved.py \
        --scenario S4_interleaved --base-url "$NGROK_URL" \
        --out timing_s4_interleaved.csv --timeout 30 --rounds 99 --per-round 10

Cost note: interleaving means one /api/switch_model per model per round (9 per
round, 891 over the default 99 rounds) versus 9 for the whole original Block-4
run. Budget the switch latency, especially over WAN; --rounds is the knob that
trades cost for balance, and must stay a multiple of 9 to keep slot balance exact.

Output schema is a superset of the Block-4 files, so the existing analysis
scripts consume it unchanged (they select columns by name):
    scenario,model_id,rep,sample_id,true_class,enclave_us,roundtrip_us,
    wall_ts,resp_bytes,resp_body
`wall_ts`, `resp_bytes` and `resp_body` were added later and are absent/-1 in
older legs. `resp_bytes` = len(r.content), the body bytes the client actually
received -- it turns the byte-length channel from a reconstruction into a
measurement; `resp_body` keeps those bytes so the length model can be checked
against the real format string rather than assumed.
`rep` now carries the ROUND index (models are interleaved within a round), and
within-round collection order is recoverable from row order.

Model order is NOT reshuffled independently per round (default --order-schedule
latin): a base order is rotated so that within every 9-round cycle each model
sits in each within-round slot exactly once. Independent shuffling leaves a
residual slot imbalance of ~0.8 slots at realistic round counts, and against a
monotone drift that is enough for a classifier to find the label in transport
noise alone (measured in test_interleaved_mock.py).

Runs are resumable: if --out already exists, completed (round, model, sample_id)
triples are skipped. The order and the sample picks are seeded per (round, model)
rather than drawn from one global stream, so a resumed run reproduces exactly the
schedule an uninterrupted run would have used. Progress is flushed per model.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):
        return x

MODELS = ["cfc_h16", "cfc_h32", "cfc_h64",
          "lstm_h16", "lstm_h32", "lstm_h64",
          "mlp_h16", "mlp_h32", "mlp_h64"]

COLUMNS = ["scenario", "model_id", "rep", "sample_id",
           "true_class", "enclave_us", "roundtrip_us", "wall_ts",
           "resp_bytes", "resp_body"]

# Columns added after the first legs were collected. A prior file that lacks only
# these is still usable (they are filled NaN); anything else means a real schema
# change and the file is not a prefix of the current one.
OPTIONAL_COLUMNS = ("wall_ts", "resp_bytes", "resp_body")

DEFAULT_TEST_META = Path(r"<data-root>/SampleEnclave/x64/Debug/cinc2017_test_meta.csv")


def load_gap_series(path: Path, n_needed: int, seed: int) -> list[float]:
    """Empirical inter-request gaps (seconds) replayed from an existing leg.

    Used to test a *duty cycle* as the cause of an accuracy difference: the same
    server, same machine, loopback, but idle for the same distribution of pauses
    the real WAN leg saw. --jitter-us only matches the mean rate; long-tail pauses
    are the part that plausibly exposes timer coalescing, and they are what a
    matched-mean control throws away.

    Sampled without replacement so the subsample keeps the source distribution's
    mean and tail. The recorded order is dropped deliberately: the model that
    follows a pause is decided by this run's own schedule, not the source's.
    """
    src = pd.read_csv(path)
    if "wall_ts" not in src:
        raise SystemExit(f"[il] {path} has no wall_ts column, cannot replay gaps")
    gaps = src.sort_values("wall_ts")["wall_ts"].diff().dropna().to_numpy(float)
    gaps = gaps[gaps > 0]
    if len(gaps) < n_needed:
        raise SystemExit(f"[il] {path} yields only {len(gaps)} gaps, need {n_needed}")
    picks = random.Random(f"{seed}-gaps").sample(list(gaps), n_needed)
    print(f"[il] gap replay from {path}: n={len(picks)} "
          f"p10={np.percentile(picks, 10):.3f}s med={np.median(picks):.3f}s "
          f"p90={np.percentile(picks, 90):.3f}s max={max(picks):.1f}s "
          f"mean rate={1.0 / np.mean(picks):.2f} req/s")
    return picks


def load_true_map(path: Optional[Path]) -> dict[int, int]:
    """sample_id -> true_label, verified against Block-3 S1 data (100/100 agree).

    Absent file (e.g. running on the remote WAN box) is not an error: labels are
    unused by the confound test, which keys on model_id.
    """
    if path is None or not Path(path).exists():
        print(f"[il] no test meta at {path}; true_class will be -1")
        return {}
    meta = pd.read_csv(path).set_index("index")["true_label"]
    return {int(k): int(v) for k, v in meta.items()}


def make_session() -> requests.Session:
    s = requests.Session()
    s.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0))
    s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0))
    return s


def wait_healthy(sess: requests.Session, base_url: str, timeout: float,
                 attempts: int = 5) -> None:
    for i in range(attempts):
        try:
            h = sess.get(f"{base_url}/health", timeout=timeout)
            if h.status_code == 200:
                print(f"[il] health OK, round-trip "
                      f"{h.elapsed.total_seconds() * 1000:.1f} ms")
                return
            print(f"[il] health -> {h.status_code}")
        except Exception as e:
            print(f"[il] health attempt {i + 1} failed: {e}")
        time.sleep(3)
    raise RuntimeError(f"[il] {base_url}/health never came up, aborting")


def switch(sess: requests.Session, base_url: str, model_id: str,
           timeout: float, attempts: int = 3) -> bool:
    for i in range(attempts):
        try:
            r = sess.post(f"{base_url}/api/switch_model",
                          json={"model_id": model_id},
                          timeout=max(30.0, timeout * 2))
            if r.status_code == 200:
                return True
            print(f"[il] switch {model_id} -> {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[il] switch {model_id} attempt {i + 1} failed: {e}")
        time.sleep(2)
    return False


def warmup(sess: requests.Session, base_url: str, model_id: str,
           n: int, n_samples: int, timeout: float) -> None:
    """Untimed requests right after a switch, so weight-load/EPC paging cost
    doesn't land in the first logged sample.

    The original collectors did 50 once per contiguous block. Here every model
    is switched once per round, so the cold-start cost is symmetric across
    models by construction; a small warmup removes it without buying back the
    block/time collinearity. Raise --warmup for strict parity with Block 4.
    """
    for i in range(n):
        try:
            sess.post(f"{base_url}/api/timing_query",
                      json={"model_id": model_id, "sample_id": i % n_samples},
                      timeout=timeout)
        except Exception:
            pass


def timed_query(sess: requests.Session, base_url: str, model_id: str,
                sample_id: int, timeout: float, jitter_s: float,
                jitter_inside: bool) -> tuple[int, float, int, str]:
    """Return (enclave_us, roundtrip_us, resp_bytes, resp_body). Failed requests
    report -1 and are still timed, matching the original collector's behaviour;
    the merge step drops rows with roundtrip_us <= 0.

    resp_bytes is len(r.content) and resp_body the same bytes decoded -- what the
    client actually received, not a reconstruction from the parsed fields. That is
    what an off-path observer sees as payload volume, so it is the measured version
    of the "deleting the value leaves its byte width behind" claim
    (DEFENSE_attested_timer.md, claim 2). The body is kept so the length model can
    be checked against the real format string instead of assumed.
    """
    pre = jitter_s if (jitter_s > 0 and not jitter_inside) else 0.0
    if pre:
        time.sleep(pre)

    t0 = time.perf_counter_ns()
    if jitter_s > 0 and jitter_inside:
        time.sleep(jitter_s)

    enclave_us = -1
    resp_bytes = -1
    resp_body = ""
    try:
        r = sess.post(f"{base_url}/api/timing_query",
                      json={"model_id": model_id, "sample_id": sample_id},
                      timeout=timeout)
        if r.status_code == 200:
            enclave_us = int(r.json().get("enclave_us", -1))
            resp_bytes = len(r.content)
            resp_body = r.content.decode("utf-8", "replace")
    except Exception:
        pass
    return (enclave_us, (time.perf_counter_ns() - t0) / 1000.0,
            resp_bytes, resp_body)


def load_prior(out_csv: Path) -> pd.DataFrame:
    """Rows already on disk from an interrupted run.

    Every flush must carry them forward: writing only the current run's rows
    would overwrite the partial file we are resuming and silently delete it.
    """
    empty = pd.DataFrame(columns=COLUMNS)
    if not out_csv.exists():
        return empty
    try:
        prev = pd.read_csv(out_csv)
    except Exception as e:
        raise RuntimeError(
            f"[il] {out_csv} exists but cannot be read ({e}); refusing to "
            f"overwrite it. Move it aside or point --out at a new file.") from e
    missing = [c for c in COLUMNS if c not in prev.columns]
    if missing:
        # Rows collected before these columns existed are still usable; dropping
        # the whole file here would silently delete a finished run on the first
        # flush, and there is no VCS in this repo to get it back.
        real = [c for c in missing if c not in OPTIONAL_COLUMNS]
        if real:
            raise RuntimeError(
                f"[il] existing {out_csv} lacks columns {real}; refusing to "
                f"overwrite a run this script cannot read. Point --out at a new "
                f"file instead.")
        print(f"[il] existing {out_csv} predates {missing}; filling NaN")
        for c in missing:
            prev[c] = float("nan")
    if len(prev):
        print(f"[il] resume: {len(prev)} rows already in {out_csv}")
    return prev


def latin_order(base_order: list[str], rnd: int) -> list[str]:
    """Rotate by the actual fleet size, not by a fleet-specific constant."""
    if not base_order:
        raise ValueError("latin rotation requires at least one model")
    offset = rnd % len(base_order)
    return base_order[offset:] + base_order[:offset]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=os.environ.get("NGROK_URL", "http://localhost:8000"))
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=99,
                    help="interleaved rounds; each model gets --per-round samples per "
                         "round (default 99 = 11 complete 9-round balance cycles)")
    ap.add_argument("--per-round", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=100,
                    help="sample_id pool size (server-side test set)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed requests after every model switch")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--jitter-us", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                    help="uniform sleep per request, in microseconds")
    ap.add_argument("--jitter-inside", action="store_true",
                    help="count the injected sleep inside the measured round-trip")
    ap.add_argument("--gap-file", type=Path, default=None,
                    help="replay the idle-pause distribution recorded in this leg's CSV "
                         "(needs a wall_ts column) instead of --jitter-us: matches the "
                         "gaps' tail, not just the mean request rate")
    ap.add_argument("--order-schedule", choices=["latin", "shuffle"], default="latin",
                    help="latin = one shuffled base order rotated every round, so "
                         "each model sits in each position exactly once per 9 rounds "
                         "(default). shuffle = independent reshuffle per round, which "
                         "leaks residual within-round-position signal at low round "
                         "counts.")
    ap.add_argument("--test-meta", type=Path, default=DEFAULT_TEST_META)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args(argv)

    if a.per_round > a.n_samples:
        print(f"[il] --per-round {a.per_round} > --n-samples {a.n_samples}, "
              f"samples would repeat within a round; clamping")
        a.per_round = a.n_samples

    gap_iter = None
    if a.gap_file:
        gap_iter = iter(load_gap_series(a.gap_file, a.rounds * a.per_round * len(MODELS),
                                        a.seed))

    base_url = a.base_url.rstrip("/")
    true_map = load_true_map(a.test_meta)
    prior = load_prior(a.out)
    # only this scenario's rows are a reason to skip: (rep, model, sample) repeats across
    # scenarios that share one file, and keying without it silently loses the second run
    done = {(int(r.rep), str(r.model_id), int(r.sample_id))
            for r in prior.itertuples()
            if str(r.model_id) in MODELS and str(r.scenario) == a.scenario
            and int(r.enclave_us) > 0 and int(r.roundtrip_us) > 0}

    sess = make_session()
    try:
        wait_healthy(sess, base_url, a.timeout)
    except RuntimeError as e:
        print(f"[il] {e}")
        return 1

    total = a.rounds * a.per_round
    n_models = len(MODELS)
    print(f"[il] {a.scenario} via {base_url}: {a.rounds} rounds x {a.per_round} "
          f"per model x {n_models} = {total * n_models} timed requests ({total} per model)")
    print(f"[il] model order is re-shuffled every round; resident model tracked "
          f"and switched on demand, then {a.warmup} untimed warmup requests")
    print(f"[il] schedule is seeded per (round, model), so an interrupted run "
          f"resumes to exactly the schedule an uninterrupted run would have used")

    resident: Optional[str] = None
    rows: list[dict] = []
    failures = 0
    t_run0 = time.perf_counter()

    if a.order_schedule == "latin" and a.rounds % n_models:
        print(f"[il] WARNING: --rounds {a.rounds} is not a multiple of {n_models}, so the "
              f"last {a.rounds % n_models} rounds leave a residual within-round slot "
              f"imbalance. Use a multiple of {n_models} for exact position balance.")

    # Rotate with period len(MODELS): every model occupies every slot once per
    # complete cycle, independent of whether the fleet contains 7 or 9 models.
    base_order = list(MODELS)
    random.Random(f"{a.seed}-base").shuffle(base_order)

    for rnd in range(a.rounds):
        if a.order_schedule == "latin":
            if rnd % n_models == 0:
                random.Random(f"{a.seed}-base-{rnd}").shuffle(base_order)
            order = latin_order(base_order, rnd)
        else:
            order = list(MODELS)
            random.Random(f"{a.seed}-order-{rnd}").shuffle(order)
        for model_id in order:
            if model_id != resident:
                if not switch(sess, base_url, model_id, a.timeout):
                    print(f"[il] round {rnd}: cannot switch to {model_id}, "
                          f"skipping its {a.per_round} samples")
                    failures += 1
                    continue
                resident = model_id
                warmup(sess, base_url, model_id, a.warmup, a.n_samples, a.timeout)

            sched = random.Random(f"{a.seed}-{rnd}-{model_id}")
            picks = sched.sample(range(a.n_samples), a.per_round)
            for sample_id in picks:
                # consumed for every slot, resumed or not, so a resumed run
                # assigns the same gaps to the same positions
                jitter_s = next(gap_iter) if gap_iter is not None else 0.0
                if (rnd, model_id, sample_id) in done:
                    continue
                if gap_iter is None and a.jitter_us:
                    jitter_s = sched.uniform(a.jitter_us[0], a.jitter_us[1]) / 1e6
                wall_s = time.time()
                enclave_us, rt_us, resp_bytes, resp_body = timed_query(
                    sess, base_url, model_id, sample_id,
                    a.timeout, jitter_s, a.jitter_inside)
                if enclave_us < 0:
                    failures += 1
                rows.append({
                    "scenario": a.scenario, "model_id": model_id, "rep": rnd,
                    "sample_id": sample_id,
                    "true_class": true_map.get(sample_id, -1),
                    "enclave_us": enclave_us, "roundtrip_us": rt_us,
                    "wall_ts": wall_s, "resp_bytes": resp_bytes,
                    "resp_body": resp_body,
                })
            a.out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([prior, pd.DataFrame(rows, columns=COLUMNS)],
                      ignore_index=True).to_csv(a.out, index=False)

        if rows and (rnd % 10 == 0 or rnd == a.rounds - 1):
            got = pd.concat([prior, pd.DataFrame(rows, columns=COLUMNS)],
                            ignore_index=True)
            per = got.groupby("model_id").size()
            elapsed = time.perf_counter() - t_run0
            rate = len(rows) / max(elapsed, 1e-9)
            print(f"[il] round {rnd + 1}/{a.rounds}: {len(got)} rows, "
                  f"{per.min()}-{per.max()} per model, "
                  f"median roundtrip "
                  f"{got['roundtrip_us'].median() / 1000:.1f} ms, "
                  f"{rate:.1f} req/s, eta "
                  f"{(a.rounds - rnd - 1) * n_models * a.per_round / max(rate, 1e-9) / 60:.0f} min")

    df = pd.concat([prior, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"\n[il] DONE {a.scenario}: {len(df)} rows -> {a.out}  "
          f"(switch/request failures counted: {failures})")
    bal = df.groupby("model_id").agg(n=("roundtrip_us", "size"),
                                     rt_med_ms=("roundtrip_us", lambda s: s.median() / 1000),
                                     encl_med_us=("enclave_us", "median"),
                                     resp_med_B=("resp_bytes", "median"))
    print(bal.round(2).to_string())
    print("\n[il] balance check: max/min per-model n should be ~1.0; "
          "if a model is over-represented late in the run the shuffle is not "
          "doing its job")
    if len(bal):
        print(f"[il]   n ratio = {bal['n'].max() / max(1, bal['n'].min()):.3f}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
