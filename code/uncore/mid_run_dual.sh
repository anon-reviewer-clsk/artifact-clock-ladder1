#!/bin/bash
# mid_run_dual.sh v2.1 — UncoreBleed mid_run 修复版（不覆盖原文件）
#
# 与原版的差异见 DIFF_mid_run.md。相比 v2：纯逻辑抽到 mid_lib.sh 便于离线单测；
# 去掉 EXIT trap（容器/msys 下会与后台任务互相拖住），改为显式 cleanup。
#
# 默认 dry-run，必须 --execute 才写 PCIe。
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/mid_lib.sh"

EXECUTE=0; DRY=1
LAYOUT=${LAYOUT:-cross}; REPS=${REPS:-1}; WINDOW=${WINDOW:-1.2}
ITERS=${ITERS:-10000000}; BIND=${BIND:-1}; OUTDIR=${OUTDIR:-.}

while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; DRY=0 ;;
    --dry-run) DRY=1 ;;
    --layout) LAYOUT=$2; shift ;;
    --reps) REPS=$2; shift ;;
    --window) WINDOW=$2; shift ;;
    --iters) ITERS=$2; shift ;;
    --out) OUTDIR=$2; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac; shift
done
case "$LAYOUT" in cross|all-a|all-b|only0|only1|only2|only3) ;; *) echo "bad layout: $LAYOUT" >&2; exit 2 ;; esac

run() { if [ "$DRY" = "1" ]; then echo "[dry] $*" >&2; else "$@"; fi; }

prog_box() { local line; while read -r line; do run setpci -s $line; done < <(prog_args "$1" "$2"); }
dis_box()  { local line; while read -r line; do run setpci -s $line; done < <(disable_args "$1"); }
cleanup()  { local b; for b in "${BOXES[@]}"; do dis_box "$b"; done; }
trap 'cleanup; exit 130' INT TERM

TSV="$OUTDIR/mid_run_dual.tsv"
printf "rep\tmode\tlayout\tpa_a\tpa_b\twindow\titers\tbox0\tbox1\tbox2\tbox3\talive\tperf_log\n" > "$TSV"

if [ "$DRY" = "1" ]; then
  echo "[dry-run] layout=$LAYOUT reps=$REPS window=${WINDOW}s iters=$ITERS bind=$BIND" >&2
  echo "[dry-run] victim: numactl --membind=$BIND ./mid_victim <mode> $ITERS" >&2
  echo "[dry-run] perf: perf $(perf_args "$WINDOW" | tr '\n' ' ')" >&2
  echo "[dry-run] 不写 PCIe、不起 victim。加 --execute 执行。" >&2
  exit 0
fi

VICTIM_PID=""; VFILE=""

run_once() {
  local rep=$1 mode=$2 b pa perf_pid i
  VFILE=$(mktemp /tmp/midv.XXXXXX) || { echo "mktemp failed" >&2; return 1; }

  numactl --membind="$BIND" ./mid_victim "$mode" "$ITERS" >"$VFILE" 2>&1 &
  VICTIM_PID=$!
  disown "$VICTIM_PID" 2>/dev/null || true

  for i in $(seq 1 100); do
    grep -q 'PA_A=' "$VFILE" 2>/dev/null && break
    sleep 0.1
  done

  PA_A=$(parse_pa "$VFILE" A) || PA_A=""
  PA_B=$(parse_pa "$VFILE" B) || PA_B=""
  if [ -z "$PA_A" ] || [ -z "$PA_B" ]; then
    echo "FATAL: victim 未同时报出 PA_A / PA_B (A='${PA_A}' B='${PA_B}')" >&2
    kill -9 "$VICTIM_PID" 2>/dev/null; VICTIM_PID=""
    cleanup; return 1
  fi

  echo "== rep=$rep mode=$mode layout=$LAYOUT PA_A=$PA_A PA_B=$PA_B" | tee -a "$OUTDIR/mid_run_dual.log"
  while IFS=: read -r b pa; do
    if [ "$pa" = "0" ]; then dis_box "${BOXES[$b]}"; else prog_box "${BOXES[$b]}" "$pa"; fi
    echo "  box$b=${BOXES[$b]} pa=${pa}" >> "$OUTDIR/mid_run_dual.log"
  done < <(layout_for "$LAYOUT" "$mode")

  perf $(perf_args "$WINDOW") >"$OUTDIR/mid_run_dual.perf.$rep.$mode.log" 2>&1 &
  perf_pid=$!
  kill -USR1 "$VICTIM_PID" 2>/dev/null
  wait "$perf_pid" 2>/dev/null

  local alive=0 plog
  kill -0 "$VICTIM_PID" 2>/dev/null && alive=1
  kill -TERM "$VICTIM_PID" 2>/dev/null
  for i in $(seq 1 20); do kill -0 "$VICTIM_PID" 2>/dev/null || break; sleep 0.05; done
  kill -9 "$VICTIM_PID" 2>/dev/null; VICTIM_PID=""

  plog="$OUTDIR/mid_run_dual.perf.$rep.$mode.log"
  local c0 c1 c2 c3
  c0=$(awk '/uncore_m2m_0/{gsub(",",""); print $1; exit}' "$plog")
  c1=$(awk '/uncore_m2m_1/{gsub(",",""); print $1; exit}' "$plog")
  c2=$(awk '/uncore_m2m_2/{gsub(",",""); print $1; exit}' "$plog")
  c3=$(awk '/uncore_m2m_3/{gsub(",",""); print $1; exit}' "$plog")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$rep" "$mode" "$LAYOUT" "$PA_A" "$PA_B" "$WINDOW" "$ITERS" \
    "${c0:-NA}" "${c1:-NA}" "${c2:-NA}" "${c3:-NA}" "$alive" "$(basename "$plog")" >> "$TSV"
  rm -f "$VFILE"; VFILE=""
}

for rep in $(seq 1 "$REPS"); do
  if [ $(( RANDOM % 2 )) -eq 0 ]; then order="0 1"; else order="1 0"; fi
  echo "rep=$rep order=$order" >> "$OUTDIR/mid_run_dual.log"
  for mode in $order; do run_once "$rep" "$mode" || exit 1; done
done

cleanup
echo "done. results: $TSV"
