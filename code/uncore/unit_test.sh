#!/bin/bash
# unit_test.sh — mid_lib.sh 的离线单测（纯逻辑，无后台进程、无 PCIe、无 PMU）
#
# 覆盖六项要求：
#   U1 PA_A / PA_B 分别落到 box0 / box1 的预期寄存器（cross 布局）
#   U2 掩码写 0、opcode=0xF0；未启用的 box 被显式关闭
#   U3 64B 粒度为对齐：低位被 &0xFFFFFFC0 截断
#   U4 perf 计数窗口只由 WINDOW 决定，与 victim 迭代次数无关
#   U5 victim 只报 PA_A 不报 PA_B 时，parse_pa 必须失败（主脚本据此 FATAL）
#   U6 解析只认传入的那个临时文件，不会读到别处的旧 PA
set -u
BASE=$(cd "$(dirname "$0")" && pwd)
. "$BASE/../mid_lib.sh"

PASS=0; FAIL=0
chk() { if [ "$2" = "$3" ]; then echo "  [PASS] $1"; PASS=$((PASS+1));
        else echo "  [FAIL] $1 (expect=$2 got=$3)"; FAIL=$((FAIL+1)); fi; }

echo "== U1: cross 布局把 A/B 分到 box0/box1 =="
PA_A=1a2b3000; PA_B=1a2b4000
map=$(layout_for cross 0 | tr '\n' ' ')
chk "mode0 映射" "0:1a2b3000 1:1a2b4000 2:0 3:0 " "$map"
map=$(layout_for cross 1 | tr '\n' ' ')
chk "mode1 映射（cross 下不随 mode 变）" "0:1a2b3000 1:1a2b4000 2:0 3:0 " "$map"

echo "== U2: 寄存器与关闭 =="
g=$(prog_args fe:0c.0 1a2b3000 | tr '\n' '|')
chk "box0 编程序列" "fe:0c.0 0x490.L=1a2b3000|fe:0c.0 0x494.L=00000000|fe:0c.0 0x498.L=00000000|fe:0c.0 0x49c.L=00000000|fe:0c.0 0x4a0.L=000000f0|" "$g"
d=$(disable_args fe:0e.0 | tr '\n' '|')
chk "关闭序列" "fe:0e.0 0x490.L=00000000|fe:0e.0 0x4a0.L=00000000|" "$d"

echo "== U3: 64B 对齐 =="
chk "1a2b3fc1 对齐" "1a2b3fc0 00000000" "$(split_pa 1a2b3fc1)"
chk "高位拆分" "00003000 00000001" "$(split_pa 100003001)"

echo "== U4: 窗口与迭代次数解耦 =="
p1=$(perf_args 1.2 | tr '\n' ' ')
chk "perf 参数含固定窗口" "1" "$(echo "$p1" | grep -c 'sleep 1.2')"
chk "perf 参数不含迭代次数" "0" "$(echo "$p1" | grep -c '10000000')"
chk "四 box 事件齐全" "4" "$(echo "$p1" | grep -o 'uncore_m2m_[0-3]' | sort -u | wc -l)"

echo "== U5: 缺 PA_B 必须解析失败 =="
t1=$(mktemp); printf 'MODEL=0 PA_A=1a2b3000\n' >"$t1"
a=$(parse_pa "$t1" A); ra=$?
parse_pa "$t1" B >/dev/null; rb=$?
chk "PA_A 可解析" "1a2b3000" "$a"
chk "PA_A 返回 0" "0" "$ra"
chk "PA_B 缺失返回 1" "1" "$rb"
rm -f "$t1"

echo "== U6: 只读传入文件，不跨文件取旧值 =="
t2=$(mktemp); printf 'MODEL=0 PA_A=dead0000 PA_B=beef0000\n' >"$t2"
t3=$(mktemp); printf 'MODEL=1 PA_A=1a2b3000 PA_B=1a2b4000\n' >"$t3"
chk "旧文件里的值" "dead0000" "$(parse_pa "$t2" A)"
chk "新文件里的值不受旧文件影响" "1a2b3000" "$(parse_pa "$t3" A)"
chk "新文件 PA_B" "1a2b4000" "$(parse_pa "$t3" B)"
rm -f "$t2" "$t3"

echo "== U8: only-N 布局 =="
chk "only2 只启用 box2" "0:0 1:0 2:1a2b3000 3:0 " "$(layout_for only2 0 | tr '\n' ' ')"
chk "only0 只启用 box0" "0:1a2b3000 1:0 2:0 3:0 " "$(layout_for only0 0 | tr '\n' ' ')"

echo "== U7: 主脚本确实用 mktemp 且先检查两个 PA =="
S="$BASE/../mid_run_dual.sh"
chk "使用 mktemp" "1" "$(grep -c 'mktemp /tmp/midv' "$S")"
chk "缺任一 PA 即 FATAL" "1" "$(grep -c 'FATAL: victim 未同时报出' "$S")"
chk "默认 dry-run（需 --execute）" "1" "$(grep -c '^EXECUTE=0; DRY=1' "$S")"

echo
echo "结果: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
