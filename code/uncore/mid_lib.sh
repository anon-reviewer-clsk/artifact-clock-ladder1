#!/bin/bash
# mid_lib.sh — mid_run_dual.sh 的纯逻辑部分（无副作用、可单测）
#
# 抽出来的原因：把"映射哪个 box 到哪个 PA""写哪些寄存器"这类可判定的逻辑
# 与"起 victim、写 PCIe、跑 perf"这类有副作用的部分分开，前者可以离线单测。

BOXES=(fe:0c.0 fe:0d.0 fe:0e.0 fe:0f.0)
REG_ADDR_LO=0x490
REG_ADDR_HI=0x494
REG_MASK_LO=0x498
REG_MASK_HI=0x49c
REG_OPCODE=0x4a0
OPCODE_ALL=000000f0
GRANULE_MASK=0xFFFFFFC0

# layout_for <layout> <mode> → 每行 "box_index:pa"，pa=0 表示该 box 不启用
layout_for() {
  local layout=$1 mode=$2
  case "$layout:$mode" in
    cross:0|cross:1) echo "0:$PA_A"; echo "1:$PA_B"; echo "2:0"; echo "3:0" ;;
    all-a:*) echo "0:$PA_A"; echo "1:$PA_A"; echo "2:$PA_A"; echo "3:$PA_A" ;;
    all-b:*) echo "0:$PA_B"; echo "1:$PA_B"; echo "2:$PA_B"; echo "3:$PA_B" ;;
    only0:*) echo "0:$PA_A"; echo "1:0"; echo "2:0"; echo "3:0" ;;
    only1:*) echo "0:0"; echo "1:$PA_A"; echo "2:0"; echo "3:0" ;;
    only2:*) echo "0:0"; echo "1:0"; echo "2:$PA_A"; echo "3:0" ;;
    only3:*) echo "0:0"; echo "1:0"; echo "2:0"; echo "3:$PA_A" ;;
    *) return 2 ;;
  esac
}

# split_pa <pa-hex> → "lo hi"（lo 按 64B 对齐）
split_pa() {
  local pa=$1
  local lo=$(( 0x$pa & GRANULE_MASK ))
  local hi=$(( 0x$pa >> 32 ))
  printf "%08x %08x" "$lo" "$hi"
}

# parse_pa <file> <A|B> → 打印 PA，缺失则返回 1
parse_pa() {
  local f=$1 which=$2 v
  v=$(grep -oE "PA_${which}=[0-9a-f]+" "$f" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$v" ] || return 1
  printf '%s' "$v"
}

# prog_args <bdf> <pa> → 逐行打印要写的光标（供 main 脚本执行，也供单测断言）
prog_args() {
  local bdf=$1 pa=$2 lo hi
  read -r lo hi <<<"$(split_pa "$pa")"
  echo "$bdf $REG_ADDR_LO.L=$lo"
  echo "$bdf $REG_ADDR_HI.L=$hi"
  echo "$bdf $REG_MASK_LO.L=00000000"
  echo "$bdf $REG_MASK_HI.L=00000000"
  echo "$bdf $REG_OPCODE.L=$OPCODE_ALL"
}

# disable_args <bdf> → 关闭该 box 的地址过滤
disable_args() {
  echo "$1 $REG_ADDR_LO.L=00000000"
  echo "$1 $REG_OPCODE.L=00000000"
}

# perf_args <window> → perf stat 的完整参数（窗口固定，不含 victim 迭代次数）
perf_args() {
  # 注意：不能用 echo "-e"，bash 会当成选项吞掉（实测导致 perf 把事件串当 workload）
  local w=$1
  printf '%s\n' \
    "stat" \
    "-e" \
    "uncore_m2m_0/event=0x4c,umask=0x3/,uncore_m2m_1/event=0x4c,umask=0x3/,uncore_m2m_2/event=0x4c,umask=0x3/,uncore_m2m_3/event=0x4c,umask=0x3/" \
    "-a" "--" "sleep" "$w"
}
