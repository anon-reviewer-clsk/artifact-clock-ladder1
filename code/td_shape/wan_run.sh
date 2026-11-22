#!/bin/bash
# wan_run.sh — WAN 腿：经 SSH 隧道从本地打 TD 内的服务
set -u
KEY=/tmp/_test111.pem
HOST=root@47.94.176.91
PORT=8089
N=60
BLOCKS=5
SSH() { ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"; }

start_srv() {  # $1 = model file
  SSH $HOST "cd /root/a3 && (setsid python3 a3_server.py /root/models/$1 $PORT </dev/null > wan_srv.log 2>&1 &); sleep 0.3; echo started"
  for i in $(seq 1 100); do
    if curl -s -m 2 "http://127.0.0.1:18089/ping" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  echo "  !! server not ready for $1"; return 1
}

stop_srv() {
  SSH $HOST "pkill -f '[a]3_server.py' 2>/dev/null; sleep 0.2; true" >/dev/null 2>&1
}

MODELS=(dsv1_downstream.onnx resnet18.onnx dsv1_tok_v4.onnx)
for mi in "${!MODELS[@]}"; do
  m=${MODELS[$mi]}
  start_int8=$(( mi % 2 ))
  for b in $(seq 1 $((BLOCKS*2))); do
    v=$(( (b-1) % 2 ^ start_int8 ))
    if [ "$v" = "1" ]; then variant=int8; f="q_$m"; else variant=fp32; f="$m"; fi
    stop_srv
    if ! start_srv "$f"; then continue; fi
    python wan_client.py --tag "${m}:${variant}:b${b}" --n $N --port 18089 || echo "  client failed ${m} ${variant} b${b}"
    stop_srv
  done
done
stop_srv
echo "WAN_DONE"
