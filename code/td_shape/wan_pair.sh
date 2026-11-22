#!/bin/bash
# wan_pair.sh <labelA> <fileA> <labelB> <fileB> [blocks]
set -u
KEY=/tmp/_test111.pem; HOST=root@47.94.176.91; PORT=8089
LA=$1; FA=$2; LB=$3; FB=$4; BLOCKS=${5:-5}; N=60
SSH() { ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"; }
start_srv() {
  SSH $HOST "cd /root/a3 && (setsid python3 a3_server.py /root/models/$1 $PORT </dev/null > wan_srv.log 2>&1 &); sleep 0.3; true" >/dev/null 2>&1
  for i in $(seq 1 100); do curl -s -m 2 "http://127.0.0.1:18089/ping" >/dev/null 2>&1 && return 0; sleep 0.2; done
  echo "  !! not ready: $1"; return 1
}
stop_srv() { SSH $HOST "pkill -f '[a]3_server.py' 2>/dev/null; sleep 0.2; true" >/dev/null 2>&1; }
for b in $(seq 1 $((BLOCKS*2))); do
  if [ $(( b % 2 )) -eq 1 ]; then lab=$LA; f=$FA; else lab=$LB; f=$FB; fi
  stop_srv
  start_srv "$f" || continue
  python wan_client.py --tag "${LA}|${LB}:${lab}:b${b}" --n $N --port 18089
  stop_srv
done
stop_srv
echo "PAIR_DONE"
