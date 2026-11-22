#!/usr/bin/env python3
"""a3_moe_run.py — 合成 MoE：不同输入 → 不同专家对 → 时延是否可分辨。"""
import http.client, json, os, signal, statistics as st, subprocess, sys, time
from pathlib import Path
OUT=Path("/root/a3"); MODEL="/root/models/moe_synth.onnx"; PORT=8097
KINDS=["rand","checker","ones","zeros","small"]
BLOCKS=5; REQ=30
def ready(p,f,t=300):
    t0=time.time()
    while time.time()-t0<t:
        time.sleep(0.1)
        if "READY" in Path(f).read_text(errors="ignore"): return True
        if p.poll() is not None: return False
    return False
def main():
    logf=OUT/"moe_srv.log"
    with open(logf,"w") as lf:
        p=subprocess.Popen([sys.executable,"/root/a3/a3_server.py",MODEL,str(PORT)],
                           stdout=lf,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
    rows=[]
    try:
        if not ready(p,logf): print("server failed"); return
        c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=120)
        for b in range(1,BLOCKS+1):
            order=KINDS[b%len(KINDS):]+KINDS[:b%len(KINDS)]
            for k in order:
                body=json.dumps({"kind":k}).encode(); rtts=[]; idx=None
                for _ in range(REQ):
                    t0=time.perf_counter()
                    c.request("POST","/infer",body=body,headers={"Content-Type":"application/json"})
                    r=c.getresponse(); payload=json.loads(r.read())
                    rtts.append((time.perf_counter()-t0)*1e3); idx=payload.get("idx")
                rows.append(dict(kind=k,block=b,idx=idx,rtts=rtts,med=st.median(rtts)))
                print(f"  b{b} {k:8s} idx={idx} med={st.median(rtts):7.2f}ms",flush=True)
        c.close()
    finally:
        try: os.killpg(os.getpgid(p.pid),signal.SIGKILL)
        except Exception: pass
        p.wait(timeout=10)
    json.dump(rows,open(OUT/"moe_raw.json","w")); print("DONE",flush=True)
if __name__=="__main__": main()
