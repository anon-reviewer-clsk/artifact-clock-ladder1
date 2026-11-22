#!/usr/bin/env python3
"""a3_ee_run.py — early-exit 模型：输入幅度 → 计算深度 → 时延是否可读。"""
import http.client, json, os, signal, statistics as st, subprocess, sys, time
from pathlib import Path
OUT=Path("/root/a3"); MODEL="/root/models/earlyexit.onnx"; PORT=8095
MAGS=["mag:0.1","mag:0.8","mag:1.6","mag:3.2","mag:6.0"]   # trip≈1,8,16,32,60
BLOCKS=5; REQ=30
def ready(p,f,t=180):
    t0=time.time()
    while time.time()-t0<t:
        time.sleep(0.05)
        if "READY" in Path(f).read_text(errors="ignore"): return True
        if p.poll() is not None: return False
    return False
def main():
    logf=OUT/"ee_srv.log"
    with open(logf,"w") as lf:
        p=subprocess.Popen([sys.executable,"/root/a3/a3_server.py",MODEL,str(PORT)],
                           stdout=lf,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
    rows=[]
    try:
        if not ready(p,logf): print("server failed"); return
        c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=120)
        for b in range(1,BLOCKS+1):
            order=MAGS[b%len(MAGS):]+MAGS[:b%len(MAGS)]
            for m in order:
                body=json.dumps({"kind":m}).encode(); rtts=[]
                for _ in range(REQ):
                    t0=time.perf_counter()
                    c.request("POST","/infer",body=body,headers={"Content-Type":"application/json"})
                    r=c.getresponse(); r.read()
                    rtts.append((time.perf_counter()-t0)*1e3)
                rows.append(dict(mag=m,block=b,rtts=rtts,med=st.median(rtts)))
                print(f"  b{b} {m:9s} med={st.median(rtts):7.2f}ms",flush=True)
        c.close()
    finally:
        try: os.killpg(os.getpgid(p.pid),signal.SIGKILL)
        except Exception: pass
        p.wait(timeout=10)
    json.dump(rows,open(OUT/"ee_raw.json","w")); print("DONE",flush=True)
if __name__=="__main__": main()
