from pathlib import Path
import concurrent.futures,shlex,subprocess
root=Path(__file__).resolve().parents[1]
ips=['192.168.0.184','192.168.0.200','192.168.0.205','192.168.0.189']
rel='experiments/2026-10-18-local-engram'
def check(item):
 rank,ip=item
 args=['python3',str(root/rel/'local_engram.py'),'verify','--root','/home/spark/.local/share/deepseek1017-vllm/engram-local','--rank',str(rank)]
 command='cd '+shlex.quote(str(root))+' && sha256sum -c '+rel+'/SHA256SUMS && '+shlex.join(args)
 subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','spark@'+ip,command],check=True,timeout=600)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(check,enumerate(ips)))
print('All4 local Engram files and source hashes verified',flush=True)
