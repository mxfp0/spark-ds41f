from pathlib import Path
import argparse
import os
import concurrent.futures
import runpy
import shlex
import subprocess

root=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(root/'bin/spark'))
cluster,nodes,_=ns['configuration'](argparse.Namespace(cluster_config=os.environ.get('SPARK_CLUSTER_CONFIG','config/sparks.json')))
expected=cluster['environment'].get('SPARK_NCCL_LIBRARY_SHA256')
if expected:
    mounts=[m for m in cluster['container']['mounts']
            if m[1]=='/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2']
    assert len(mounts)==1 and mounts[0][2]=='ro'
    source=mounts[0][0].format(home=cluster['host']['home'])
    def check(node):
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
                          'spark@'+node['ssh_host'],shlex.join(['sha256sum',source])],
                         check=True,capture_output=True,text=True,timeout=30)
        assert r.stdout.split()[0]==expected,(node['name'],'NCCL library hash mismatch')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(check,nodes['nodes']))
    print('All4 pinned NCCL library hashes verified',flush=True)
