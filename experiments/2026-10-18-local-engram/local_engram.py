from pathlib import Path
from functools import lru_cache
import argparse,fcntl,hashlib,json,os,shutil,subprocess

REVISION='1bae0e1088993b9b82de08f59de5be5a34e42170'
PLAN=Path(__file__).with_name('local-plan.json')
PLAN_SHA256='d8ca408aebe76a3b623a7fdc77911f89facd519b6901540acdf8ccb2edff343f'
BLOCK=4096
CHUNK=4*1024*1024

def digest_range(f,start,end):
 h=hashlib.sha256();f.seek(start)
 while start<end:
  data=f.read(min(CHUNK,end-start))
  if not data:raise ValueError('short local Engram read')
  h.update(data);start+=len(data)
 return h.hexdigest()

def ranges_for(plan,name,rank):
 info=plan['files'][name];ranges=[(0,max(BLOCK,info['header_bytes']))]
 for key,t in info['tensors'].items():
  layer=int(key.split('.')[1]);i=plan['geometry']['layer_ids'].index(layer)
  width=8 if key.endswith('.scale') else 256
  assert t['shape']==[plan['geometry']['num_embeddings'][i],width]
  for c in range(rank,24,4):
   lo=plan['geometry']['offsets'][i][c];hi=lo+plan['geometry']['primes'][i][c]
   start=t['absolute_offset']+lo*width;end=t['absolute_offset']+hi*width
   ranges.append((start//BLOCK*BLOCK,min(info['size'],(end+BLOCK-1)//BLOCK*BLOCK)))
 merged=[]
 for start,end in sorted(ranges):
  if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
  else:merged.append([start,end])
 return merged

def require_local(path):
 kind=subprocess.check_output(['stat','-f','-c','%T',str(path)],text=True).strip()
 if kind.startswith(('nfs','cifs','smb')):raise ValueError('Engram runtime storage must be local: '+str(path))

def load_plan():
 raw=PLAN.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PLAN_SHA256:raise ValueError('local Engram plan checksum mismatch')
 return json.loads(raw)

def read_manifest(root):
 m=json.loads((root/'manifest.json').read_text())
 if m['revision']!=REVISION or m['tp_size']!=4 or m['rank'] not in range(4):raise ValueError('invalid local Engram identity')
 plan=load_plan()
 if m['plan_sha256']!=PLAN_SHA256 or set(m['files'])!=set(plan['files']):raise ValueError('local Engram manifest plan mismatch')
 for name,info in m['files'].items():
  for key in ('size','header_bytes','header_sha256','tensors'):
   if info[key]!=plan['files'][name][key]:raise ValueError('local Engram manifest descriptor mismatch')
  if [[a,b] for a,b,_ in info['ranges']]!=ranges_for(plan,name,m['rank']):raise ValueError('local Engram owned ranges mismatch')
 return m

def verify(root,full=True):
 require_local(root);m=read_manifest(root)
 for name,info in m['files'].items():
  p=root/name
  if p.is_symlink() or p.stat().st_size!=info['size']:raise ValueError('invalid local Engram file: '+name)
  require_local(p)
  with p.open('rb') as f:
   if digest_range(f,0,info['header_bytes'])!=info['header_sha256']:raise ValueError('local Engram header mismatch')
   if full:
    for start,end,wanted in info['ranges']:
     if digest_range(f,start,end)!=wanted:raise ValueError('local Engram range mismatch: '+name)
  allocated=p.stat().st_blocks*512
  expected=sum(end-start for start,end,_ in info['ranges'])
  if not info['hardlink'] and allocated>expected+16*1024*1024:raise ValueError('local Engram is not a sparse owned-rank copy')
 return m

@lru_cache(maxsize=4)
def local_source(path,offset,scale,table_rows,columns):
 root=Path(os.environ['SPARK3_ENGRAM_LOCAL_DIR']);m=read_manifest(root)
 if columns!=sum(1<<c for c in range(m['rank'],24,4)):raise ValueError('wrong-rank local Engram rows')
 name=Path(path).name;info=m['files'][name]
 matches=[t for k,t in info['tensors'].items() if k.endswith('.scale')==scale and t['shape'][0]==table_rows and t['absolute_offset']==offset]
 if len(matches)!=1:raise ValueError('local Engram source geometry mismatch')
 p=root/name
 if p.is_symlink() or p.stat().st_size!=info['size']:raise ValueError('invalid local Engram source')
 require_local(p)
 return str(p)

def build(source,root,rank,hardlink):
 plan=load_plan()
 if source.resolve()==root.resolve() or (root.exists() and os.path.samefile(source,root)):raise ValueError('source and local Engram destination must differ')
 assert plan['revision']==REVISION and rank in range(4)
 root.mkdir(parents=True,exist_ok=True);require_local(root)
 with (root/'.build.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'manifest.json').exists():
   old=verify(root)
   assert old['rank']==rank and old['plan_sha256']==PLAN_SHA256
   print('LOCAL_ENGRAM_ALREADY_VERIFIED',rank,flush=True);return
  expected=sum(end-start for name in plan['files'] for start,end in ranges_for(plan,name,rank))
  if not hardlink and shutil.disk_usage(root).free<expected+10*1024**3:raise ValueError('insufficient local disk')
  manifest={'schema':'local-engram/1','revision':REVISION,'tp_size':4,'rank':rank,'plan_sha256':PLAN_SHA256,'files':{}}
  for name,info in plan['files'].items():
   assert name in ('model-00047-of-00048.safetensors','model-00048-of-00048.safetensors')
   src=source/name;dst=root/name;tmp=root/(name+'.partial')
   if src.resolve()==dst.resolve():raise ValueError('refusing to replace checkpoint source')
   assert src.stat().st_size==info['size']
   with src.open('rb') as f:assert digest_range(f,0,info['header_bytes'])==info['header_sha256']
   ranges=ranges_for(plan,name,rank);hashes=[]
   if tmp.exists():tmp.unlink()
   if hardlink:
    require_local(src);os.link(src.resolve(),tmp)
    with tmp.open('rb') as f:
     hashes=[[start,end,digest_range(f,start,end)] for start,end in ranges]
   else:
    with src.open('rb') as f,tmp.open('x+b') as target:
     target.truncate(info['size'])
     for start,end in ranges:
      f.seek(start);target.seek(start);h=hashlib.sha256();pos=start
      while pos<end:
       data=f.read(min(CHUNK,end-pos))
       if not data:raise ValueError('short source Engram read')
       target.write(data);h.update(data);pos+=len(data)
      hashes.append([start,end,h.hexdigest()]);print(json.dumps({'rank':rank,'file':name,'copied_range':[start,end]}),flush=True)
     target.flush();os.fsync(target.fileno())
     for start,end,wanted in hashes:assert digest_range(target,start,end)==wanted
     os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
     os.posix_fadvise(target.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    tmp.chmod(0o444)
   os.replace(tmp,dst)
   manifest['files'][name]={**info,'ranges':hashes,'hardlink':hardlink}
  tmp=root/'manifest.json.partial'
  with tmp.open('w') as f:
   json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,root/'manifest.json')
  fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
  verify(root,full=False)
  print(json.dumps({'status':'LOCAL_ENGRAM_BUILT','rank':rank,'owned_bytes':expected,'hardlink':hardlink}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify']);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--rank',type=int);p.add_argument('--hardlink',action='store_true');a=p.parse_args()
 if a.action=='build':build(a.source,a.root,a.rank,a.hardlink)
 else:
  m=verify(a.root)
  if a.rank is not None and a.rank!=m['rank']:raise ValueError('wrong local Engram rank')
  print(json.dumps({'status':'LOCAL_ENGRAM_VERIFIED','rank':m['rank']}),flush=True)
