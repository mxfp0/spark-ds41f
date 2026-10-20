from pathlib import Path
import hashlib,importlib.util,json,os,tempfile,unittest
spec=importlib.util.spec_from_file_location('local_engram',Path(__file__).with_name('local_engram.py'))
local=importlib.util.module_from_spec(spec);spec.loader.exec_module(local)

class LocalEngramTests(unittest.TestCase):
 def test_owned_rows_match_and_invalid_local_data_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);source=base/'source';source.mkdir();root=base/'local'
   rows=24*1024;weight_offset=4096;scale_offset=weight_offset+rows*256
   data=bytes(range(256))*((scale_offset+rows*8)//256)
   files={}
   for layer,n in [(1,47),(14,48)]:
    name=f'model-{n:05}-of-00048.safetensors';(source/name).write_bytes(data)
    files[name]={'size':len(data),'header_bytes':4096,'header_sha256':hashlib.sha256(data[:4096]).hexdigest(),'tensors':{f'layers.{layer}.engram.embed.weight':{'shape':[rows,256],'absolute_offset':weight_offset},f'layers.{layer}.engram.embed.scale':{'shape':[rows,8],'absolute_offset':scale_offset}}}
   plan={'revision':local.REVISION,'geometry':{'layer_ids':[1,14],'num_embeddings':[rows,rows],'primes':[[1024]*24]*2,'offsets':[[i*1024 for i in range(24)]]*2},'files':files}
   plan_path=base/'local-plan.json';plan_path.write_text(json.dumps(plan))
   local.PLAN=plan_path;local.PLAN_SHA256=hashlib.sha256(plan_path.read_bytes()).hexdigest()
   local.build(source,root,1,False);m=local.verify(root)
   local.build(source,root,1,False)
   os.environ['SPARK3_ENGRAM_LOCAL_DIR']=str(root);
   name=next(iter(files));path=root/name
   self.assertEqual(path.stat().st_size,len(data))
   with path.open('rb') as f:
    for c in range(1,24,4):
     for width,offset in [(256,weight_offset),(8,scale_offset)]:
      for row in [c*1024,c*1024+513,(c+1)*1024-1]:
       f.seek(offset+row*width);self.assertEqual(f.read(width),data[offset+row*width:offset+(row+1)*width])
   mask=sum(1<<c for c in range(1,24,4))
   self.assertEqual(local.local_source('/models/'+name,weight_offset,False,rows,mask),str(path))
   with self.assertRaises(ValueError):local.local_source('/models/'+name,weight_offset,False,rows,mask<<1)
   with self.assertRaises(ValueError):local.local_source('/models/'+name,weight_offset+1,False,rows,mask)
   with self.assertRaises(ValueError):local.build(source,source,1,False)
   bad={**m,'files':{}};(root/'manifest.json').write_text(json.dumps(bad))
   with self.assertRaises(ValueError):local.verify(root)
   (root/'manifest.json').write_text(json.dumps(m))

   path.chmod(0o644)
   with path.open('r+b') as f:f.seek(weight_offset+1024*256);f.write(b'bad')
   with self.assertRaises(ValueError):local.verify(root)
   self.assertEqual((source/name).read_bytes(),data)

if __name__=='__main__':unittest.main()
