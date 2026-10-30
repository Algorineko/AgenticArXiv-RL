"""Bounded full-parameter FP32 SFT feasibility arm; paired source/data controls."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'AgenticArxiv'))
from benchmark.rescue_catalog import sha256

def tokenize_row(tokenizer,row,max_length=4096):
 messages=row['messages']
 prefix=tokenizer.apply_chat_template(messages[:1],tokenize=False,add_generation_prompt=True)
 full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
 p=tokenizer(prefix,add_special_tokens=False)['input_ids']
 ids=tokenizer(full,add_special_tokens=False)['input_ids']
 if ids[:len(p)]!=p or len(ids)<=len(p):raise ValueError('Assistant prefix mismatch')
 if len(ids)>max_length:raise ValueError(f'Refuse to truncate training example: {len(ids)} > {max_length}')
 return ids,[-100]*len(p)+ids[len(p):]

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--mode',choices=['hidden','visible'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.environ.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 a.output.mkdir(parents=True,exist_ok=False)
 manifest={'kind':'new_SFT_setup_rendering_feasibility','mode':a.mode,'status':'running','seed':123,'max_steps':16,'batch_size':1,'gradient_accumulation':4,'learning_rate':2e-5,'weight_decay':0.0,'max_length':4096,'precision':'FP32 master parameters and optimizer; BF16 autocast; BF16 checkpoint export','started_at':datetime.now(timezone.utc).isoformat(),'source_sha256':{n:sha256(ROOT/n) for n in ['scripts/train_rescue_sft_pair.py','scripts/generate_sft_data.py']},'actual_steps':0}
 path=a.output/'rescue_training_manifest.json'
 def save():path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 save();start=time.monotonic()
 def timeout(*_):raise TimeoutError('1800 second arm limit')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
 lock=open(f'/tmp/agenticarxiv-rescue-{os.getuid()}-gpu1.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  free=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
  if int(free)<38000:raise RuntimeError('Fewer than 38000 MiB free on physical GPU1')
  declared=json.loads((a.data/'manifest.json').read_text());data_file=a.data/f'{a.mode}.jsonl'
  if sha256(data_file)!=declared['files'][data_file.name]:raise ValueError('Training file changed')
  manifest.update(data_manifest_sha256=sha256(a.data/'manifest.json'),data_sha256=sha256(data_file),base_config_sha256=sha256(a.model/'config.json'),base_weights_sha256={x.name:sha256(x) for x in sorted(a.model.glob('*.safetensors'))})
  import torch,transformers
  from transformers import AutoTokenizer,AutoModelForCausalLM
  torch.set_num_threads(4);torch.manual_seed(123);random.seed(123);torch.cuda.set_per_process_memory_fraction(.85,0)
  torch.backends.cuda.matmul.allow_tf32=True
  tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
  rows=[json.loads(x) for x in data_file.read_text().splitlines()]
  encoded=[tokenize_row(tokenizer,row) for row in rows]
  order=list(range(len(rows)));random.Random(123).shuffle(order);order=order[:64]
  if len(order)!=64:raise ValueError('Need 64 unique decision examples')
  manifest.update(versions={'torch':torch.__version__,'transformers':transformers.__version__},row_indices=order,row_task_ids=[rows[i]['source_task_id'] for i in order],supervised_tokens=sum(sum(t!=-100 for t in encoded[i][1]) for i in order),input_tokens=sum(len(encoded[i][0]) for i in order));save()
  model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32,local_files_only=True).to('cuda')
  model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
  probe=next(model.parameters())
  before=hashlib.sha256(probe.detach().cpu().numpy().tobytes()).hexdigest()
  optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.0,foreach=False)
  losses=[];optimizer.zero_grad(set_to_none=True)
  with (a.output/'steps.jsonl').open('x') as log:
   for step in range(16):
    loss_sum=0
    for index in order[step*4:(step+1)*4]:
     ids,labels=encoded[index]
     with torch.autocast('cuda',dtype=torch.bfloat16):
      loss=model(input_ids=torch.tensor([ids],device='cuda'),labels=torch.tensor([labels],device='cuda')).loss
     if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
     loss_sum+=loss.item();(loss/4).backward()
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if not torch.isfinite(grad_norm):raise RuntimeError('Non-finite gradient norm')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    record={'step':step+1,'loss':loss_sum/4,'grad_norm':grad_norm.item()}
    losses.append(record);log.write(json.dumps(record)+'\n');log.flush();manifest['actual_steps']=step+1;save();print(json.dumps(record),flush=True)
  after=hashlib.sha256(probe.detach().cpu().numpy().tobytes()).hexdigest()
  if after==before:raise RuntimeError('Optimizer did not change tracked parameters')
  manifest.update(probe_before_sha256=before,probe_after_sha256=after,parameters_changed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())
  del optimizer
  model.config.use_cache=True;model.to(torch.bfloat16);model.save_pretrained(a.output,safe_serialization=True,max_shard_size='2GB');tokenizer.save_pretrained(a.output)
  manifest.update(status='completed',export_weights_sha256={x.name:sha256(x) for x in sorted(a.output.glob('*.safetensors'))},loss_first=losses[0]['loss'],loss_last=losses[-1]['loss'])
 except Exception as e:
  manifest.update(status='failed',error_type=type(e).__name__,error=str(e));raise
 finally:
  signal.alarm(0);manifest['wall_seconds']=round(time.monotonic()-start,3);save();lock.close()
 print(json.dumps({'status':manifest['status'],'steps':manifest['actual_steps'],'wall_seconds':manifest['wall_seconds']}))
if __name__=='__main__':main()
