#!/usr/bin/env python3
"""Measure fixed-array MPS training steps without JSON/collation overhead."""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch
from jp_lexer.losses import multitask_loss
from jp_lexer.model import JpLexer
from scripts.train_prepared import FIELDS, load_batch

def sync(): torch.mps.synchronize()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('data',type=Path); ap.add_argument('--sizes',nargs='+',default=['50k','150k','250k']); ap.add_argument('--batches',nargs='+',type=int,default=[128,256,512]); ap.add_argument('--bucket',type=int,default=128); ap.add_argument('--warmup',type=int,default=2); ap.add_argument('--steps',type=int,default=5); args=ap.parse_args()
    if not torch.backends.mps.is_available(): raise SystemExit('MPS is not available')
    out=[]; device=torch.device('mps')
    for size in args.sizes:
        root=args.data/size; stores={b:{f:__import__('numpy').load(root/f'train-{b}-{f}.npy',mmap_mode='r') for f in FIELDS} for b in (32,64,128) if (root/f'train-{b}-mask.npy').exists()}
        for batch_size in args.batches:
            model=JpLexer.from_size(size).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); inputs,targets=load_batch(stores,args.bucket,0,batch_size); inputs={k:v.to(device) for k,v in inputs.items()}; targets={k:v.to(device) for k,v in targets.items()}; model.train(); sync()
            for _ in range(args.warmup):
                opt.zero_grad(set_to_none=True); loss=multitask_loss(model(inputs),targets)['total']; loss.backward(); opt.step()
            sync(); times=[]
            for _ in range(args.steps):
                start=time.perf_counter(); opt.zero_grad(set_to_none=True); loss=multitask_loss(model(inputs),targets)['total']; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sync(); times.append((time.perf_counter()-start)*1000)
            mean=statistics.mean(times); row={'size':size,'bucket':args.bucket,'batch_size':batch_size,'step_ms':mean,'sentences_per_second':batch_size/(mean/1000),'codepoints_per_second':batch_size*args.bucket/(mean/1000)}; out.append(row); print(json.dumps(row),flush=True); del model,opt,inputs,targets; torch.mps.empty_cache()
    print(json.dumps({'results':out},indent=2))
if __name__=='__main__': main()
