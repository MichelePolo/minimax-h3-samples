#!/usr/bin/env python
"""FASE 1 — Precalcola e SALVA gli stati di condizionamento (Qwen3-VL).
Carica Qwen-VL UNA volta, codifica ogni prompt, salva uno stato .pt (~100 KB) per prompt.
Poi `denoise_cached.py` genera i video SENZA mai ricaricare Qwen-VL (63 GB).
Lo stato NON dipende dalla risoluzione: lo stesso .pt vale per qualsiasi WxH/durata.

Uso: env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python encode_prompt.py <prompts.json> <cache_dir>
"""
import sys, os, json, time, gc, torch
import torch.nn as nn
from diffusers import ModularPipeline
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL = "/home/michele/Scaricati/MiniMax-H3"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

prompts_path = sys.argv[1] if len(sys.argv) > 1 else "prompts.json"
cache_dir = sys.argv[2] if len(sys.argv) > 2 else "cond_cache"
os.makedirs(cache_dir, exist_ok=True)
jobs = json.load(open(prompts_path))
SKIP = ("visual", "embed_tokens", ".norm", "lm_head")

# --- carica SOLO il conditioner (workflow t2va, blocco text_encoder) ---
log("Carico il conditioner Qwen3-VL (loader a shard int8) ...")
workflow = ModularPipeline.from_pretrained(MODEL).blocks.get_workflow("t2va")
conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(MODEL)
cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
with init_empty_weights():
    enc = Qwen3VLForConditionalGeneration(cfg)
enc = load_checkpoint_and_dispatch(enc, checkpoint=os.path.join(MODEL, "text_encoder"),
                                   device_map={"": 0}, dtype=torch.bfloat16)
quantize_(enc, Int8WeightOnlyConfig(version=2),
          filter_fn=lambda m, f: isinstance(m, nn.Linear) and not any(s in f for s in SKIP))
gc.collect(); torch.cuda.empty_cache()
conditioner.update_components(text_encoder=enc)
conditioner.load_components(dtype=torch.bfloat16)
log(f"Conditioner pronto (VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB). Codifico {len(jobs)} prompt ...")

for i, job in enumerate(jobs):
    name = job.get("name", f"clip{i+1:03d}")
    state = conditioner(prompt=job["prompt"])
    # salvo SOLO i tensori/valori -> caricabili con weights_only=True (no code-exec)
    data = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.values.items()}
    path = os.path.join(cache_dir, f"{name}.pt")
    torch.save(data, path)
    log(f"[{i+1}/{len(jobs)}] {name}: stato salvato ({os.path.getsize(path)/1024:.0f} KB) -> {path}")

log(f"FATTO: {len(jobs)} stati in {cache_dir}/ (Qwen-VL ora si puo' spegnere)")
