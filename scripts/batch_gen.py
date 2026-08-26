#!/usr/bin/env python
"""Generatore BATCH MiniMax-H3 (t2va): carica il modello UNA volta, poi cicla su una lista
di prompt scrivendo un .mp4 per prompt. Pensato per Sullivan (Strix Halo gfx1151).

Uso:
  env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python batch_gen.py [prompts.json] [output_dir]

prompts.json = lista di oggetti:
  [{"name":"clip01","prompt":"...","width":512,"height":288,"num_frames":124,"steps":8,"seed":42}, ...]
I campi width/height/num_frames/steps/seed sono OPZIONALI (default globali in DEF).
Vincoli H3: 24fps, 5-15s, num_frames = 17*n+5, lati multipli di 32.
"""
import sys, os, json, time, gc, torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL = "/home/michele/Scaricati/MiniMax-H3"
DEF = dict(width=512, height=288, num_frames=124, steps=8, seed=42)

prompts_path = sys.argv[1] if len(sys.argv) > 1 else "prompts.json"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "batch_out"
os.makedirs(out_dir, exist_ok=True)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

with open(prompts_path) as f:
    jobs = json.load(f)
log(f"{len(jobs)} prompt da generare -> {out_dir}/")

Q_DIT = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
ENC_SKIP = ("visual", "embed_tokens", ".norm", "lm_head")

# ---------- CARICAMENTO (una volta sola) ----------
t_load = time.time()
log("Pipeline modulare ...")
pipe = ModularPipeline.from_pretrained(MODEL)

log("text_encoder Qwen3-VL: loader a shard diretto in VRAM ...")
cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
with init_empty_weights():
    enc = Qwen3VLForConditionalGeneration(cfg)
enc = load_checkpoint_and_dispatch(enc, checkpoint=os.path.join(MODEL, "text_encoder"),
                                   device_map={"": 0}, dtype=torch.bfloat16)
quantize_(enc, Int8WeightOnlyConfig(version=2),
          filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and not any(s in fqn for s in ENC_SKIP))
gc.collect(); torch.cuda.empty_cache()

log("transformer (int8) ...")
dit = MiniMaxH3Transformer3DModel.from_pretrained(
    MODEL, subfolder="transformer", dtype=torch.bfloat16,
    quantization_config=Q_DIT, device_map="cuda")
gc.collect(); torch.cuda.empty_cache()

pipe.update_components(transformer=dit, text_encoder=enc)
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
pipe.transformer.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.vae.to("cuda")
pipe.audio_vae.to("cuda")
log(f"Modello pronto in {time.time()-t_load:.0f}s, VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ---------- LOOP (genera una clip per prompt) ----------
ok = 0
for i, job in enumerate(jobs):
    name = job.get("name", f"clip{i+1:03d}")
    p = dict(DEF); p.update({k: job[k] for k in ("width", "height", "num_frames", "steps", "seed") if k in job})
    out = os.path.join(out_dir, f"{name}.mp4")
    try:
        log(f"[{i+1}/{len(jobs)}] {name}: {p['width']}x{p['height']} {p['num_frames']}f {p['steps']}step ...")
        t0 = time.time()
        r = pipe(prompt=job["prompt"], width=p["width"], height=p["height"],
                 num_frames=p["num_frames"], num_inference_steps=p["steps"],
                 generator=torch.Generator().manual_seed(p["seed"]),
                 output=["videos", "audio", "sampling_rate"])
        encode_video(r["videos"][0], fps=24, output_path=out,
                     audio=r["audio"][0], audio_sample_rate=r["sampling_rate"])
        log(f"    OK in {time.time()-t0:.0f}s -> {out}")
        ok += 1
    except Exception as e:
        log(f"    ERRORE su {name}: {e!r}")

log(f"FATTO: {ok}/{len(jobs)} clip generate in {out_dir}/")
