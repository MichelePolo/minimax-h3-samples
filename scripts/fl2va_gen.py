#!/usr/bin/env python
"""MiniMax-H3 — fl2va: da un'IMMAGINE keyframe (primo e/o ultimo frame) -> video+audio.

Uso:
  env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python fl2va_gen.py <image.jpg> "<prompt>" [out.mp4] [last_image.jpg]

- <image.jpg>  = fotogramma da cui parte il video (il canvas segue il suo aspect ratio).
- [last_image] = opzionale, fotogramma su cui finisce (interpola image -> last_image).
Usa il workflow `fl2va` (partizione transformer/, gia' scaricata). 24fps, 5-15s.
"""
import sys, os, time, gc, torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL = "/home/michele/Scaricati/MiniMax-H3"   # checkpoint su Sullivan
NUM_FRAMES, STEPS, SEED = 124, 8, 42
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

image_path = sys.argv[1]
prompt = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "fl2va_out.mp4"
last_image_path = sys.argv[4] if len(sys.argv) > 4 else None
SKIP = ("visual", "embed_tokens", ".norm", "lm_head")

log("Carico la pipeline (workflow fl2va) — loader a shard per il conditioner ...")
pipe = ModularPipeline.from_pretrained(MODEL)
cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
with init_empty_weights():
    enc = Qwen3VLForConditionalGeneration(cfg)
enc = load_checkpoint_and_dispatch(enc, checkpoint=os.path.join(MODEL, "text_encoder"),
                                   device_map={"": 0}, dtype=torch.bfloat16)
quantize_(enc, Int8WeightOnlyConfig(version=2),
          filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and not any(s in fqn for s in SKIP))
gc.collect(); torch.cuda.empty_cache()
Q_DIT = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
dit = MiniMaxH3Transformer3DModel.from_pretrained(MODEL, subfolder="transformer",
        dtype=torch.bfloat16, quantization_config=Q_DIT, device_map="cuda")
gc.collect(); torch.cuda.empty_cache()
pipe.update_components(transformer=dit, text_encoder=enc)
pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
pipe.transformer.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
pipe.vae.to("cuda"); pipe.audio_vae.to("cuda")

kw = dict(prompt=prompt, image=load_image(image_path), num_frames=NUM_FRAMES,
          num_inference_steps=STEPS, generator=torch.Generator().manual_seed(SEED),
          output=["videos", "audio", "sampling_rate"])
if last_image_path:
    kw["last_image"] = load_image(last_image_path)
log(f"Genero da keyframe '{image_path}'{' -> '+last_image_path if last_image_path else ''} ...")
r = pipe(**kw)
encode_video(r["videos"][0], fps=24, output_path=out,
             audio=r["audio"][0], audio_sample_rate=r["sampling_rate"])
log(f"Salvato: {out}")
