#!/usr/bin/env python
"""
MiniMax-H3 — test text-to-video+audio (t2va) su Sullivan (Strix Halo gfx1151).

Strategia di memoria specifica per Sullivan:
  - VRAM enorme (96 GB) ma RAM di sistema piccola (32 GB, per il carve-out BIOS).
  - Il modello pieno bf16 (~135 GB) NON entra nei 128 GB totali.
  - Quindi: quantizzo transformer + conditioner a int8 (~75 GB) e carico
    TUTTO in VRAM (no offload verso la RAM di sistema, che qui è troppo piccola).

Uso:
  env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python t2va_test.py [prompt] [num_frames] [WxH]
  (l'unset di HSA_OVERRIDE serve per usare i kernel NATIVI gfx1151, non la maschera gfx1100)
"""
import sys, time, torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

MODEL = "/home/michele/Scaricati/MiniMax-H3"

# --- parametri (con default sensati per una PRIMA prova veloce) ---
prompt = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else (
    "integrated_multimodal_description: [Shot 1] Cinematic medium shot, slow push-in. "
    "A red fox trots through a snowy pine forest at dawn, soft golden light filtering "
    "through the trees, snow crunching under its paws, breath visible in the cold air.\n"
    "overall_soundscape: quiet forest ambience, gentle wind, the soft crunch of snow underfoot, "
    "a distant bird call.\nnon_diegetic_music: a light, warm acoustic guitar melody, calm and unhurried."
)
num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 124  # 17*n+5; 124 = ~5.17s @24fps (durata minima)
if len(sys.argv) > 3:
    width, height = (int(x) for x in sys.argv[3].lower().split("x"))
else:
    width, height = 960, 544  # canvas piccola = ~2.3x piu' veloce (multipli di 32)
steps = int(sys.argv[4]) if len(sys.argv) > 4 else None  # None = default del modello

Q_DIT = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
Q_ENC = TransformersTorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"])

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("Costruisco la pipeline modulare...")
pipe = ModularPipeline.from_pretrained(MODEL)

import gc, os
import torch.nn as _nn
from torchao.quantization import quantize_ as _quantize_
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig

# Sullivan ha solo ~30GB di RAM: from_pretrained (cuda/auto) accumula i 63GB bf16 in RAM -> swap thrash.
# LOADER MANUALE: init su meta (zero RAM) + load_checkpoint_and_dispatch che legge i 14 shard
# uno alla volta scrivendoli DIRETTAMENTE in VRAM (RAM = ~un tensore alla volta).
log("Carico text_encoder Qwen3-VL con loader a shard diretto in VRAM ...")
_cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
with init_empty_weights():
    enc = Qwen3VLForConditionalGeneration(_cfg)
enc = load_checkpoint_and_dispatch(
    enc, checkpoint=os.path.join(MODEL, "text_encoder"),
    device_map={"": 0}, dtype=torch.bfloat16)
log(f"  bf16 in VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB; quantizzo a int8 su GPU ...")
_enc_skip = ("visual", "embed_tokens", ".norm", "lm_head")
_quantize_(enc, Int8WeightOnlyConfig(version=2),
           filter_fn=lambda m, fqn: isinstance(m, _nn.Linear) and not any(s in fqn for s in _enc_skip))
gc.collect(); torch.cuda.empty_cache()
log(f"  text_encoder OK (int8 su GPU), VRAM allocata {torch.cuda.memory_allocated()/1e9:.1f} GB")

log("Carico transformer (int8) ...")
dit = MiniMaxH3Transformer3DModel.from_pretrained(
    MODEL, subfolder="transformer", dtype=torch.bfloat16,
    quantization_config=Q_DIT, device_map="cuda")
gc.collect(); torch.cuda.empty_cache()
log(f"  transformer OK, VRAM allocata {torch.cuda.memory_allocated()/1e9:.1f} GB")
pipe.update_components(transformer=dit, text_encoder=enc)

log("load_components(workflow=t2va) ...")
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
pipe.transformer.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

# transformer e text_encoder sono gia' in VRAM (device_map). Sposto VAE piccoli.
log("Sposto i VAE in VRAM (cuda) ...")
pipe.vae.to("cuda")
pipe.audio_vae.to("cuda")
log(f"VRAM allocata: {torch.cuda.memory_allocated()/1e9:.1f} GB")

log(f"Genero: {width}x{height}, {num_frames} frame (~{num_frames/24:.1f}s), steps={steps or 'default'} ...")
t0 = time.time()
kw = dict(prompt=prompt, width=width, height=height, num_frames=num_frames,
          generator=torch.Generator().manual_seed(42),
          output=["videos", "audio", "sampling_rate"])
if steps is not None:
    kw["num_inference_steps"] = steps
results = pipe(**kw)
log(f"Generazione completata in {time.time()-t0:.0f}s")

out = f"/home/michele/Documenti/Workspaces/minimax-h3/t2va_out_s{steps or 'def'}.mp4"
encode_video(results["videos"][0], fps=24, output_path=out,
             audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
log(f"Salvato: {out}")
