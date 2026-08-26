#!/usr/bin/env python
"""MiniMax-H3 t2va con OFFLOAD del conditioner: codifica il prompt col Qwen3-VL, poi lo
LIBERA dalla VRAM (~35 GB) prima del denoising -> permette durate lunghe ad alta risoluzione
(es. 960x544 / 345 frame / 14,4 s) che altrimenti vanno OOM.

Uso: env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python t2va_offload.py [prompts.json] [out_dir]
"""
import sys, os, json, time, gc, torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL = "/home/michele/Scaricati/MiniMax-H3"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def vram(): return torch.cuda.memory_allocated() / 1e9

prompts_path = sys.argv[1] if len(sys.argv) > 1 else "prompts_taichi960.json"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "offload_out"
os.makedirs(out_dir, exist_ok=True)
job = json.load(open(prompts_path))[0]
name = job.get("name", "clip")
W, H, NF = job.get("width", 960), job.get("height", 544), job.get("num_frames", 345)
STEPS, SEED = job.get("steps", 8), job.get("seed", 42)
ENC_SKIP = ("visual", "embed_tokens", ".norm", "lm_head")

# ---------- STAGE 1: conditioner (codifica il prompt, poi lo liberiamo) ----------
log("Costruisco il workflow t2va e separo il blocco text_encoder ...")
workflow = ModularPipeline.from_pretrained(MODEL).blocks.get_workflow("t2va")
cond_blocks = workflow.sub_blocks.pop("text_encoder")
conditioner = cond_blocks.init_pipeline(MODEL)

log("Carico il text_encoder Qwen3-VL con loader a shard (int8, diretto in VRAM) ...")
cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
with init_empty_weights():
    enc = Qwen3VLForConditionalGeneration(cfg)
enc = load_checkpoint_and_dispatch(enc, checkpoint=os.path.join(MODEL, "text_encoder"),
                                   device_map={"": 0}, dtype=torch.bfloat16)
quantize_(enc, Int8WeightOnlyConfig(version=2),
          filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and not any(s in fqn for s in ENC_SKIP))
gc.collect(); torch.cuda.empty_cache()
# inietto il nostro encoder e carico SOLO i componenti piccoli (tokenizer/processor)
conditioner.update_components(text_encoder=enc)
conditioner.load_components(dtype=torch.bfloat16)
log(f"Conditioner pronto (VRAM {vram():.1f} GB). Codifico il prompt ...")
state = conditioner(prompt=job["prompt"])

# ---------- OFFLOAD: libera il conditioner ----------
del conditioner, enc, cond_blocks
gc.collect(); torch.cuda.empty_cache()
log(f"Conditioner LIBERATO -> VRAM ora {vram():.1f} GB (dovrebbe essere crollata)")

# ---------- STAGE 2: denoiser (transformer int8 + VAE) ----------
Q_DIT = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
log("Carico il denoiser (transformer int8) ...")
rest = workflow.init_pipeline(MODEL)
dit = MiniMaxH3Transformer3DModel.from_pretrained(
    MODEL, subfolder="transformer", dtype=torch.bfloat16,
    quantization_config=Q_DIT, device_map="cuda")
rest.update_components(transformer=dit)
rest.load_components(dtype=torch.bfloat16)
try:
    rest.vae.to("cuda"); rest.audio_vae.to("cuda")
except Exception as e:
    log(f"(vae .to cuda: {e!r})")
gc.collect(); torch.cuda.empty_cache()
log(f"Denoiser pronto (VRAM {vram():.1f} GB). Genero {W}x{H} {NF}f (~{NF/24:.1f}s) steps={STEPS} ...")

t0 = time.time()
results = rest(state=state, width=W, height=H, num_frames=NF, num_inference_steps=STEPS,
               generator=torch.Generator().manual_seed(SEED),
               output=["videos", "audio", "sampling_rate"])
log(f"Generazione completata in {time.time()-t0:.0f}s")
out = os.path.join(out_dir, f"{name}.mp4")
encode_video(results["videos"][0], fps=24, output_path=out,
             audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
log(f"Salvato: {out}")
