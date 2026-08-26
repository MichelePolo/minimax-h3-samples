#!/usr/bin/env python
"""MiniMax-H3 — ref2va: RIFERIMENTI (immagini/video/audio) -> video+audio.
E' il workflow per la COERENZA DEI PERSONAGGI (vedi MANUALE.md §6).

Uso:
  env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python ref2va_gen.py "<prompt>" <ref1> [ref2] ...
  ref = .jpg/.png (immagine, max 9) | .mp4/.mov (video, max 3) | .wav/.mp3 (audio, max 3).
  L'ORDINE e' semantico (etichetta <Picture 1>/<Video 1>/<Audio 1> nel prompt).
  Un audio non puo' essere l'unico riferimento: va con >=1 immagine/video.

RICHIEDE la cartella `transformer_ref/` nel checkpoint (~66 GB), che su Sullivan
NON e' ancora scaricata. Scaricarla prima (stesso metodo delle altre cartelle top-level).
"""
import sys, os, time, gc, torch
import torch.nn as nn
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference)
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL = "/home/michele/Scaricati/MiniMax-H3"
NUM_FRAMES, STEPS, SEED = 124, 8, 42
W, H = 512, 288   # forza canvas piccola: senza, ref2va usa il default ~768 (16:9) -> OOM sui 96GB
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

if not os.path.isdir(os.path.join(MODEL, "transformer_ref")):
    sys.exit("ERRORE: manca la cartella transformer_ref/ (~66 GB). Scaricala prima (vedi MANUALE.md §8).")

prompt = sys.argv[1]
ref_paths = sys.argv[2:]
if not ref_paths:
    sys.exit("Passa almeno un riferimento (immagine/video/audio).")

IMG = (".jpg", ".jpeg", ".png", ".webp"); VID = (".mp4", ".mov", ".mkv", ".webm"); AUD = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
def make_ref(p):
    e = os.path.splitext(p)[1].lower()
    if e in IMG: return MiniMaxH3ImageReference.from_file(p)   # from_file porta con se' i rate
    if e in VID: return MiniMaxH3VideoReference.from_file(p)
    if e in AUD: return MiniMaxH3AudioReference.from_file(p)
    raise ValueError(f"estensione non riconosciuta: {p}")
refs = [make_ref(p) for p in ref_paths]
SKIP = ("visual", "embed_tokens", ".norm", "lm_head")

log("Carico la pipeline (workflow ref2va -> transformer_ref) ...")
pipe = ModularPipeline.from_pretrained(MODEL)
cfg = AutoConfig.from_pretrained(MODEL, subfolder="text_encoder")
with init_empty_weights():
    enc = Qwen3VLForConditionalGeneration(cfg)
enc = load_checkpoint_and_dispatch(enc, checkpoint=os.path.join(MODEL, "text_encoder"),
                                   device_map={"": 0}, dtype=torch.bfloat16)
quantize_(enc, Int8WeightOnlyConfig(version=2),
          filter_fn=lambda m, fqn: isinstance(m, nn.Linear) and not any(s in fqn for s in SKIP))
gc.collect(); torch.cuda.empty_cache()
Q = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
dit = MiniMaxH3Transformer3DModel.from_pretrained(MODEL, subfolder="transformer_ref",
        dtype=torch.bfloat16, quantization_config=Q, device_map="cuda")
gc.collect(); torch.cuda.empty_cache()
pipe.update_components(transformer_ref=dit, text_encoder=enc)
pipe.load_components(workflow="ref2va", dtype=torch.bfloat16)
pipe.text_encoder.requires_grad_(False)
pipe.vae.to("cuda"); pipe.audio_vae.to("cuda")

log(f"Genero con {len(refs)} riferimenti ...")
r = pipe(prompt=prompt, references=refs, width=W, height=H, num_frames=NUM_FRAMES, num_inference_steps=STEPS,
         generator=torch.Generator().manual_seed(SEED), output=["videos", "audio", "sampling_rate"])
encode_video(r["videos"][0], fps=24, output_path="ref2va_out.mp4",
             audio=r["audio"][0], audio_sample_rate=r["sampling_rate"])
log("Salvato: ref2va_out.mp4")
