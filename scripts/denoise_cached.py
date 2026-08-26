#!/usr/bin/env python
"""FASE 2 — Genera i video dagli stati di condizionamento gia' calcolati, SENZA Qwen-VL.
Carica solo il denoiser (transformer int8 + VAE), legge gli .pt della cache e genera un mp4
per stato. Qwen-VL (63 GB) NON viene mai caricato -> avvio piu' rapido, meno memoria.
Lo stesso stato .pt puo' essere reso a qualsiasi risoluzione/durata (li passi qui).

Uso: env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python denoise_cached.py <cache_dir> <out_dir> [WxH] [num_frames] [steps]
"""
import sys, os, glob, time, gc, torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.utils.export_utils import encode_video
from torchao.quantization import Int8WeightOnlyConfig

MODEL = "/home/michele/Scaricati/MiniMax-H3"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

cache_dir = sys.argv[1] if len(sys.argv) > 1 else "cond_cache"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "cached_out"
W, H = (int(x) for x in (sys.argv[3].lower().split("x") if len(sys.argv) > 3 else ["512", "288"]))
NF = int(sys.argv[4]) if len(sys.argv) > 4 else 124
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SEED = 42
os.makedirs(out_dir, exist_ok=True)
states = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
if not states:
    sys.exit(f"Nessuno stato .pt in {cache_dir}/ (esegui prima encode_prompt.py)")

# --- carica SOLO il denoiser (niente Qwen-VL) ---
log("Carico il denoiser (transformer int8 + VAE), senza Qwen-VL ...")
workflow = ModularPipeline.from_pretrained(MODEL).blocks.get_workflow("t2va")
workflow.sub_blocks.pop("text_encoder")            # niente conditioner nella pipeline
rest = workflow.init_pipeline(MODEL)
Q_DIT = TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
dit = MiniMaxH3Transformer3DModel.from_pretrained(MODEL, subfolder="transformer",
        dtype=torch.bfloat16, quantization_config=Q_DIT, device_map="cuda")
rest.update_components(transformer=dit)
rest.load_components(dtype=torch.bfloat16)
rest.vae.to("cuda"); rest.audio_vae.to("cuda")
gc.collect(); torch.cuda.empty_cache()
log(f"Denoiser pronto (VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB). {len(states)} stati da rendere -> {W}x{H} {NF}f {STEPS}step")

ok = 0
for i, sp in enumerate(states):
    name = os.path.splitext(os.path.basename(sp))[0]
    try:
        data = torch.load(sp, weights_only=True)          # solo tensori/valori -> sicuro
        st = PipelineState()
        for k, v in data.items():
            if torch.is_tensor(v) and v.is_floating_point():
                v = v.to("cuda")                          # prompt_embeds in VRAM
            st.set(k, v)
        log(f"[{i+1}/{len(states)}] {name}: genero ...")
        t0 = time.time()
        r = rest(state=st, width=W, height=H, num_frames=NF, num_inference_steps=STEPS,
                 generator=torch.Generator().manual_seed(SEED),
                 output=["videos", "audio", "sampling_rate"])
        out = os.path.join(out_dir, f"{name}.mp4")
        encode_video(r["videos"][0], fps=24, output_path=out,
                     audio=r["audio"][0], audio_sample_rate=r["sampling_rate"])
        log(f"    OK in {time.time()-t0:.0f}s -> {out}")
        ok += 1
    except Exception as e:
        log(f"    ERRORE su {name}: {e!r}")

log(f"FATTO: {ok}/{len(states)} clip generate in {out_dir}/ (senza mai caricare Qwen-VL)")
