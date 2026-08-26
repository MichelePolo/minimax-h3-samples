# MiniMax-H3 — sample generati in locale

Video di esempio generati con **MiniMax-H3** (modello omni-modale text-to-video+audio, DiT 33B)
eseguito **in locale** su "Sullivan" (GMKtec EVO-X2, AMD Ryzen AI MAX+ 395 "Strix Halo",
GPU gfx1151, 96 GB VRAM, ROCm 7.2) tramite `diffusers` + PyTorch-ROCm, quantizzazione int8.

## Sample

### `minimax-h3-t2va-fox-960x544.mp4`
- **Task**: `t2va` (text → video + audio)
- **Prompt**: una volpe rossa che trotta in una foresta di pini innevata all'alba, luce dorata tra gli alberi; ambience di foresta + musica acustica calma.
- **Output**: H.264 **960×544**, **5,17 s** (124 frame @ 24 fps), audio **AAC stereo 32 kHz** (generato *insieme* al video).
- **Generazione**: ~2h 11m (int8 weight-only, `num_inference_steps` di default, nessun kernel ottimizzato).

## Note
Primo **proof-of-concept**: dimostra che H3 gira in locale sull'APU Strix Halo. Le prestazioni
(ore per pochi secondi) lo rendono adatto a **generazione batch/offline**, non in tempo reale.
Ottimizzazione in corso (flash-attention ROCm, riduzione step, tuning MIOpen).

### `minimax-h3-t2va-fox-512x288-8steps.mp4` (ottimizzato per batch)
Stessa scena, versione **ottimizzata**: **512×288**, **8 inference steps**, 5,17 s, audio stereo.
- **Generazione: ~141 s** (contro ~2h 11m del primo sample), dopo un caricamento una-tantum di ~1m40s.
- Qualità coerente (il modello è guidance-distilled). Adatto a **generazione batch offline**.
- Sblocco chiave: loader a shard (`init_empty_weights` + `load_checkpoint_and_dispatch`) che carica il conditioner da 63 GB diretto in VRAM, evitando lo swap dei 30 GB di RAM di sistema.

## Batch samples (`batch/`)
Generati in un unico run con `batch_gen.py` (carica il modello **una volta**, poi cicla sui prompt). Tutti **512×288, 8 step**, audio stereo nativo.
- `volpe-neve.mp4` — volpe rossa nella foresta innevata all'alba (5,2 s)
- `onde-tramonto.mp4` — onde dell'oceano al tramonto, ora dorata (5,2 s)
- `citta-pioggia.mp4` — strada cittadina al neon sotto la pioggia, di notte (5,2 s)
- `maestro-taichi.mp4` — maestro di Tai Chi ("Wave Hands Like Clouds") in palestra con parquet, musica ambient orientale (**14,4 s**, durata massima di H3)

Nota tempi (8 step): 512×288 ≈ 141 s/clip da 5 s; 960×544 ≈ 1181 s (l'attenzione cresce col quadrato dei token → 512×288 è il punto dolce per il batch).

### `batch/maestro-taichi-960x544.mp4` (hero, alta risoluzione)
Il maestro di Tai Chi a **960×544** (5,2 s, ~19 min di generazione, audio stereo). A questa risoluzione la durata piena 14,4 s va in **OOM** sui 96 GB (pesi ~70 GB + attivazioni di 345 frame sforano); 5 s è il massimo sicuro senza offload del conditioner.

## 📖 Manuale d'uso completo
Vedi **[MANUALE.md](MANUALE.md)** — tutti i casi d'uso di MiniMax-H3: i 3 workflow (`t2va`/`fl2va`/`ref2va`), immagini come keyframe, video/audio come riferimenti, **coerenza dei personaggi** per storie multi-shot, vincoli, e le ricette pratiche su Sullivan (loader a shard, offload del conditioner, tempi).

### `batch/maestro-taichi-960x544-14s.mp4` (durata piena, via offload)
Il maestro a **960×544 per 14,4 s** (durata massima di H3), reso possibile dall'**offload del conditioner** (`t2va_offload.py`): libera ~35 GB del Qwen3-VL dopo la codifica del prompt → i 345 frame entrano in VRAM (prima andavano OOM). Costo: **~3h15m** di generazione (960×544 a piena durata è molto pesante). Audio stereo.
