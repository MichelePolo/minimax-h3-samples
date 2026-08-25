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
