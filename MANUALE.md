# MiniMax-H3 — Manuale d'uso (su Sullivan)

Modello **omni-modale di generazione video+audio**: legge testo, immagini, video e audio come
un unico contesto e produce **video con colonna sonora stereo** generata *insieme* al video
(niente vocoder separato). Open-weight (MIT preview), DiT da 33B. Su Sullivan gira via
**diffusers + PyTorch-ROCm** (gfx1151), quantizzazione int8, con un loader a shard custom.

- **Vincoli fissi**: 24 fps, durata **5–15 s**, `num_frames = 17·n + 5`, lati **multipli di 32**.
- **Guidance-distilled**: niente `negative_prompt`, niente `guidance_scale`; **pochi step bastano** (8 rende bene).
- **Risoluzione**: 512×288 è il punto dolce (~141 s/clip da 5 s); 960×544 costa ~8× (attenzione quadratica) e a 96 GB regge ~5 s.

---

## 1. I tre modi d'uso (workflow)

Un'unica pipeline (`MiniMaxH3ModularPipeline`) con tre workflow, scelti in base agli input passati:

| Workflow | Input richiesti | Cosa fa | Partizione pesi |
|---|---|---|---|
| **`t2va`** | `prompt` | **testo → video+audio** | `transformer/` |
| **`fl2va`** | `prompt` + `image` e/o `last_image` | **keyframe (primo/ultimo frame) → video+audio** | `transformer/` |
| **`ref2va`** | `prompt` + `references` | **riferimenti (immagini/video/audio) → video+audio** | `transformer_ref/` |

`t2va` e `fl2va` condividono la stessa partizione del transformer; `ref2va` ne usa una dedicata
(`transformer_ref/`, da scaricare a parte — vedi §8).

---

## 2. Come si scrivono i prompt

Convenzione a **tre sezioni** (una per canale), così il modello non confonde ciò che *vede* con
ciò che *sente*:

```
integrated_multimodal_description: <descrizione visiva>. Puoi usare più inquadrature
  [Shot 1] ... [Shot 2] ... e timecode espliciti (es. "At 00:04.500, the camera cuts to...").
overall_soundscape: <suoni diegetici della scena: ambiente, passi, pioggia, respiro...>
non_diegetic_music: <colonna sonora: strumenti, tempo, mood>
```

Consigli:
- **Scrivi in inglese** (rese migliori: il modello è addestrato soprattutto su descrizioni EN).
- Sii cinematografico: tipo di inquadratura, movimento camera, luce, materiali.
- Per una mini-regia usa `[Shot N]` + timecode; l'audio si sincronizza con gli eventi descritti.

---

## 3. `t2va` — testo → video+audio (il caso base)

```python
results = pipe(
    prompt="integrated_multimodal_description: ...\noverall_soundscape: ...\nnon_diegetic_music: ...",
    width=512, height=288,          # multipli di 32
    num_frames=124,                 # 17·n+5; 124 ≈ 5,2 s (durata minima)
    num_inference_steps=8,          # 8 rende bene (guidance-distilled)
    generator=torch.Generator().manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)
encode_video(results["videos"][0], fps=24, output_path="out.mp4",
             audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
```

Su Sullivan: usa `t2va_test.py` (singola clip) o `batch_gen.py` + `prompts.json` (batch). Vedi §8.

---

## 4. Immagini come input — `fl2va` (keyframe)

Serve a **far partire (e/o finire) il video da un'immagine data**. Il canvas segue di default
l'aspect ratio del primo keyframe.

```python
from diffusers.utils import load_image
image = load_image("prima_immagine.jpg")          # o URL

# a) Primo frame → video: l'immagine è il fotogramma iniziale, il modello anima da lì
results = pipe(prompt="...", image=image, num_frames=124,
               generator=torch.Generator().manual_seed(42),
               output=["videos","audio","sampling_rate"])

# b) Ultimo frame: passa last_image (da solo = "arriva a" quel frame)
results = pipe(prompt="...", last_image=load_image("finale.jpg"), num_frames=124, ...)

# c) Primo E ultimo: interpola tra i due (image = di partenza, last_image = di arrivo)
results = pipe(prompt="...", image=image, last_image=load_image("finale.jpg"), num_frames=124, ...)
```

Note:
- `image` viene **stirata** sul canvas (che di default deriva dal suo aspect ratio).
- `last_image` combinata con `image` è "l'inseguitrice", ritagliata (cover-crop) sul canvas.
- Usi tipici: animare una foto/still, morphing tra due immagini, intro/outro controllati.

---

## 5. Video e audio come input — `ref2va` (riferimenti omni)

`ref2va` condiziona su una **lista ordinata di riferimenti** (fino a **12**: ≤9 immagini,
≤3 video, ≤3 audio). L'**ordine è semantico**: etichetta i riferimenti nel prompt come
`<Picture 1>`, `<Video 1>`, `<Audio 1>` e li dispone sul "clock" temporale condiviso, quindi
riordinarli = richiesta diversa. I riferimenti **non** vincolano la geometria (canvas default 16:9).

```python
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference,
)

refs = [
    MiniMaxH3ImageReference.from_file("personaggio.jpg"),     # soggetto/stile/scena (max 9)
    MiniMaxH3VideoReference.from_file("movimento.mp4"),       # riferimento di movimento/camera (max 3, con la sua audio)
    MiniMaxH3AudioReference.from_file("voce.wav"),            # voce/musica (max 3, MAI da sola: va con ≥1 img/video)
]
results = pipe(prompt="The character speaks in time with the reference recording ...",
               references=refs, num_frames=124,
               output=["videos","audio","sampling_rate"])
```

Regole d'oro dei riferimenti:
- **Decodifica sempre con `from_file`** (porta con sé i "rate": fps del video, sample_rate dell'audio).
  `load_video()` **perde** il frame-rate → il riferimento verrebbe condizionato alla velocità sbagliata.
- Le tre classi: `MiniMaxH3ImageReference(image)`, `MiniMaxH3VideoReference(frames, fps, audio?, sample_rate?)`,
  `MiniMaxH3AudioReference(audio, sample_rate?)`.
- Video di riferimento: porta anche la **sua** colonna sonora come condizionamento; lascialo senza
  audio per condizionare **solo sul movimento**.
- `num_frames` è **obbligatorio** in `ref2va`. Per una durata pari a una traccia audio:
  `round(samples / sample_rate * 24)` (poi arrotondato al `17·n+5` più vicino, entro 5–15 s).

---

## 6. Coerenza dei personaggi (storie multi-shot)

Tre tecniche, combinabili:

### 6a. Riferimento immagine del personaggio (`ref2va`)
Passa una foto/ritratto del personaggio come `MiniMaxH3ImageReference`: il video mantiene quel
soggetto. Puoi darne fino a 9 (es. viso + outfit + oggetto di scena) per un controllo forte.

### 6b. **Una generazione come riferimento** (la chiave per le storie)
Il punto forte di H3: una clip **`t2va` appena generata** può essere reimmessa come **riferimento
`ref2va`** per la scena successiva, mantenendo **lo stesso personaggio e stile**. La media generata
è già ai rate nativi (24 fps, sample_rate del VAE audio), quindi il costruttore in-memory non
ri-codifica nulla:

```python
# Shot 1 (t2va)
r1 = pipe(prompt="An astronaut hiking through the mountains, humming a tune",
          num_frames=124, output=["videos","audio","sampling_rate"])

# Shot 2 (ref2va): stesso astronauta, nuova scena — si passa il risultato come VideoReference
ref = MiniMaxH3VideoReference(frames=r1["videos"][0], audio=r1["audio"][0],
                              sample_rate=r1["sampling_rate"])
r2 = pipe(prompt="The same astronaut now walks along a beach at sunset, humming the same tune",
          references=[ref], num_frames=124, output=["videos","audio","sampling_rate"])
```

Concatenando Shot 1 → 2 → 3 … costruisci una **storia** con il protagonista coerente. Puoi anche
mescolare: un `ImageReference` del volto + un `VideoReference` della scena precedente.

### 6c. Seed + descrizione stabili
Riusare lo stesso `seed` e ripetere nel prompt i tratti fissi del personaggio (età, corporatura,
abito, colori) aiuta la coerenza anche in `t2va`, entro i limiti.

---

## 7. Vincoli e parametri (riferimento rapido)

| Parametro | Valore |
|---|---|
| `num_frames` | `17·n + 5`, con durata risultante **5–15 s** a 24 fps (es. 124=5,2 s; 345=14,4 s) |
| `width`, `height` | multipli di **32** (es. 512×288, 960×544) |
| `num_inference_steps` | 8 consigliato (guidance-distilled); più step = più tempo, poco guadagno |
| `generator` | `torch.Generator().manual_seed(seed)` — stesso seed = stesso output |
| `output` | `["videos","audio","sampling_rate"]` |
| assenti | `negative_prompt`, `guidance_scale` (baked-in) |

---

## 8. Prestazioni e ricette pratiche su Sullivan

- **Ambiente**: venv Python 3.12 in `~/Documenti/Workspaces/minimax-h3/.venv`; PyTorch 2.13+rocm7.2
  (gfx1151 nativo). Lanciare **sempre** con `env -u HSA_OVERRIDE_GFX_VERSION` (l'override serve a Ollama,
  non a questo stack) e `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.
- **Pesi**: `~/Scaricati/MiniMax-H3/` (layout diffusers top-level). `ref2va` richiede anche la cartella
  `transformer_ref/` (~66 GB), non ancora scaricata: serve per i riferimenti (§5).
- **Caricamento (loader a shard)**: il conditioner Qwen3-VL da 63 GB non si carica con `from_pretrained`
  (accumula in RAM → swap thrash, i 30 GB di RAM di sistema non bastano). Si usa `init_empty_weights()` +
  `load_checkpoint_and_dispatch(..., device_map={"":0}, dtype=bf16)` → shard diretti in VRAM (~48 s), poi
  `torchao.quantize_` a int8 su GPU.
- **Tempi (8 step)**: 512×288 ≈ **141 s**/clip da 5 s; 960×544 ≈ **1181 s** (5 s). 512×288 = punto dolce batch.
- **Memoria**: pesi ~70 GB int8 in VRAM; a 960×544 la durata piena 14,4 s va **OOM** (serve l'**offload del
  conditioner** dopo la codifica per liberare ~35 GB — vedi `t2va_offload.py`).

### Script pronti
- **`t2va_test.py`** — singola clip: `... t2va_test.py "<prompt>|-" <num_frames> <WxH> <steps>`
- **`batch_gen.py`** + **`prompts.json`** — batch: carica una volta, cicla sui prompt, un `.mp4` per prompt.
  Campi per-clip opzionali: `width`, `height`, `num_frames`, `steps`, `seed`.
  ```
  cd ~/Documenti/Workspaces/minimax-h3
  env -u HSA_OVERRIDE_GFX_VERSION ./.venv/bin/python batch_gen.py prompts.json batch_out
  ```
- **`t2va_offload.py`** — variante con offload del conditioner per durate lunghe ad alta risoluzione.

### Uso consigliato per la tua app
Genera **in batch offline** (512×288 per il volume, 960×544 per i clip "hero" brevi), l'app fa solo
**streaming** dei video già pronti. Per storie con personaggi coerenti: `t2va` per lo Shot 1, poi
`ref2va` reimmettendo il risultato come `MiniMaxH3VideoReference` per gli shot successivi (§6b).
