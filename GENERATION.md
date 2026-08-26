# Provenienza dei sample — script + prompt per ogni video

Ogni clip è **riproducibile** con lo script e i parametri indicati. Tutti generati su **Sullivan**
(AMD Ryzen AI MAX+ 395 "Strix Halo", gfx1151, ROCm 7.2), sempre con:
```
env -u HSA_OVERRIDE_GFX_VERSION PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python <script> ...
```
Modello: `MiniMaxAI/MiniMax-H3` (layout diffusers). Default: 24 fps, seed 42, 8 inference steps.

## Mappa video → script + parametri

| Video | Script | Prompt | Canvas | Frame (durata) | Step | Tempo |
|---|---|---|---|---|---|---|
| `minimax-h3-t2va-fox-960x544.mp4` | `t2va_test.py` | *fox* | 960×544 | 124 (5,2 s) | default | ~2h11m |
| `minimax-h3-t2va-fox-512x288-8steps.mp4` | `t2va_test.py` | *fox* | 512×288 | 124 (5,2 s) | 8 | 141 s |
| `batch/volpe-neve.mp4` | `batch_gen.py` (`prompts.json`) | *fox/volpe* | 512×288 | 124 (5,2 s) | 8 | ~141 s |
| `batch/onde-tramonto.mp4` | `batch_gen.py` (`prompts.json`) | *onde* | 512×288 | 124 (5,2 s) | 8 | ~141 s |
| `batch/citta-pioggia.mp4` | `batch_gen.py` (`prompts.json`) | *città* | 512×288 | 124 (5,2 s) | 8 | ~141 s |
| `batch/maestro-taichi.mp4` | `batch_gen.py` (`prompts.json`) | *taichi* | 512×288 | 345 (14,4 s) | 8 | ~15 min |
| `batch/maestro-taichi-960x544.mp4` | `batch_gen.py` (`prompts_taichi960.json`, 124f) | *taichi* | 960×544 | 124 (5,2 s) | 8 | 1159 s |
| `batch/maestro-taichi-960x544-14s.mp4` | `t2va_offload.py` (`prompts_taichi960.json`, 345f) | *taichi* | 960×544 | 345 (14,4 s) | 8 | ~3h15m |

Esempi di comando:
```
# singola clip
env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python t2va_test.py - 124 512x288 8
# batch (tutti i prompt di prompts.json)
env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python batch_gen.py prompts.json batch_out
# durata piena ad alta risoluzione (offload del conditioner)
env -u HSA_OVERRIDE_GFX_VERSION .venv/bin/python t2va_offload.py prompts_taichi960.json out
```

## Prompt completi

Struttura a tre canali (vedi MANUALE.md §2): descrizione visiva + soundscape + musica.

### fox / volpe-neve
```
integrated_multimodal_description: [Shot 1] Cinematic medium shot, slow push-in. A red fox trots through a snowy pine forest at dawn, soft golden light filtering through the trees, snow crunching under its paws, breath visible in the cold air.
overall_soundscape: quiet forest ambience, gentle wind, the soft crunch of snow underfoot, a distant bird call.
non_diegetic_music: a light, warm acoustic guitar melody, calm and unhurried.
```

### onde-tramonto
```
integrated_multimodal_description: [Shot 1] Wide static shot at golden hour. Ocean waves roll gently onto an empty sandy beach, warm sunset light shimmering on the wet sand, seafoam sliding back into the water.
overall_soundscape: rhythmic ocean waves, distant seagulls, a soft breeze.
non_diegetic_music: mellow ambient synth pads, peaceful and slow.
```

### citta-pioggia
```
integrated_multimodal_description: [Shot 1] Cinematic shot, shallow depth of field. A neon-lit city street at night in the rain, colorful reflections on the wet asphalt, blurred headlights passing by.
overall_soundscape: steady rain, distant traffic, tires hissing on wet road.
non_diegetic_music: moody lo-fi beat, mellow and atmospheric.
```

### maestro-taichi (usato per le 3 versioni: 512×288/14,4 s, 960×544/5 s, 960×544/14,4 s)
```
integrated_multimodal_description: [Shot 1] Cinematic wide shot, static camera, soft natural light from tall windows. In a spacious martial arts studio with a warm honey-toned parquet wood floor, an elderly Tai Chi master in a flowing white silk uniform stands centered, demonstrating the "Wave Hands Like Clouds" movement. He shifts his weight slowly from one leg to the other, arms circling smoothly and continuously in front of his torso, hands rotating with deliberate control, his movements fluid and unhurried. Calm focused expression, relaxed posture, precise.
overall_soundscape: quiet studio ambience, the soft creak of the wooden floor, faint sound of bare feet pivoting on parquet, slow deep breathing, gentle rustle of the silk uniform.
non_diegetic_music: soft ambient oriental music, a slow guzheng melody over a bamboo flute and warm sustained pads, meditative and serene.
```
