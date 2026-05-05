# whisper-subtitles — Plan

Generador de subtítulos `.srt` desde videos usando Whisper local
(`whisper.cpp`). Sin llamadas a APIs externas, sin que tus datos salgan
de tu máquina.

## Donde estamos hoy (Phase 1.5)

### Pipeline funcional end-to-end
```
video → ffmpeg (WAV 16kHz mono) → whisper.cpp (JSON, word-level) → chunker → .srt
```

El JSON intermedio con timestamps por palabra es la **source of truth**
del sistema. Todo lo demás (chunking, formato de salida, reglas) opera
encima.

### Lo que ya hace
- **Multi-modelo**: `large-v3-turbo` (default, rápido) y `large-v3` (max calidad).
- **Multi-idioma**: auto-detect o forzar (`--language es`, `en`, etc).
- **Tres presets de chunking**:
  - `traditional` (default) — Netflix-style, 42×2 chars, max 6s. Ideal para videos largos.
  - `social` — TikTok/Reels, 30×2 chars + overflow soft, orphan absorption, max 3s. Para virales.
  - `karaoke` — palabra por palabra, para el efecto pop word-by-word.
- **Chunker inteligente** con prioridades: oración (`. ? !`) > cláusula (`, ; :`) > greedy.
- **Orphan absorption**: combina cues huérfanos de 1-2 palabras con el cue anterior usando margen de overflow.

### Performance baseline (Apple M4 Pro, video de 70s)
| Modelo | Tiempo | Velocidad | RAM peak |
|---|---|---|---|
| Turbo | 5.10s | 13.8× realtime | 1.98 GB |
| Large-v3 | 14.70s | 4.8× realtime | 4.17 GB |

## A donde vamos

### Phase 1.5 — pulir el motor base
Mejoras sin cambiar la arquitectura. Trackeadas como issues etiquetadas
`phase-1.5`:

- Tests unitarios sobre chunker, srt formatting, presets
- Progress feedback durante transcripciones largas
- Setup script para Windows (PowerShell + CUDA)
- Aplicar orphan absorption también a `traditional`
- Investigar balanced line wrap (anti-greedy)

### Phase 2 — modelo de Clientes
El gran salto: cada video ya no es solo un video. Es un video **para un
cliente**. Cada cliente tiene reglas custom de subtitulado.

Issues etiquetadas `phase-2`:

- **Client model + persistencia** (yaml/json/sqlite)
- **Glosarios y reemplazos**: ej. "dios" → "DIOS", "ia" → "IA", regexes
  configurables.
- **Overrides de presets**: cliente arranca con un preset base
  (`traditional`/`social`/`karaoke`) y le sobreescribe campos puntuales
  (`max_duration_seconds`, `max_chars_per_line`, etc).
- **Dashboard web** para gestionar clientes y disparar transcripciones.

### Phase 3 — features avanzados
Issues etiquetadas `phase-3` o `future`:

- Speaker diarization → orphan absorption no fusiona cues entre
  hablantes distintos.
- Export multi-formato: `.vtt`, `.ass`, captions YouTube.
- Streaming / transcripción en realtime.

## Architecture seams

El motor está diseñado de forma que Phase 2 enchufe **sin reescribir
nada del core**:

1. **`ChunkRules`** es la única superficie que Phase 2 expande. Las
   reglas por cliente son ChunkRules con campos sobreescritos.
2. **El JSON intermedio con word timestamps** es source of truth — Phase
   2 puede aplicar reemplazos textuales sobre él sin tocar el chunker.
3. **CLI y futura API consumen el mismo motor**: `cli.py` solo orquesta.
   El dashboard de Phase 2 importa las mismas funciones.

## Cómo correrlo hoy

```bash
./scripts/setup.sh           # Mac (brew) o Linux (source build + CUDA detect)
uv sync
uv run whisper-subtitles transcribe video.mp4 --preset social
```

Las iteraciones sobre videos de prueba quedan guardadas en
[`generated-srt/`](generated-srt/) para comparar evoluciones del
chunker.
