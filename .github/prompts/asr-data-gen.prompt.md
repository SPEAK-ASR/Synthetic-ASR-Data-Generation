---
agent: 'agent'
tools: [vscode, execute, read, agent, browser, edit, search, web, vscode.mermaid-chat-features/renderMermaidDiagram, ms-toolsai.jupyter/configureNotebook, ms-toolsai.jupyter/listNotebookPackages, ms-toolsai.jupyter/installNotebookPackages, todo]
description: 'Generate a complete Sinhala ASR synthetic data pipeline using Gemini 2.5 Pro Preview TTS with Batch API'
---

# ASR Synthetic Data Generation Pipeline (Sinhala)

Generate a complete Python pipeline for synthetic **Sinhala-language** ASR (Automatic Speech Recognition) data generation using the **Gemini 2.5 Pro Preview TTS** (`gemini-2.5-pro-preview-tts`) API with Batch API for 50% cost reduction.

The input CSV contains **Sinhala text** (BCP-47 language code: `si`). The Gemini TTS model detects the language automatically from the input text — no explicit language parameter is needed. All generated audio files will be Sinhala speech.

## Files to Create

```
generate_asr_data.py    # Main entry-point script
speaking_styles.py      # 100+ non-contradictory speaking style generator
requirements.txt        # Python dependencies
.env.example            # API key template
```

---

## 1. `speaking_styles.py` — Speaking Style Generator

Build a module that generates 100+ unique, non-contradictory speaking styles by combining attributes from several buckets.

### Attribute Buckets

Define the following Python lists:

```python
# Each tuple is a compatible pair of tone/emotion adjectives
TONE_PAIRS = [
    ("warm", "welcoming"),
    ("cold", "distant"),
    ("cheerful", "bright"),
    ("somber", "reflective"),
    ("excited", "enthusiastic"),
    ("calm", "serene"),
    ("serious", "authoritative"),
    ("playful", "lighthearted"),
    ("tense", "anxious"),
    ("confident", "assertive"),
    ("gentle", "tender"),
    ("dramatic", "intense"),
    ("dry", "detached"),
    ("melancholic", "wistful"),
    ("upbeat", "energetic"),
    ("hushed", "reverent"),
    ("joyful", "exuberant"),
    ("nostalgic", "sentimental"),
    ("urgent", "compelling"),
    ("bored", "monotone"),
]

PACE_WORDS = [
    "fast", "slow", "measured", "brisk", "leisurely",
    "deliberate", "flowing", "staccato", "unhurried", "clipped",
]

MANNER_WORDS = [
    "conversationally", "formally", "casually", "in a storytelling style",
    "instructionally", "meditatively", "journalistically", "theatrically",
    "intimately", "professionally", "whimsically", "matter-of-factly",
    "poetically", "in a documentary style", "in a broadcast style",
    "in a lecture style", "in a confessional style", "sarcastically",
    "enthusiastically", "gravely",
]

CONTEXT_SUFFIXES = [
    ", as if reading a bedtime story",
    ", as if presenting the evening news",
    ", as if narrating a nature documentary",
    ", as if hosting a podcast",
    ", as if dictating a formal letter",
    ", as if reading poetry aloud",
    ", as if calling an exciting sports play",
    ", as if speaking to a young child",
    ", as if delivering a TED talk",
    ", as if leaving a voicemail",
    ", as if reading an audiobook",
    ", as if announcing a product launch",
    ", as if recounting a ghost story",
    ", as if presenting a cooking show",
    "",  # no context suffix
    "",  # allow no-suffix more often
    "",
]
```

### Contradiction Rules

A combination is **invalid** if it contains any of these conflicts:

| Attribute | Conflicts with |
|-----------|---------------|
| `fast`, `brisk`, `clipped`, `staccato` | `slow`, `leisurely`, `unhurried`, `flowing`, `deliberate` |
| `slow`, `leisurely`, `unhurried`, `flowing` | `fast`, `brisk`, `clipped`, `staccato`, `urgent` |
| `calm`, `serene`, `gentle`, `hushed`, `meditative` | `tense`, `dramatic`, `excited`, `urgent`, `enthusiastic`, `upbeat` |
| `cheerful`, `playful`, `joyful`, `upbeat`, `warm` | `somber`, `melancholic`, `cold`, `bored`, `tense`, `dry` |
| `formal`, `professional`, `journalistic`, `lecture` | `casual`, `whimsical`, `conversational`, `sarcastic` |
| `bored`, `monotone`, `dry` | `excited`, `enthusiastic`, `dramatic`, `joyful`, `upbeat` |

### Style String Format

Construct styles using this template:

```
"Read aloud {manner}, {tone_adj1} and {tone_adj2}, at a {pace} pace{context_suffix}."
```

**Example outputs:**
- `"Read aloud conversationally, warm and welcoming, at a measured pace, as if hosting a podcast."`
- `"Read aloud theatrically, dramatic and intense, at a brisk pace, as if recounting a ghost story."`
- `"Read aloud formally, serious and authoritative, at a deliberate pace."`

### Function to Implement

```python
def generate_unique_styles(n: int = 150) -> list[str]:
    """
    Generate n unique non-contradictory speaking styles.
    Uses rejection sampling: draw random combinations, check all
    contradiction rules, add to a set until n unique styles collected.
    Has a safety cap to avoid infinite loops (max 50_000 attempts).
    """
```

Also expose:
```python
ALL_STYLES: list[str]  # module-level constant: generate_unique_styles(150)
```

---

## 2. `generate_asr_data.py` — Main Pipeline Script

### Imports and Setup

```python
import argparse, csv, hashlib, os, time, wave
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types
from speaking_styles import generate_unique_styles

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]  # hard-fail if missing
```

### Available Voices

```python
VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]
```

### CLI Arguments

```
--input       Path to input CSV (default: input.csv)
--output      Path to output CSV (default: output.csv)
--data-dir    Directory to save audio WAV files (default: data/)
--num-styles  Number of speaking styles to generate (default: 150)
--batch-size  Max requests per Batch API job (default: 100)
--model       Gemini TTS model to use (default: gemini-2.5-pro-preview-tts)
```

> **Default model is `gemini-2.5-pro-preview-tts`** (Pro). Pass `--model gemini-2.5-flash-preview-tts` to use the cheaper Flash model.

### Input CSV

Load a CSV that must have at least a `text` column containing **Sinhala text**. Each row becomes one TTS request.

### Per-Row Assignment

For each row, randomly assign:
- `speaking_style`: sampled (with replacement) from `generate_unique_styles(num_styles)`
- `voice`: randomly sampled from `VOICES`

### Audio Filename

```python
def make_filename(index: int, voice: str, style: str) -> str:
    style_hash = hashlib.md5(style.encode()).hexdigest()[:6]
    return f"{index:05d}_{voice}_{style_hash}.wav"
```

### Idempotency

Before processing each batch, skip rows whose output WAV file already exists in `data-dir`. Only submit new / missing files to the API.

### WAV Utilities

```python
def save_wave(filepath: Path, pcm: bytes,
              channels: int = 1, rate: int = 24000, sample_width: int = 2):
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def get_duration(filepath: Path) -> float:
    with wave.open(str(filepath), "r") as wf:
        return wf.getnframes() / float(wf.getframerate())
```

### Batch API Usage

Use the **Gemini Batch API** (`client.batches`) to submit all requests in sub-batches of `--batch-size` for the 50 % cost reduction.

Each individual request within the batch:
```python
# `model` and `sinhala_prompt_prefix` come from CLI args / constants
# The speaking style prompt is in English so the model understands the direction;
# the Sinhala text follows on a new line — the model auto-detects the language.
{
    "model": f"models/{model}",   # e.g. "models/gemini-2.5-pro-preview-tts"
    "contents": [{"parts": [{"text": f"{speaking_style}\n\n{text}"}]}],
    "generation_config": {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": voice}
            }
        }
    }
}
```

Batch creation:
```python
batch_job = client.batches.create(
    model=f"models/{model}",   # uses --model arg, default gemini-2.5-pro-preview-tts
    src=requests,              # list of GenerateContentRequest dicts
    config={"display_name": f"asr-batch-{sub_batch_index}"},
)
```

Polling loop:
```python
def wait_for_batch(client, batch_name: str, poll_interval: int = 15):
    while True:
        job = client.batches.get(name=batch_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state == "JOB_STATE_SUCCEEDED":
            return job
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(f"Batch job {batch_name} ended with state: {state}")
        time.sleep(poll_interval)
```

After the job succeeds, iterate `job.responses` (or the SDK equivalent results iterator) to extract inline audio data:
```python
for response in job.responses:
    audio_b64 = response.candidates[0].content.parts[0].inline_data.data
    pcm_bytes = base64.b64decode(audio_b64)
```

Match each response back to its originating row by maintaining an ordered list of `(index, filename, text, style, voice)` tuples that mirrors the request list.

### Failed Items

Collect any rows that produced no audio (API errors, empty responses). After all batches complete, print a warning listing failed row indices. Do **not** abort the entire job for individual failures.

### Cost Estimate

Before submitting, calculate and print an estimate:
```
Total characters: {total_chars}
Estimated input tokens: ~{total_chars // 4}
Batch input cost (Gemini 2.5 Pro Preview TTS): $0.50 / 1M tokens
Estimated cost: ~${estimated_cost:.4f}
```

### Output CSV

After all audio is saved, write `--output` CSV with these columns:

```
audio_filename, text, speaking_style, voice, duration_seconds
```

Use `pandas.DataFrame.to_csv(..., index=False)`.

### Progress Display

Wrap the sub-batch loop in `tqdm` showing batch progress. After each batch completes, show how many files were saved vs. skipped.

---

## 3. `requirements.txt`

```
google-genai>=0.8.0
python-dotenv>=1.0.0
tqdm>=4.66.0
pandas>=2.0.0
```

---

## 4. `.env.example`

```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Implementation Notes

- Target Python 3.10+
- Use `random.choice` / `random.sample` from the standard library (no extra deps)
- Use `base64` from the standard library to decode audio bytes
- Keep all batch-related logic in `generate_asr_data.py`; keep style generation isolated in `speaking_styles.py`
- If the Batch API is unavailable or rate-limited, fall back gracefully to sequential `client.models.generate_content()` calls with a printed warning
- Add a `if __name__ == "__main__":` guard and call `main()` cleanly
