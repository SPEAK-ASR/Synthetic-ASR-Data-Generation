# Synthetic-ASR-Data-Generation

Generates synthetic Sinhala-language speech audio for ASR training using the Gemini TTS Batch API. Each text sample is paired with a randomly assigned voice and speaking style, producing a WAV file and a metadata CSV. API costs are tracked via LangSmith.

## How to use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=asr-data-gen   # optional, defaults to "asr-data-gen"
```

### 3. Prepare the input CSV

The input file must contain a `text` column with the Sinhala sentences to synthesise:

```csv
text
ඔබට ස්තූතියි.
මම ශ්‍රී ලංකාවෙන්.
```

### 4. Run the pipeline

```bash
python generate_asr_data.py \
  --input input.csv \
  --output output.csv \
  --data-dir data \
  --model gemini-2.5-pro-preview-tts \
  --num-styles 150 \
  --batch-size 100
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `input.csv` | Path to the input CSV (must have a `text` column) |
| `--output` | `output.csv` | Path for the output metadata CSV |
| `--data-dir` | `data` | Directory where WAV files are saved |
| `--model` | `gemini-2.5-pro-preview-tts` | Gemini TTS model (`gemini-2.5-flash-preview-tts` is cheaper) |
| `--num-styles` | `150` | Number of unique speaking styles to generate |
| `--batch-size` | `100` | Max requests per Batch API job |

### 5. Output

- **WAV files** are written to `--data-dir` (e.g. `data/00001_Zephyr_a3f9b2.wav`).
- **`output.csv`** contains one row per sample with columns: `audio_filename`, `text`, `speaking_style`, `voice`, `duration_seconds`.
- Re-running the script skips files that already exist, making the pipeline resumable.
