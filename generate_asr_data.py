"""ASR Synthetic Data Generation Pipeline (Sinhala).

Generates Sinhala-language synthetic speech audio using the Gemini TTS Batch
API and tracks costs via LangSmith.
"""

import argparse
import csv
import hashlib
import os
import random
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import Client as LangSmithClient
from tqdm import tqdm

from speaking_styles import generate_unique_styles

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "asr-data-gen")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# ---------------------------------------------------------------------------
# Available voices
# ---------------------------------------------------------------------------
VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------
ls_client: LangSmithClient | None = None
SESSION_RUN_ID = str(uuid.uuid4())


def _init_langsmith() -> None:
    global ls_client
    ls_client = LangSmithClient()


def create_pipeline_run(model: str, input_csv: str, total_rows: int) -> str:
    """Create a top-level LangSmith run for the full pipeline execution."""
    ls_client.create_run(
        name="asr-data-gen-pipeline",
        run_type="chain",
        inputs={
            "model": model,
            "input_csv": input_csv,
            "total_rows": total_rows,
        },
        project_name=LANGSMITH_PROJECT,
        id=SESSION_RUN_ID,
    )
    return SESSION_RUN_ID


def log_batch_to_langsmith(
    parent_run_id: str,
    batch_index: int,
    model: str,
    num_requests: int,
    input_chars: int,
    output_audio_tokens: int,
    input_tokens: int,
) -> float:
    """Log a child run for a completed sub-batch with cost data."""
    # Pricing constants (Batch API, paid tier)
    if "flash" in model.lower():
        INPUT_PRICE_PER_1M = 0.25
        OUTPUT_PRICE_PER_1M = 5.00
    else:
        INPUT_PRICE_PER_1M = 0.50   # gemini-2.5-pro-preview-tts
        OUTPUT_PRICE_PER_1M = 10.00

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (output_audio_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost

    ls_client.create_run(
        name=f"tts-batch-{batch_index}",
        run_type="llm",
        parent_run_id=parent_run_id,
        project_name=LANGSMITH_PROJECT,
        inputs={"num_requests": num_requests, "input_chars": input_chars},
        outputs={
            "input_tokens": input_tokens,
            "output_audio_tokens": output_audio_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        },
        extra={"model": model},
        end_time=datetime.now(timezone.utc),
    )
    return total_cost


def close_pipeline_run(
    parent_run_id: str,
    total_cost: float,
    files_saved: int,
    files_skipped: int,
) -> None:
    """Close the top-level LangSmith run with cumulative totals."""
    ls_client.update_run(
        run_id=parent_run_id,
        outputs={
            "total_cost_usd": round(total_cost, 6),
            "files_saved": files_saved,
            "files_skipped": files_skipped,
        },
        end_time=datetime.now(timezone.utc),
    )

# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def save_wave(
    filepath: Path,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def get_duration(filepath: Path) -> float:
    with wave.open(str(filepath), "r") as wf:
        return wf.getnframes() / float(wf.getframerate())

# ---------------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------------


def make_filename(index: int, voice: str, style: str) -> str:
    style_hash = hashlib.md5(style.encode()).hexdigest()[:6]
    return f"{index:05d}_{voice}_{style_hash}.wav"

# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------


def _build_request(model: str, text: str, style: str, voice: str) -> types.InlinedRequest:
    """Build a single InlinedRequest for the Batch API."""
    return types.InlinedRequest(
        model=f"models/{model}",
        contents=[{"parts": [{"text": f"{style}\n\n{text}"}]}],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        ),
    )


def wait_for_batch(client, batch_name: str, poll_interval: int = 15):
    """Poll until the batch job completes or fails."""
    while True:
        job = client.batches.get(name=batch_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state == "JOB_STATE_SUCCEEDED":
            return job
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(
                f"Batch job {batch_name} ended with state: {state}"
            )
        time.sleep(poll_interval)


def _process_batch_api(
    client,
    model: str,
    sub_batch: list[tuple[int, str, str, str, str]],
    data_dir: Path,
    batch_index: int,
) -> tuple[list[dict], list[int], int]:
    """Submit a sub-batch via the Batch API and return results.

    Returns (saved_rows, failed_indices, output_audio_tokens).
    """
    requests = []
    for _idx, _filename, text, style, voice in sub_batch:
        requests.append(_build_request(model, text, style, voice))

    batch_job = client.batches.create(
        model=f"models/{model}",
        src=requests,
        config={"display_name": f"asr-batch-{batch_index}"},
    )

    job = wait_for_batch(client, batch_job.name)

    saved_rows: list[dict] = []
    failed_indices: list[int] = []
    output_audio_tokens = 0

    for i, inlined_resp in enumerate(job.dest.inlined_responses):
        idx, filename, text, style, voice = sub_batch[i]
        try:
            if inlined_resp.error:
                raise RuntimeError(inlined_resp.error)
            response = inlined_resp.response
            if not response or not response.candidates:
                raise RuntimeError("No candidates in response")
            candidate = response.candidates[0]
            if candidate.content is None:
                finish = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
                raise RuntimeError(f"Empty content (finish_reason={finish})")
            audio_data = candidate.content.parts[0].inline_data.data
            pcm_bytes = audio_data
            filepath = data_dir / filename
            save_wave(filepath, pcm_bytes)
            duration = get_duration(filepath)

            # Rough audio token estimate: ~25 tokens/sec at 24 kHz
            output_audio_tokens += int(duration * 25)

            saved_rows.append({
                "audio_filename": filename,
                "text": text,
                "speaking_style": style,
                "voice": voice,
                "duration_seconds": round(duration, 3),
            })
        except Exception as exc:
            print(f"  [WARN] Row {idx} failed: {exc}")
            failed_indices.append(idx)

    return saved_rows, failed_indices, output_audio_tokens


def _process_sequential_fallback(
    client,
    model: str,
    sub_batch: list[tuple[int, str, str, str, str]],
    data_dir: Path,
) -> tuple[list[dict], list[int], int]:
    """Fallback: process requests one-by-one via generate_content."""
    saved_rows: list[dict] = []
    failed_indices: list[int] = []
    output_audio_tokens = 0

    for idx, filename, text, style, voice in sub_batch:
        try:
            response = client.models.generate_content(
                model=f"models/{model}",
                contents=[{"parts": [{"text": f"{style}\n\n{text}"}]}],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            if not response or not response.candidates:
                raise RuntimeError("No candidates in response")
            candidate = response.candidates[0]
            if candidate.content is None:
                finish = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
                raise RuntimeError(f"Empty content (finish_reason={finish})")
            audio_data = candidate.content.parts[0].inline_data.data
            pcm_bytes = audio_data
            filepath = data_dir / filename
            save_wave(filepath, pcm_bytes)
            duration = get_duration(filepath)
            output_audio_tokens += int(duration * 25)

            saved_rows.append({
                "audio_filename": filename,
                "text": text,
                "speaking_style": style,
                "voice": voice,
                "duration_seconds": round(duration, 3),
            })
        except Exception as exc:
            print(f"  [WARN] Row {idx} failed (sequential): {exc}")
            failed_indices.append(idx)

    return saved_rows, failed_indices, output_audio_tokens

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Sinhala ASR data via Gemini TTS."
    )
    parser.add_argument(
        "--input", default="input.csv",
        help="Path to input CSV (must have a 'text' column).",
    )
    parser.add_argument("--output", default="output.csv", help="Path to output CSV.")
    parser.add_argument(
        "--data-dir", default="data", help="Directory to save WAV files."
    )
    parser.add_argument(
        "--num-styles", type=int, default=150,
        help="Number of speaking styles to generate.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Max requests per Batch API job.",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-pro-preview-tts",
        help="Gemini TTS model name.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load input CSV ----
    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise SystemExit("Input CSV must contain a 'text' column.")

    texts: list[str] = df["text"].astype(str).tolist()
    total_rows = len(texts)
    print(f"Loaded {total_rows} rows from {args.input}")

    # ---- Generate speaking styles ----
    styles = generate_unique_styles(args.num_styles)
    print(f"Generated {len(styles)} unique speaking styles")

    # ---- Assign voice + style per row ----
    rows: list[tuple[int, str, str, str, str]] = []
    for i, text in enumerate(texts):
        style = random.choice(styles)
        voice = random.choice(VOICES)
        filename = make_filename(i, voice, style)
        rows.append((i, filename, text, style, voice))

    # ---- Idempotency: skip existing files ----
    pending: list[tuple[int, str, str, str, str]] = []
    skipped = 0
    for row in rows:
        filepath = data_dir / row[1]
        if filepath.exists():
            skipped += 1
        else:
            pending.append(row)

    print(f"Skipping {skipped} already-existing files, {len(pending)} to generate")

    if not pending:
        print("Nothing to do — all files already exist.")
        _write_output_csv(args.output, rows, data_dir)
        return

    # ---- Cost estimate ----
    total_chars = sum(len(r[2]) + len(r[3]) for r in pending)
    est_tokens = total_chars // 4
    if "flash" in args.model.lower():
        cost_per_1m = 0.25
    else:
        cost_per_1m = 0.50
    est_cost = (est_tokens / 1_000_000) * cost_per_1m
    print(f"Total characters: {total_chars}")
    print(f"Estimated input tokens: ~{est_tokens}")
    print(f"Batch input cost ({args.model}): ${cost_per_1m:.2f} / 1M tokens")
    print(f"Estimated input cost: ~${est_cost:.4f}")

    # ---- Initialise clients ----
    client = genai.Client(api_key=GEMINI_API_KEY)
    _init_langsmith()
    parent_run_id = create_pipeline_run(args.model, args.input, total_rows)

    # ---- Split into sub-batches ----
    sub_batches = [
        pending[i: i + args.batch_size]
        for i in range(0, len(pending), args.batch_size)
    ]

    all_saved: list[dict] = []
    all_failed: list[int] = []
    cumulative_cost = 0.0

    use_batch_api = True

    for bi, sub_batch in enumerate(tqdm(sub_batches, desc="Batches")):
        input_chars = sum(len(r[2]) + len(r[3]) for r in sub_batch)
        input_tokens = input_chars // 4

        if use_batch_api:
            try:
                saved, failed, audio_tokens = _process_batch_api(
                    client, args.model, sub_batch, data_dir, bi,
                )
            except Exception as exc:
                print(f"\n[WARN] Batch API failed: {exc}")
                print("Falling back to sequential generate_content calls.")
                use_batch_api = False
                saved, failed, audio_tokens = _process_sequential_fallback(
                    client, args.model, sub_batch, data_dir,
                )
        else:
            saved, failed, audio_tokens = _process_sequential_fallback(
                client, args.model, sub_batch, data_dir,
            )

        all_saved.extend(saved)
        all_failed.extend(failed)

        batch_cost = log_batch_to_langsmith(
            parent_run_id=parent_run_id,
            batch_index=bi,
            model=args.model,
            num_requests=len(sub_batch),
            input_chars=input_chars,
            output_audio_tokens=audio_tokens,
            input_tokens=input_tokens,
        )
        cumulative_cost += batch_cost

        print(
            f"  Batch {bi}: saved={len(saved)}, failed={len(failed)}, "
            f"batch_cost=${batch_cost:.4f}"
        )

    # ---- Retry failed rows sequentially ----
    if all_failed:
        print(f"\nRetrying {len(all_failed)} failed rows sequentially...")
        failed_rows = [r for r in rows if r[0] in set(all_failed)]
        retry_saved, all_failed, retry_tokens = _process_sequential_fallback(
            client, args.model, failed_rows, data_dir
        )
        all_saved.extend(retry_saved)
        retry_input_chars = sum(len(r[2]) + len(r[3]) for r in failed_rows)
        retry_cost = log_batch_to_langsmith(
            parent_run_id=parent_run_id,
            batch_index=len(sub_batches),
            model=args.model,
            num_requests=len(failed_rows),
            input_chars=retry_input_chars,
            output_audio_tokens=retry_tokens,
            input_tokens=retry_input_chars // 4,
        )
        cumulative_cost += retry_cost
        print(f"  Retry: saved={len(retry_saved)}, still_failed={len(all_failed)}")

    # ---- Report failures ----
    if all_failed:
        print(f"\n[WARN] {len(all_failed)} rows failed: {all_failed}")

    # ---- Close LangSmith run ----
    close_pipeline_run(parent_run_id, cumulative_cost, len(all_saved), skipped)
    print(
        f"\nLangSmith run: https://smith.langchain.com/projects/{LANGSMITH_PROJECT} "
        f"| Total cost: ~${cumulative_cost:.4f}"
    )

    # ---- Write output CSV ----
    _write_output_csv(args.output, rows, data_dir, all_saved)


def _write_output_csv(
    output_path: str,
    rows: list[tuple[int, str, str, str, str]],
    data_dir: Path,
    saved_rows: list[dict] | None = None,
) -> None:
    """Write the output CSV with audio metadata.

    If saved_rows is provided, use those directly and augment with any
    pre-existing files from a previous run. Otherwise scan data_dir for
    all existing files matching the row filenames.
    """
    records: list[dict] = []

    if saved_rows is not None:
        # Index saved rows by filename for quick lookup
        saved_map = {r["audio_filename"]: r for r in saved_rows}
    else:
        saved_map = {}

    for idx, filename, text, style, voice in rows:
        if filename in saved_map:
            records.append(saved_map[filename])
        else:
            filepath = data_dir / filename
            if filepath.exists():
                duration = get_duration(filepath)
                records.append({
                    "audio_filename": filename,
                    "text": text,
                    "speaking_style": style,
                    "voice": voice,
                    "duration_seconds": round(duration, 3),
                })

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)
    print(f"Output CSV written to {output_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
