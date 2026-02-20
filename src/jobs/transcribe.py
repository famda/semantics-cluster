"""
Distributed audio transcription using faster-whisper on a Ray cluster.

Usage (submitted as a Ray job):
    python transcribe_job.py --job-id <uuid> --blob-name <filename>

Flow:
  1. Download original audio from Azurite to a temp file on the head node
  2. Single ffmpeg pass decodes to 16 kHz mono s16le PCM; in-memory slicing
     with zlib-compressed chunks (peak RAM ≈ full decoded audio ~340 MB)
  3. Each chunk is ray.put() into the Object Store immediately
  4. Per-job GPU actors receive numpy arrays and call
     model.transcribe(ndarray) directly — no temp files on workers
  5. Head node merges segments, deduplicates at boundaries
  6. Upload transcription.json to Azurite

Actors are created fresh per job (not detached) so the Ray Dashboard
progress bar correctly tracks tasks for each job independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import List

# Silence onnxruntime C++ GPU-discovery warning (must be set before ort loads)
os.environ["ORT_LOG_LEVEL"] = "3"
# Suppress HF "unauthenticated requests" nag
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Filter noisy Python-level warnings from Ray and HuggingFace
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="ray")

import zlib

import numpy as np
import ray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000         # Whisper expects 16 kHz mono float32
CHUNK_LENGTH_SEC = 150      # 150 seconds core region (smaller = more parallelism)
OVERLAP_SEC = 5.0           # 5 seconds overlap on each side (matches baseline)
EPSILON = 0.3                # tolerance for boundary comparisons (seconds)
PREFETCH_DEPTH = 3           # pipeline depth to keep GPU busy
CONTAINER_NAME = "files"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "distil-large-v3.5")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("transcribe_job")

# Silence noisy third-party loggers
for _name in (
    "azure", "azure.storage", "azure.core", "urllib3",
    "faster_whisper", "ctranslate2", "huggingface_hub",
    "fsspec", "numba", "onnxruntime",
):
    logging.getLogger(_name).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkInfo:
    """Metadata for a single audio chunk."""
    index: int
    chunk_start: float   # actual start of the audio chunk (with overlap)
    chunk_end: float     # actual end of the audio chunk (with overlap)
    core_start: float    # start of non-overlapping core region
    core_end: float      # end of non-overlapping core region


# ---------------------------------------------------------------------------
# Blob helpers  (head-node only — workers never touch Azurite)
# ---------------------------------------------------------------------------

def _blob_service():
    from azure.storage.blob import BlobServiceClient
    conn_str = os.environ["AZURITE_CONN_STR"]
    return BlobServiceClient.from_connection_string(conn_str)


def download_blob_to_file(job_id: str, blob_name: str, dest_path: str) -> None:
    """Stream-download a blob directly to *dest_path*."""
    client = _blob_service().get_blob_client(CONTAINER_NAME, f"{job_id}/{blob_name}")
    log.info("Downloading %s/%s from blob storage …", job_id, blob_name)
    with open(dest_path, "wb") as f:
        stream = client.download_blob()
        stream.readinto(f)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    log.info("Downloaded %.2f MB → %s", size_mb, dest_path)


def upload_blob(job_id: str, blob_name: str, data: bytes) -> None:
    client = _blob_service().get_blob_client(CONTAINER_NAME, f"{job_id}/{blob_name}")
    client.upload_blob(data, overwrite=True)
    log.info("Uploaded %s/%s (%.2f KB)", job_id, blob_name, len(data) / 1024)


# ---------------------------------------------------------------------------
# Audio loading & chunking  (ffmpeg decode + soundfile — runs on head node)
# ---------------------------------------------------------------------------

def decode_and_chunk_audio(
    input_path: str,
    work_dir: str,
    chunk_length: int = CHUNK_LENGTH_SEC,
    overlap: float = OVERLAP_SEC,
) -> tuple[float, list[tuple[ChunkInfo, ray.ObjectRef]]]:
    """
    Decode audio with ffmpeg piped to stdout as raw s16le PCM, slice
    the full int16 array in memory, and zlib-compress each chunk into
    the Ray Object Store.

    Uses multi-threaded compression (zlib releases the GIL) so that
    72 chunks compress in ~2 s instead of ~8 s.

    Peak RAM ≈ full decoded audio as int16 (~340 MB for 3 h) plus one
    compressed copy per active thread.
    """
    from concurrent.futures import ThreadPoolExecutor

    sr = SAMPLE_RATE

    # Pipe ffmpeg → raw s16le PCM directly to memory (no WAV on disk)
    log.info("Decoding %s → 16 kHz mono s16le …", os.path.basename(input_path))
    t0 = time.time()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", input_path,
         "-ar", str(sr), "-ac", "1",
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, timeout=600, check=True,
    )
    full_audio = np.frombuffer(proc.stdout, dtype=np.int16)
    log.info("Decoded in %.1fs", time.time() - t0)

    # Remove original to free disk space
    try:
        os.unlink(input_path)
    except OSError:
        pass

    total_samples = len(full_audio)
    total_duration = total_samples / sr
    log.info(
        "Audio: %.1fs (%d samples at %d Hz)",
        total_duration, total_samples, sr,
    )

    if total_duration <= 0:
        raise ValueError("Audio file has zero duration")

    samples_per_chunk = int(chunk_length * sr)
    overlap_samples = int(overlap * sr)

    # Build chunk slices
    chunk_slices: list[tuple[ChunkInfo, np.ndarray]] = []
    core_start_sample = 0
    index = 0

    while core_start_sample < total_samples:
        core_end_sample = min(total_samples, core_start_sample + samples_per_chunk)

        chunk_start_sample = (
            max(0, core_start_sample - overlap_samples) if index > 0 else 0
        )
        chunk_end_sample = min(
            total_samples,
            core_end_sample + overlap_samples
            if core_end_sample < total_samples
            else total_samples,
        )

        chunk_int16 = full_audio[chunk_start_sample:chunk_end_sample].copy()

        info = ChunkInfo(
            index=index,
            chunk_start=chunk_start_sample / sr,
            chunk_end=chunk_end_sample / sr,
            core_start=core_start_sample / sr,
            core_end=core_end_sample / sr,
        )
        chunk_slices.append((info, chunk_int16))

        core_start_sample = core_end_sample
        index += 1

    # Free the large buffer now that slices are copied
    del full_audio

    # Compress + ray.put() in parallel (zlib releases the GIL)
    def _compress_and_put(item):
        info, chunk_int16 = item
        compressed = zlib.compress(chunk_int16.tobytes(), level=1)
        ref = ray.put(compressed)
        return info, ref, len(chunk_int16)

    chunk_refs: list[tuple[ChunkInfo, ray.ObjectRef]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for info, ref, n_samples in pool.map(_compress_and_put, chunk_slices):
            chunk_refs.append((info, ref))
            log.info(
                "Chunk %d: [%.1f–%.1fs] core=[%.1f–%.1fs] "
                "(%d samples) → Object Store",
                info.index, info.chunk_start, info.chunk_end,
                info.core_start, info.core_end, n_samples,
            )

    del chunk_slices

    log.info("Split complete — %d chunks in Object Store", len(chunk_refs))
    return total_duration, chunk_refs


# ---------------------------------------------------------------------------
# GPU transcription actor  (per-job, non-detached)
# ---------------------------------------------------------------------------

@ray.remote
class TranscriptionWorker:
    """
    GPU worker — loads the Whisper model once in ``__init__``.

    Created fresh per job so the Ray Dashboard correctly attributes tasks.
    The model is pre-cached in the Docker image so load time is ~5-10 s.
    """

    def __init__(self, model_name: str):
        os.environ["ORT_LOG_LEVEL"] = "3"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="ray")

        log.info("Loading model %s …", model_name)

        # Suppress onnxruntime C++ device_discovery.cc warning on stderr
        _real_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_name,
                device="cuda",
                compute_type="float16",
            )
            log.info("Model %s loaded successfully", model_name)

            # Warm up CUDA context: first inference triggers JIT kernel
            # compilation and may emit ONNX runtime warnings on stderr.
            _dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
            self.model.transcribe(_dummy, beam_size=1)
            log.info("CUDA warmup complete")
        finally:
            sys.stderr.close()
            sys.stderr = _real_stderr

    def ready(self) -> bool:
        """Health-check — returns True once the model is loaded."""
        return True

    def transcribe(
        self,
        chunk_data,
        chunk_index: int,
        chunk_start: float,
        core_start: float,
        core_end: float,
        epsilon: float = EPSILON,
    ) -> dict:
        """
        Transcribe *chunk_data* (zlib-compressed int16 bytes → float32).

        Segments are trimmed to the core region before returning.
        """
        from dataclasses import asdict

        # Decompress + convert int16 → float32
        chunk_audio = np.frombuffer(
            zlib.decompress(chunk_data), dtype=np.int16,
        ).astype(np.float32) / 32768.0

        log.info(
            "Chunk %d: transcribing (chunk_start=%.1fs, core=[%.1f–%.1f]s) …",
            chunk_index, chunk_start, core_start, core_end,
        )
        t0 = time.time()

        try:
            seg_iter, info = self.model.transcribe(
                chunk_audio,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            log.error(
                "Chunk %d: transcription failed — %s: %s",
                chunk_index, type(exc).__name__, exc,
            )
            return {
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "core_start": core_start,
                "core_end": core_end,
                "language": "unknown",
                "language_probability": 0.0,
                "segments": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        segments: List[dict] = []
        for seg in seg_iter:
            seg_dict = asdict(seg)
            seg_dict.pop("tokens", None)
            seg_dict["text"] = (seg_dict.get("text") or "").strip()

            # Map local chunk time → original timeline
            seg_dict["start"] = (seg_dict.get("start") or 0.0) + chunk_start
            seg_dict["end"] = (seg_dict.get("end") or 0.0) + chunk_start

            # Clean and offset words
            words = seg_dict.get("words") or []
            cleaned_words = []
            for w in words:
                if not w:
                    continue
                cleaned_words.append({
                    "word": (w.get("word") or "").strip(),
                    "start": (w.get("start") or 0.0) + chunk_start,
                    "end": (w.get("end") or 0.0) + chunk_start,
                })
            seg_dict["words"] = cleaned_words

            # ---- Trim segment to core region ----

            # Leading overlap: skip segments entirely before core_start
            if core_start > 0:
                if seg_dict["end"] <= core_start + epsilon:
                    continue

                if seg_dict["start"] < core_start - epsilon:
                    trimmed = [w for w in cleaned_words if w["end"] > core_start]
                    if trimmed:
                        for w in trimmed:
                            w["start"] = max(w["start"], core_start)
                        seg_dict["words"] = trimmed
                        seg_dict["start"] = trimmed[0]["start"]
                        seg_dict["end"] = max(w["end"] for w in trimmed)
                        seg_dict["text"] = " ".join(
                            w["word"] for w in trimmed if w["word"]
                        ).strip()
                    else:
                        seg_dict["start"] = max(seg_dict["start"], core_start)
                        if seg_dict["end"] - seg_dict["start"] <= epsilon:
                            continue

            # No trailing trim — baseline only trims leading overlap.
            # The next chunk's leading trim + merge dedup handles boundary overlap.

            if not seg_dict.get("text"):
                continue

            segments.append(seg_dict)

        elapsed = time.time() - t0
        log.info(
            "Chunk %d: done — %d segments in %.1fs (lang=%s, prob=%.2f)",
            chunk_index, len(segments), elapsed,
            info.language, info.language_probability,
        )

        return {
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "core_start": core_start,
            "core_end": core_end,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
        }


# ---------------------------------------------------------------------------
# Merge & deduplication  (head-node only)
# ---------------------------------------------------------------------------

def merge_results(
    chunk_results: list[dict],
    total_duration_sec: float,
    epsilon: float = EPSILON,
) -> dict:
    """
    Merge per-chunk segment results.

    Segments are already trimmed to their core regions by the workers.
    This function concatenates, deduplicates at boundaries, and assigns IDs.
    """
    chunk_results.sort(key=lambda r: r["core_start"])

    all_segments: list[dict] = []
    for result in chunk_results:
        for seg_dict in result["segments"]:
            if all_segments:
                prev = all_segments[-1]
                if (
                    prev.get("text") == seg_dict["text"]
                    and (
                        seg_dict["start"] <= prev["end"] + epsilon
                        or abs(prev["start"] - seg_dict["start"]) <= epsilon
                    )
                ):
                    continue
            all_segments.append(seg_dict)

    for i, seg in enumerate(all_segments):
        seg["id"] = i

    lang_counter: Counter = Counter()
    for r in chunk_results:
        lang_counter[r["language"]] += 1

    full_text = " ".join(s["text"] for s in all_segments if s.get("text"))
    all_words = []
    for seg in all_segments:
        all_words.extend(seg.get("words", []))

    overlap_used = OVERLAP_SEC if len(chunk_results) > 1 else 0.0

    return {
        "transcription": full_text,
        "segments": all_segments,
        "languages": sorted(lang_counter.keys()),
        "model": {
            "name": WHISPER_MODEL,
            "device": "cuda",
            "compute_type": "float16",
        },
        "chunking": {
            "length_seconds": CHUNK_LENGTH_SEC,
            "overlap_seconds": overlap_used,
            "chunk_count": len(chunk_results),
        },
        "duration_seconds": round(total_duration_sec, 3),
        "word_count": len(all_words),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--blob-name", required=True)
    args = parser.parse_args()

    ray.init(namespace="transcription")

    log.info("=== Transcription job %s starting ===", args.job_id)
    log.info("Model: %s", WHISPER_MODEL)

    # ------------------------------------------------------------------
    # 1. Azurite → Head Node: download audio to temp file
    # ------------------------------------------------------------------
    work_dir = tempfile.mkdtemp(prefix="transcribe_")
    ext = os.path.splitext(args.blob_name)[1] or ".wav"
    input_path = os.path.join(work_dir, f"input{ext}")
    download_blob_to_file(args.job_id, args.blob_name, input_path)

    # ------------------------------------------------------------------
    # 2. Per-job GPU actor creation  (non-blocking)
    #    One exclusive worker per GPU.  We kick off actor construction
    #    NOW so that model loading + CUDA warmup happen in the background
    #    while the head node decodes and chunks the audio (step 3).
    # ------------------------------------------------------------------
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    job_tag = args.job_id[:8]

    worker_plans: list[tuple[str, float]] = []
    for node in ray.nodes():
        if not node["Alive"]:
            continue
        res = node["Resources"]
        gpu_count = int(res.get("GPU", 0))
        if gpu_count < 1:
            continue
        node_id = node["NodeID"]
        for _ in range(gpu_count):
            worker_plans.append((node_id, 1.0))
        log.info(
            "Node %s: %d GPU(s) → %d worker(s) (exclusive, gpu_frac=1.0)",
            node_id[:8], gpu_count, gpu_count,
        )

    if not worker_plans:
        raise RuntimeError("No GPU nodes found in the cluster!")

    workers: list = []
    init_futures = []
    for wi, (node_id, gpu_frac) in enumerate(worker_plans):
        actor_name = f"w_{job_tag}_{wi}"
        strategy = NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False,
        )
        log.info(
            "Creating actor %s on node %s (num_gpus=%.4f) …",
            actor_name, node_id[:8], gpu_frac,
        )
        handle = TranscriptionWorker.options(
            name=actor_name,
            namespace="transcription",
            num_gpus=gpu_frac,
            scheduling_strategy=strategy,
        ).remote(WHISPER_MODEL)
        workers.append(handle)
        init_futures.append(handle.ready.remote())

    # Actors now loading model + CUDA warmup in the background …

    # ------------------------------------------------------------------
    # 3. Decode → chunk → ray.put()  (runs while actors warm up)
    # ------------------------------------------------------------------
    total_duration_sec, chunk_refs = decode_and_chunk_audio(input_path, work_dir)

    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    # Cap workers to actual chunk count (unlikely with MAX=1/GPU)
    if len(workers) > len(chunk_refs):
        # kill extras
        for h in workers[len(chunk_refs):]:
            ray.kill(h, no_restart=True)
        workers = workers[: len(chunk_refs)]
        init_futures = init_futures[: len(chunk_refs)]
        worker_plans = worker_plans[: len(chunk_refs)]

    # ------------------------------------------------------------------
    # 4. Wait for actors to be ready  (should be instant — they loaded
    #    during decode + chunk which takes ~19 s vs ~8 s for model load)
    # ------------------------------------------------------------------
    ray.get(init_futures, timeout=300)
    num_workers = len(workers)
    log.info(
        "%d TranscriptionWorker(s) ready across %d node(s)",
        num_workers, len({nid for nid, _ in worker_plans}),
    )

    # ------------------------------------------------------------------
    # 5. Ray Object Store → GPU Workers: work-stealing dispatch
    #    PREFETCH_DEPTH keeps each worker's queue full so the GPU never
    #    idles waiting for the next chunk to arrive.
    # ------------------------------------------------------------------
    log.info(
        "Dispatching %d chunks across %d worker(s) (prefetch depth %d) …",
        len(chunk_refs), num_workers, PREFETCH_DEPTH,
    )
    t0 = time.time()

    chunk_queue = list(chunk_refs)
    del chunk_refs
    active: dict = {}                   # future → worker_idx
    worker_inflight: dict = {wi: 0 for wi in range(num_workers)}
    chunk_results: list[dict] = []

    def _submit(wi: int) -> None:
        if not chunk_queue:
            return
        info, audio_ref = chunk_queue.pop(0)
        fut = workers[wi].transcribe.remote(
            audio_ref, info.index, info.chunk_start,
            info.core_start, info.core_end,
        )
        active[fut] = wi
        worker_inflight[wi] += 1

    # Seed each worker up to PREFETCH_DEPTH
    for wi in range(num_workers):
        for _ in range(PREFETCH_DEPTH):
            _submit(wi)

    # Drain loop
    while active:
        done, _ = ray.wait(list(active.keys()), num_returns=1)
        finished_fut = done[0]
        freed_worker = active.pop(finished_fut)
        worker_inflight[freed_worker] -= 1

        result = ray.get(finished_fut)
        chunk_results.append(result)
        if "error" in result:
            log.warning(
                "Chunk %d failed: %s",
                result["chunk_index"], result["error"],
            )

        while worker_inflight[freed_worker] < PREFETCH_DEPTH and chunk_queue:
            _submit(freed_worker)

    elapsed = time.time() - t0
    errors = [r for r in chunk_results if "error" in r]
    if errors:
        log.warning("%d of %d chunks had errors", len(errors), len(chunk_results))

    total_segs = sum(len(r["segments"]) for r in chunk_results)
    log.info("All chunks completed in %.1fs (%d segments total)", elapsed, total_segs)

    # ------------------------------------------------------------------
    # 6. Head Node: merge results
    # ------------------------------------------------------------------
    transcription = merge_results(chunk_results, total_duration_sec)
    transcription["job_id"] = args.job_id
    transcription["source_file"] = args.blob_name

    # ------------------------------------------------------------------
    # 7. Head Node → Azurite: upload result
    # ------------------------------------------------------------------
    result_json = json.dumps(transcription, ensure_ascii=False, indent=4)
    upload_blob(args.job_id, "transcription.json", result_json.encode("utf-8"))

    # ------------------------------------------------------------------
    # 8. Cleanup: kill per-job actors to free GPU memory immediately
    # ------------------------------------------------------------------
    for handle in workers:
        try:
            ray.kill(handle, no_restart=True)
        except Exception:
            pass

    log.info(
        "=== Transcription job %s completed — %d segments, %d words, %.1fs audio ===",
        args.job_id,
        len(transcription["segments"]),
        transcription["word_count"],
        transcription["duration_seconds"],
    )


if __name__ == "__main__":
    main()
