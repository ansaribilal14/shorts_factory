#!/usr/bin/env python3
"""
transcriber.py — Local speech-to-text using OpenAI Whisper.
Produces SRT subtitles and word-level timestamp JSON files.
"""

import os
import json
import logging
from datetime import timedelta

import whisper
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)


def _format_timestamp_srt(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_video(video_path: str, model_size: str = "base",
                     language: str = "en") -> tuple:
    print(f"\n{Fore.CYAN}  Loading Whisper model '{model_size}'...{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}(First run downloads the model weights){Style.RESET_ALL}\n")
    device = "cuda" if whisper.available_devices() and "cuda" in whisper.available_devices() else "cpu"
    model = whisper.load_model(model_size, device=device)
    print(f"  Transcribing audio ({device})...")
    result = model.transcribe(video_path, language=language, verbose=False, word_timestamps=True)
    word_timestamps = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word_timestamps.append({"word": word_info["word"].strip(), "start": round(word_info["start"], 3), "end": round(word_info["end"], 3)})
    srt_lines = []
    for i, segment in enumerate(result.get("segments", []), 1):
        start_time = _format_timestamp_srt(segment["start"])
        end_time = _format_timestamp_srt(segment["end"])
        text = segment["text"].strip()
        srt_lines.append(str(i))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
    srt_text = "\n".join(srt_lines)
    print(f"  {Fore.GREEN}Transcription complete!{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}{len(word_timestamps)} words detected{Style.RESET_ALL}")
    return srt_text, word_timestamps


def save_transcription(srt_text: str, word_timestamps: list, video_id: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    srt_path = os.path.join(output_dir, f"{video_id}.srt")
    json_path = os.path.join(output_dir, f"{video_id}_words.json")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    print(f"  SRT saved: {Fore.WHITE}{srt_path}{Style.RESET_ALL}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(word_timestamps, f, indent=2, ensure_ascii=False)
    print(f"  Word JSON saved: {Fore.WHITE}{json_path}{Style.RESET_ALL}")


def find_best_short_segments(word_timestamps: list, metadata: dict,
                              num_shorts: int = 5, min_sec: float = 30, max_sec: float = 58) -> list:
    hook_words = {
        "how", "why", "what", "when", "where", "who",
        "secret", "mistake", "never", "always", "incredible",
        "amazing", "shocking", "truth", "nobody", "tells",
        "actually", "real", "reveal", "hidden", "discover",
        "imagine", "believe", "fact", "prove", "wrong",
        "dangerous", "powerful", "important", "critical",
        "surprising", "unbelievable", "insane", "crazy",
    }
    question_starters = {"how", "why", "what", "when", "where", "who", "is", "are", "do", "does", "can", "will"}
    if not word_timestamps:
        return []
    total_duration = word_timestamps[-1]["end"] - word_timestamps[0]["start"]
    if total_duration < min_sec:
        return [{"start": word_timestamps[0]["start"], "end": word_timestamps[-1]["end"],
                 "transcript_excerpt": " ".join(w["word"] for w in word_timestamps[:20]), "score": 1.0}]
    segments = []
    words = word_timestamps
    n = len(words)
    target_duration = max_sec - 2
    for i in range(n):
        window_start = words[i]["start"]
        window_end_words = []
        for j in range(i, n):
            if words[j]["end"] - window_start > target_duration:
                break
            window_end_words.append(j)
        if not window_end_words:
            continue
        j = window_end_words[-1]
        seg_start = words[i]["start"]
        seg_end = words[j]["end"]
        seg_duration = seg_end - seg_start
        if seg_duration < min_sec:
            continue
        excerpt_words = [words[k]["word"] for k in range(i, min(j + 1, i + 15))]
        excerpt = " ".join(excerpt_words)
        if j + 1 > i + 15:
            excerpt += "..."
        score = 0.0
        seg_text_lower = " ".join(words[k]["word"].lower() for k in range(i, j + 1))
        first_word = words[i]["word"].lower().strip("?.,!")
        if first_word in question_starters:
            score += 3.0
        if "?" in seg_text_lower:
            score += 2.0
        seg_words_set = set(seg_text_lower.split())
        hook_matches = seg_words_set & hook_words
        score += len(hook_matches) * 0.8
        multi_word_hooks = ["what if", "the reason", "the truth", "the secret", "most people", "didn't know", "you need", "stop doing", "nobody tells", "actually happens"]
        for hook in multi_word_hooks:
            if hook in seg_text_lower:
                score += 1.5
        if seg_start < 180:
            score += 1.0
        elif seg_start < 360:
            score += 0.5
        if seg_duration < 35:
            score -= 0.5
        elif seg_duration > 55:
            score -= 0.3
        last_word_clean = words[j]["word"].strip()
        if last_word_clean and last_word_clean[-1] in ".!?":
            score += 1.0
        if i > 0:
            prev_word = words[i - 1]["word"]
            if prev_word.strip() and prev_word.strip()[-1] not in ".!?,:;":
                score -= 0.3
        segments.append({"start": round(seg_start, 2), "end": round(seg_end, 2),
                         "duration": round(seg_duration, 2), "transcript_excerpt": excerpt,
                         "score": round(score, 2), "word_start_idx": i, "word_end_idx": j})
    segments.sort(key=lambda s: s["score"], reverse=True)
    selected = []
    for seg in segments:
        overlap = False
        for sel in selected:
            overlap_start = max(seg["start"], sel["start"])
            overlap_end = min(seg["end"], sel["end"])
            overlap_duration = max(0, overlap_end - overlap_start)
            seg_dur = min(seg["duration"], sel["duration"])
            if seg_dur > 0 and overlap_duration / seg_dur > 0.3:
                overlap = True
                break
        if not overlap:
            selected.append(seg)
        if len(selected) >= num_shorts:
            break
    if len(selected) < num_shorts and total_duration > min_sec:
        spacing = (total_duration - min_sec) / max(1, num_shorts - len(selected))
        idx = 0
        while len(selected) < num_shorts:
            candidate_start = idx * spacing
            candidate_end = min(candidate_start + target_duration, total_duration)
            if candidate_end - candidate_start < min_sec:
                break
            start_word = min(range(n), key=lambda k: abs(words[k]["start"] - candidate_start))
            end_word = min(range(n), key=lambda k: abs(words[k]["end"] - candidate_end))
            if end_word <= start_word:
                end_word = min(start_word + 30, n - 1)
            seg_start = words[start_word]["start"]
            seg_end = words[end_word]["end"]
            if seg_end - seg_start >= min_sec:
                excerpt = " ".join(words[k]["word"] for k in range(start_word, min(end_word + 1, start_word + 15)))
                selected.append({"start": round(seg_start, 2), "end": round(seg_end, 2),
                                 "duration": round(seg_end - seg_start, 2), "transcript_excerpt": excerpt + "...",
                                 "score": 0.0, "word_start_idx": start_word, "word_end_idx": end_word})
            idx += 1
    for seg in selected:
        seg.pop("word_start_idx", None)
        seg.pop("word_end_idx", None)
        seg.pop("duration", None)
    selected.sort(key=lambda s: s["start"])
    selected = selected[:num_shorts]
    return selected