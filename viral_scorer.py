#!/usr/bin/env python3
"""
viral_scorer.py — Multi-signal Viral Score Engine for Shorts Factory v2.
Evaluates every possible clip window and returns the top N with the
highest predicted short-form engagement potential.

5 Signals:
  1. Hook Strength (30%) — pattern-matches opening hooks
  2. Emotional Energy (25%) — VADER sentiment + librosa audio energy
  3. Self-Containment (20%) — how standalone the segment is
  4. Trend Keyword Density (15%) — 2025-2026 high-performing keywords
  5. Pacing & Duration Fit (10%) — optimal length + word density
"""

import os
import json
import re
import logging
from typing import Optional

import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Keyword Loader ──────────────────────────────────────────────────

def _load_keywords() -> dict:
    """Load viral keywords from the editable JSON asset."""
    kw_path = os.path.join(BASE_DIR, "assets", "viral_keywords.json")
    try:
        with open(kw_path, 'r') as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values())
        logger.info(f"Loaded {total} keywords across {len(data)} categories from viral_keywords.json")
        return data
    except FileNotFoundError:
        logger.warning("viral_keywords.json not found — using empty keyword map")
        return {}


KEYWORDS = _load_keywords()


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 1: HOOK STRENGTH (weight: 30%)
# ═══════════════════════════════════════════════════════════════════════

_HOOK_PATTERNS = [
    # (label, score, pattern_regex)
    ("Curiosity Gap", 1.0, [
        r"nobody (talks about|tells you|will tell)",
        r"here'?s what they don'?t",
        r"the (truth|real reason|secret) (about|is)",
        r"what they don'?t want you",
        r"this changes everything",
        r"most people don'?t",
        r"what (nobody|no one) (realizes|knows|tells you)",
        r"the (hidden|untold|real) (truth|reason|story)",
        r"you (never|won'?t believe|probably don'?t)",
        r"there'?s a reason (why|that)",
    ]),
    ("Controversy / Bold Claim", 0.95, [
        r"everything you know (about )?(\w+) is wrong",
        r"stop doing (this|that|\w+)",
        r"I was wrong about",
        r"unpopular opinion",
        r"controversial (take|opinion)",
        r"don'?t hate me (but|for saying)",
        r"I (hate to break|need to tell) you",
        r"you'?ve been (lied to|doing it wrong|fooled)",
    ]),
    ("Shocking Statistic / Fact", 0.90, [
        r"^\d+(\.\d+)?%?\s+of\s+",
        r"in just \d+\s+(days?|weeks?|months?|years?)",
        r"(I|m[ey]|we|they) (made|lost|earned|gained|saved)\s+\$?\d+",
        r"\d+\s+million\b",
        r"\d+\s+(thousand|billion)\b",
        r"(only|just)\s+\d+%\s+",
    ]),
    ("Story Hook (personal)", 0.85, [
        r"I used to",
        r"I never thought",
        r"this (happened|occurred) to me",
        r"(last|next) (year|month|week) I",
        r"when I was",
        r"true story",
        r"let me tell you",
        r"I (remember|can still remember)",
    ]),
    ("Direct Question", 0.80, [
        r"^what if\b",
        r"^why (does|do|is|are|did|can|will|would)",
        r"^have you ever",
        r"^did you know",
        r"^can you",
        r"^how (would|could|can|do|does|did) (you|we|they|people)",
        r"^what (is|are|was|were|happens|happened)",
        r"^who (is|are|was|were)",
    ]),
    ("Problem/Pain Statement", 0.75, [
        r"if you'?re (struggling|tired|sick|stuck) (with|of)",
        r"tired of",
        r"stop wasting (time|money|energy)",
        r"the #?1 mistake",
        r"biggest mistake",
        r"(number|no\.?) ?1 (reason|cause|problem)",
        r"this is (why|exactly why) you'?re (failing|struggling|stuck)",
    ]),
    ("Authority Hook", 0.70, [
        r"after \d+\+?\s*(years?|months?) of",
        r"I'?ve (helped|coached|worked with|taught)\s+\d+",
        r"as a\s+\w+\s*,?\s*I",
        r"from my (experience|research|years)",
        r"in my \d+\+?\s*years?",
    ]),
]

# Weak/continuation words that indicate a bad hook
_WEAK_OPENERS = {
    "so", "but", "and", "also", "well", "anyway", "basically",
    "literally just", "as I said", "like I mentioned", "going back to",
}

_BAD_FIRST_WORDS = {"he", "she", "they", "it", "that", "this", "those", "these"}


def score_hook_strength(segment_text: str) -> dict:
    """
    Score the hook strength of a segment's opening.

    Returns: {"hook_score": float, "hook_type": str}
    """
    if not segment_text or len(segment_text.strip()) < 5:
        return {"hook_score": 0.30, "hook_type": "Generic / Weak"}

    words = segment_text.strip().split()
    first_15 = " ".join(words[:15]).lower()
    first_word = words[0].lower().strip("?.,!:")

    best_score = 0.30
    best_type = "Generic / Weak"

    for label, base_score, patterns in _HOOK_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, first_15, re.IGNORECASE):
                if base_score > best_score:
                    best_score = base_score
                    best_type = label
                break  # one match per pattern group is enough

    # Penalties for weak openers
    if first_word in _BAD_FIRST_WORDS:
        best_score = max(0.20, best_score - 0.15)
        best_type = "Generic / Weak (pronoun start)"

    if first_word in _WEAK_OPENERS:
        best_score = max(0.20, best_score - 0.10)
        if best_type == "Generic / Weak":
            best_type = f"Generic / Weak (\"{first_word}\" start)"

    return {"hook_score": round(best_score, 3), "hook_type": best_type}


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 2: EMOTIONAL ENERGY (weight: 25%)
# ═══════════════════════════════════════════════════════════════════════

_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def score_emotional_energy(segment_text: str, audio_path: str = None,
                           start_sec: float = 0, end_sec: float = 30) -> float:
    """
    Two-part emotion score: VADER text sentiment + librosa audio energy.

    Returns float 0.0–1.0
    """
    # Part A: Text sentiment intensity
    text_score = 0.0
    try:
        analyzer = _get_vader()
        scores = analyzer.polarity_scores(segment_text)
        text_score = abs(scores['compound'])  # High +/- = high arousal
    except Exception as e:
        logger.debug(f"VADER failed: {e}")
        text_score = 0.3

    # Part B: Audio energy (if audio available)
    audio_score = 0.5  # neutral default
    if audio_path and os.path.exists(audio_path):
        try:
            import librosa
            duration = end_sec - start_sec
            if duration <= 0 or duration > 600:
                audio_score = 0.5
            else:
                y, sr = librosa.load(audio_path, offset=start_sec,
                                     duration=min(duration, 120))
                # RMS energy
                rms = librosa.feature.rms(y=y)[0]
                mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0
                # Normalize RMS (typical speech: 0.01-0.15)
                rms_score = min(1.0, mean_rms / 0.10)

                # Speaking rate approximation via non-silent chunks
                non_silent = librosa.effects.split(y, top_db=20)
                if duration > 0:
                    chunks_per_sec = len(non_silent) / duration
                    rate_score = min(1.0, chunks_per_sec / 8.0)
                else:
                    rate_score = 0.5

                audio_score = 0.6 * rms_score + 0.4 * rate_score
        except Exception as e:
            logger.debug(f"librosa analysis failed: {e}")
            audio_score = 0.5

    final = 0.5 * text_score + 0.5 * audio_score
    return round(min(1.0, max(0.0, final)), 3)


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 3: SELF-CONTAINMENT (weight: 20%)
# ═══════════════════════════════════════════════════════════════════════

_CONTINUATION_WORDS = {
    "but", "and", "so", "also", "well", "anyway", "as i said",
    "like i mentioned", "going back to", "like we said",
}

_REFERENCE_PHRASES = [
    "as mentioned earlier", "as i said", "like i showed",
    "we talked about", "earlier i mentioned", "remember when",
    "as you know", "like we discussed",
]

_MULTI_PART_PHRASES = [
    "part 1", "part 2", "part 3", "in the next video",
    "last time we", "next video", "coming up",
    "in the previous", "we'll cover later",
]


def score_self_containment(segment_text: str) -> float:
    """
    Score how standalone / self-contained a segment is.

    Returns float 0.0–1.0
    """
    if not segment_text:
        return 0.0

    text_lower = segment_text.lower().strip()
    words = segment_text.strip().split()
    score = 1.0  # start perfect

    first_word = words[0].lower().strip("?.,!:") if words else ""

    # Penalty: pronoun as first word
    if first_word in _BAD_FIRST_WORDS:
        score -= 0.30

    # Penalty: continuation word as first word
    if first_word in _CONTINUATION_WORDS:
        score -= 0.20

    # Penalty: unresolved references
    for phrase in _REFERENCE_PHRASES:
        if phrase in text_lower:
            score -= 0.15
            break

    # Penalty: multi-part video references
    for phrase in _MULTI_PART_PHRASES:
        if phrase in text_lower:
            score -= 0.40
            break

    # Penalty: ends mid-thought
    if len(words) >= 5:
        last_5 = " ".join(words[-5:]).strip()
        if last_5 and last_5[-1] not in ".!?":
            score -= 0.25

    # Bonus: starts with proper noun (capitalized non-first word)
    if len(words) >= 2 and words[0][0].isupper() and words[0] not in ["I", "The", "This", "That", "It", "A", "An"]:
        score += 0.10

    # Bonus: ends with strong closing (punctuation + impact word)
    closing_words = {"subscribe", "follow", "comment", "share", "try", "start",
                     "stop", "remember", "think about", "watch", "check"}
    if len(words) >= 3:
        last_word = words[-1].lower().strip("?.,!:")
        if last_word in closing_words:
            score += 0.15
        elif segment_text.strip()[-1] in "?!":
            score += 0.10

    # Bonus: narrative arc indicators (setup + conflict/resolution words)
    has_setup = any(w in text_lower for w in ["but then", "however", "the problem", "the issue", "the challenge"])
    has_resolution = any(w in text_lower for w in ["so i", "that's how", "the solution", "here's what", "what i did"])
    if has_setup and has_resolution:
        score += 0.20

    # Bonus: single topic (low lexical diversity)
    if len(words) > 10:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.6:
            score += 0.10

    return round(min(1.0, max(0.0, score)), 3)


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 4: TREND KEYWORD DENSITY (weight: 15%)
# ═══════════════════════════════════════════════════════════════════════

def score_trend_keywords(segment_text: str) -> tuple:
    """
    Match segment against curated viral keyword list.

    Returns: (score: float, matched_keywords: list[str])
    """
    if not segment_text:
        return 0.0, []

    text_lower = segment_text.lower()
    text_words = set(text_lower.split())
    matched = []

    for category, keywords in KEYWORDS.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Match multi-word phrases
            if " " in kw_lower:
                if kw_lower in text_lower:
                    matched.append(kw)
            # Match single words
            elif kw_lower in text_words:
                matched.append(kw)

    total_words = len(segment_text.split())
    if total_words == 0:
        return 0.0, []

    # Density: unique matched keywords / total words, normalized
    density = len(set(matched)) / max(1, total_words)
    # Scale up: even 3-5 matches in a 50-word segment should score high
    score = min(1.0, density * 8.0)

    return round(score, 3), matched


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL 5: PACING & DURATION FIT (weight: 10%)
# ═══════════════════════════════════════════════════════════════════════

def score_pacing(start_sec: float, end_sec: float,
                 word_timestamps_in_segment: list) -> float:
    """
    Score the duration and word density of a segment.

    Returns float 0.0–1.0
    """
    duration = end_sec - start_sec

    # Duration score
    if duration < 15 or duration > 58:
        dur_score = 0.0
    elif 15 <= duration < 25:
        dur_score = 0.70
    elif 25 <= duration < 40:
        dur_score = 0.85
    elif 40 <= duration <= 55:
        dur_score = 1.0
    else:  # 55-58
        dur_score = 0.75

    # Word density (words per second)
    if duration > 0 and word_timestamps_in_segment:
        wps = len(word_timestamps_in_segment) / duration
    else:
        wps = 2.0

    if 2.0 <= wps <= 3.5:
        density_score = 1.0
    elif 3.5 < wps <= 4.5:
        density_score = 0.85
    elif 1.0 <= wps < 2.0:
        density_score = 0.70
    else:
        density_score = 0.40

    return round(0.5 * dur_score + 0.5 * density_score, 3)


# ═══════════════════════════════════════════════════════════════════════
# HELPER: Extract first sentence cleanly
# ═══════════════════════════════════════════════════════════════════════

def _extract_first_sentence(text: str) -> str:
    """Extract the first complete sentence from segment text."""
    if not text:
        return ""
    # Try to find sentence boundary
    for i, ch in enumerate(text):
        if ch in ".!?":
            sentence = text[:i + 1].strip()
            if len(sentence.split()) >= 3:
                return sentence
    # No clean sentence boundary — take first 12 words
    words = text.split()[:12]
    return " ".join(words) + ("..." if len(text.split()) > 12 else "")


# ═══════════════════════════════════════════════════════════════════════
# MASTER SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def score_all_segments(video_path: str, word_timestamps: list,
                       audio_path: str = None,
                       min_sec: float = 25, max_sec: float = 58,
                       step_sec: float = 5,
                       candidate_durations: list = None,
                       num_shorts: int = 5,
                       min_self_containment: float = 0.40) -> list:
    """
    Sliding window approach — evaluate every candidate segment.

    Returns list sorted by final_score descending, each dict containing
    all 5 signal scores plus metadata.
    """
    if candidate_durations is None:
        candidate_durations = [30, 40, 50, 55]

    if not word_timestamps:
        return []

    video_end = word_timestamps[-1]["end"]
    video_start = word_timestamps[0]["start"]
    video_duration = video_end - video_start

    if video_duration < min_sec:
        # Video too short for sliding window — return whole thing
        text = " ".join(w["word"] for w in word_timestamps)
        hook = score_hook_strength(text)
        energy = score_emotional_energy(text, audio_path, 0, video_duration)
        containment = score_self_containment(text)
        kw_score, matched = score_trend_keywords(text)
        pacing = score_pacing(video_start, video_end, word_timestamps)
        final = (0.30 * hook["hook_score"] + 0.25 * energy +
                 0.20 * containment + 0.15 * kw_score + 0.10 * pacing)
        return [{
            "start": round(video_start, 2), "end": round(video_end, 2),
            "duration": round(video_duration, 2), "final_score": round(final, 3),
            "hook_score": hook["hook_score"], "hook_type": hook["hook_type"],
            "emotional_energy": energy, "self_containment": containment,
            "trend_keywords": kw_score, "pacing": pacing,
            "matched_keywords": matched,
            "transcript_excerpt": text[:80] + ("..." if len(text) > 80 else ""),
            "first_sentence": _extract_first_sentence(text),
        }]

    # Build word index for fast lookup
    # Map time -> word index (binary search equivalent)
    word_starts = [w["start"] for w in word_timestamps]

    def _words_in_range(s, e):
        return [w for w in word_timestamps if s <= w["start"] <= e]

    def _text_in_range(s, e):
        return " ".join(w["word"] for w in _words_in_range(s, e))

    # Generate all candidate windows
    candidates = []
    pos = 0.0
    while pos < video_duration - min_sec:
        for dur in candidate_durations:
            if pos + dur > video_duration:
                continue
            s = video_start + pos
            e = s + dur
            candidates.append((s, e, dur))
        pos += step_sec

    print(f"  {Fore.WHITE}  Evaluating {len(candidates):,} candidate windows...{Style.RESET_ALL}")

    # Score each candidate
    scored = []
    for s, e, dur in candidates:
        seg_text = _text_in_range(s, e)
        if len(seg_text.split()) < 8:
            continue  # Skip nearly empty windows

        hook = score_hook_strength(seg_text)
        energy = score_emotional_energy(seg_text, audio_path, s, e)
        containment = score_self_containment(seg_text)

        # Hard cutoff: self-containment must pass minimum
        if containment < min_self_containment:
            continue

        kw_score, matched = score_trend_keywords(seg_text)
        pacing = score_pacing(s, e, _words_in_range(s, e))

        final = (0.30 * hook["hook_score"] + 0.25 * energy +
                 0.20 * containment + 0.15 * kw_score + 0.10 * pacing)

        scored.append({
            "start": round(s, 2), "end": round(e, 2),
            "duration": round(dur, 2), "final_score": round(final, 3),
            "hook_score": hook["hook_score"], "hook_type": hook["hook_type"],
            "emotional_energy": energy, "self_containment": containment,
            "trend_keywords": kw_score, "pacing": pacing,
            "matched_keywords": matched,
            "transcript_excerpt": seg_text[:60] + ("..." if len(seg_text) > 60 else ""),
            "first_sentence": _extract_first_sentence(seg_text),
        })

    # Sort by final score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Deduplication: remove candidates overlapping >50% with a higher-scored one
    selected = []
    for seg in scored:
        overlap = False
        for sel in selected:
            overlap_start = max(seg["start"], sel["start"])
            overlap_end = min(seg["end"], sel["end"])
            overlap_dur = max(0, overlap_end - overlap_start)
            shorter_dur = min(seg["duration"], sel["duration"])
            if shorter_dur > 0 and overlap_dur / shorter_dur > 0.50:
                overlap = True
                break
        if not overlap:
            selected.append(seg)

    # Sort selected by start time, take top N
    selected.sort(key=lambda x: x["start"])
    selected = selected[:num_shorts]

    print(f"  {Fore.GREEN}  {len(selected)} top segments selected from "
          f"{len(scored)} scored candidates.{Style.RESET_ALL}")

    return selected