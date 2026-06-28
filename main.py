#!/usr/bin/env python3
"""
main.py — Shorts Factory v2 Orchestrator.
Upgraded with: Speaker-Focused Smart Camera + Viral Score Engine.

Usage:
    python main.py              # Interactive mode
    python main.py --check      # Verify all dependencies
    python main.py --ci <URL>   # CI/headless mode
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init
init(autoreset=True)

# Load .env for secrets (never committed)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(_env_path)
    _DOTENV_LOADED = True
except ImportError:
    _DOTENV_LOADED = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
CAPTIONS_DIR = os.path.join(BASE_DIR, "captions")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"run_{ts}.log")
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    return logging.getLogger("shorts_factory")


logger = setup_logging()


def _progress_bar(pct, label, bar_width=30):
    filled = int(bar_width * pct / 100)
    bar = '\u2588' * filled + '\u2591' * (bar_width - filled)
    print(f"\n  {Fore.GREEN}[{bar}]{Style.RESET_ALL} {pct:5.1f}% \u2014 {Fore.WHITE}{label}{Style.RESET_ALL}")


def _fmt_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _bar(score, width=10):
    """Visual score bar for breakdown display."""
    filled = int(width * score)
    return '\u2588' * filled + '\u2591' * (width - filled)


# ═══════════════════════════════════════════════════════════════════════
# DEPENDENCY CHECK (v2)
# ═══════════════════════════════════════════════════════════════════════

def run_dependency_check():
    print(f"\n{Fore.CYAN}  Running dependency check (v2)...{Style.RESET_ALL}\n")
    checks = []
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    sym = f"{Fore.GREEN}\u2713{Style.RESET_ALL}" if py_ok else f"{Fore.RED}\u2717{Style.RESET_ALL}"
    print(f"  {sym} Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    checks.append(py_ok)

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} FFmpeg")
        checks.append(True)
    except FileNotFoundError:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} FFmpeg (NOT FOUND)")
        checks.append(False)

    modules = [
        ("yt_dlp", "yt-dlp"), ("whisper", "openai-whisper"), ("moviepy", "moviepy"),
        ("PIL", "Pillow"), ("ffmpeg", "ffmpeg-python"), ("telegram", "python-telegram-bot"),
        ("numpy", "numpy"), ("mediapipe", "mediapipe"),
        ("pyannote.audio", "pyannote.audio"), ("vaderSentiment", "vaderSentiment"),
        ("librosa", "librosa"),
    ]
    for mod, name in modules:
        try:
            __import__(mod)
            print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} {name}")
            checks.append(True)
        except ImportError:
            print(f"  {Fore.RED}\u2717{Style.RESET_ALL} {name} (missing)")
            checks.append(False)

    for fname in ["Montserrat-ExtraBold.ttf", "Montserrat-Bold.ttf"]:
        fpath = os.path.join(BASE_DIR, "assets", "fonts", fname)
        ok = os.path.exists(fpath)
        sym = f"{Fore.GREEN}\u2713{Style.RESET_ALL}" if ok else f"{Fore.RED}\u2717{Style.RESET_ALL}"
        print(f"  {sym} {fname}")
        checks.append(ok)

    # Viral keywords
    kw_path = os.path.join(BASE_DIR, "assets", "viral_keywords.json")
    if os.path.exists(kw_path):
        with open(kw_path) as f:
            kw_data = json.load(f)
        total_kw = sum(len(v) for v in kw_data.values())
        print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} assets/viral_keywords.json ({total_kw} keywords, {len(kw_data)} categories)")
        checks.append(True)
    else:
        print(f"  {Fore.RED}\u2717{Style.RESET_ALL} assets/viral_keywords.json (missing)")
        checks.append(False)

    config_ok = os.path.exists(CONFIG_PATH)
    sym = f"{Fore.GREEN}\u2713{Style.RESET_ALL}" if config_ok else f"{Fore.RED}\u2717{Style.RESET_ALL}"
    print(f"  {sym} config.json")
    checks.append(config_ok)

    # HF token warning
    if config_ok:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        hf = cfg.get('hf_token') or os.environ.get('HF_TOKEN', '')
        if not hf:
            print(f"\n  {Fore.YELLOW}  HuggingFace token not set \u2014 speaker camera will use center-crop fallback{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}    Set HF_TOKEN in .env or run python main.py to set it up.{Style.RESET_ALL}")
        else:
            print(f"  {Fore.GREEN}\u2713{Style.RESET_ALL} HuggingFace token (set via .env)")

    print()
    if all(checks):
        print(f"  {Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}  ALL CHECKS PASSED! Shorts Factory v2 ready.{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        return True
    else:
        failed = sum(1 for c in checks if not c)
        print(f"  {Fore.RED}{'=' * 50}{Style.RESET_ALL}")
        print(f"  {Fore.RED}  {failed} check(s) failed.{Style.RESET_ALL}")
        print(f"  {Fore.RED}{'=' * 50}{Style.RESET_ALL}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# CONFIG HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load_config():
    # Load base config from config.json
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    # Inject secrets from .env (never stored in config.json)
    _tg_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    _tg_chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    _hf_token = os.environ.get('HF_TOKEN', '')
    if _tg_token and _tg_token != 'YOUR_BOT_TOKEN_HERE':
        config['telegram_bot_token'] = _tg_token
    if _tg_chat and _tg_chat != 'YOUR_CHAT_ID_HERE':
        config['telegram_chat_id'] = _tg_chat
    if _hf_token:
        config['hf_token'] = _hf_token
    return config


def save_config(config):
    # Only save non-secret keys to config.json
    safe_config = {k: v for k, v in config.items()
                   if k not in ('telegram_bot_token', 'telegram_chat_id', 'hf_token')}
    with open(CONFIG_PATH, 'w') as f:
        json.dump(safe_config, f, indent=2, ensure_ascii=False)
        f.write('\n')

def is_valid_youtube_url(url):
    return any(url.startswith(p) for p in [
        "https://www.youtube.com/", "https://youtu.be/",
        "https://m.youtube.com/", "http://www.youtube.com/", "http://youtu.be/"])


# ═══════════════════════════════════════════════════════════════════════
# v2 VIRAL SCORE TABLE DISPLAY
# ═══════════════════════════════════════════════════════════════════════

def _draw_viral_table(segments, show_breakdown=True):
    if not segments:
        return
    # Header
    print(f"\n  {'+'.join('-' * 8 for _ in range(7))}")
    cols = [("Rank", 7), ("Start", 8), ("End", 8), ("Dur", 6),
            ("Score", 7), ("Hook Type", 22), ("Opening Hook", 42)]
    header_line = "  " + "|".join(f" {h:<{w-2}} " for h, w in cols) + "|"
    print(header_line)
    print(f"  {'+'.join('-' * 8 for _ in range(7))}")

    for i, seg in enumerate(segments):
        dur = int(seg["end"] - seg["start"])
        score = seg.get("final_score", 0)
        hook_type = seg.get("hook_type", "?")[:20]
        hook_text = seg.get("first_sentence", seg.get("transcript_excerpt", ""))[:40]
        row = [
            f"  #{i+1}",
            _fmt_time(seg["start"]),
            _fmt_time(seg["end"]),
            f"  {dur}s",
            f"  {score:.2f}",
            f"  {hook_type}",
            f"  {hook_text}",
        ]
        print("  " + "|".join(f"{r:<{w}}" for r, w in zip(row, [c[1] for c in cols])) + "|")

    print(f"  {'+'.join('-' * 8 for _ in range(7))}")

    # Score breakdown for top 3
    if show_breakdown:
        for i in range(min(3, len(segments))):
            seg = segments[i]
            score = seg.get("final_score", 0)
            print(f"\n  {Fore.CYAN}Score Breakdown \u2014 Short #{i+1} "
                  f"(score: {score:.2f}){Style.RESET_ALL}")
            print(f"     Hook Strength:    {_bar(seg.get('hook_score', 0))}  "
                  f"{seg.get('hook_score', 0):.2f}  "
                  f"[{seg.get('hook_type', '?')}]")
            print(f"     Emotional Energy: {_bar(seg.get('emotional_energy', 0))}  "
                  f"{seg.get('emotional_energy', 0):.2f}")
            print(f"     Self-Containment: {_bar(seg.get('self_containment', 0))}  "
                  f"{seg.get('self_containment', 0):.2f}")
            kw = seg.get('matched_keywords', [])
            kw_str = ", ".join(kw[:5]) + (f" +{len(kw)-5} more" if len(kw) > 5 else "")
            print(f"     Trend Keywords:   {_bar(seg.get('trend_keywords', 0))}  "
                  f"{seg.get('trend_keywords', 0):.2f}  "
                  f"[{kw_str}]")
            dur = seg["end"] - seg["start"]
            print(f"     Pacing:           {_bar(seg.get('pacing', 0))}  "
                  f"{seg.get('pacing', 0):.2f}  "
                  f"[{dur:.0f}s]")
    print()


# ═══════════════════════════════════════════════════════════════════════
# AUDIO FADE HELPER
# ═══════════════════════════════════════════════════════════════════════

def _apply_audio_fades(input_path, output_path):
    import ffmpeg as ff
    (ff.input(input_path).filter('afade', t='in', st=0, d=0.05)
     .filter('afade', t='out', st=-0.05, d=0.05)
     .output(output_path, c='copy', loglevel='error').run(overwrite_output=True))


# ═══════════════════════════════════════════════════════════════════════
# v2 PIPELINE (CI MODE)
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline_ci(youtube_url):
    from downloader import download_video, get_video_metadata
    from transcriber import transcribe_video, save_transcription
    from viral_scorer import score_all_segments
    from clip_extractor import extract_clip, reframe_to_vertical
    from caption_renderer import render_captions_on_clip
    from motion_graphics import compose_final_short
    from telegram_uploader import upload_shorts_to_telegram

    config = load_config()
    print(f"\n{Fore.CYAN}  CI MODE v2 \u2014 {youtube_url}{Style.RESET_ALL}")

    try:
        metadata = get_video_metadata(youtube_url)
    except Exception as e:
        logger.error(f"Metadata failed: {e}")
        return False

    print(f"  Found: {metadata['title']} ({metadata['duration']}s)")

    _progress_bar(10, "Downloading video...")
    try:
        video_info = download_video(youtube_url, DOWNLOADS_DIR)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

    video_path = video_info['path']
    video_id = video_info['video_id']

    _progress_bar(25, "Transcribing with Whisper...")
    try:
        srt_text, word_timestamps = transcribe_video(video_path,
            model_size=config.get('whisper_model', 'base'), language=config.get('language', 'en'))
        save_transcription(srt_text, word_timestamps, video_id, CAPTIONS_DIR)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return False

    if not word_timestamps:
        print(f"  {Fore.RED}No speech detected.{Style.RESET_ALL}")
        return False

    # Speaker camera pipeline (optional, graceful fallback)
    speaker_timeline = []
    if config.get('speaker_camera_enabled') and config.get('hf_token'):
        _progress_bar(40, "Running speaker diarization (pyannote)...")
        try:
            from speaker_camera import run_full_speaker_pipeline
            speaker_timeline = run_full_speaker_pipeline(
                video_path, config['hf_token'], video_id, config)
        except Exception as e:
            logger.warning(f"Speaker camera failed, using fallback: {e}")

        if speaker_timeline:
            _progress_bar(50, "Building face tracking map...")
        else:
            _progress_bar(50, "Speaker camera unavailable \u2014 using center-crop fallback")
    else:
        _progress_bar(50, "Speaker camera disabled \u2014 using center-crop fallback")

    _progress_bar(60, "Scoring all segments for virality...")
    try:
        segments = score_all_segments(
            video_path, word_timestamps, audio_path=video_path,
            min_sec=config.get('short_duration_min', 30) - 5,
            max_sec=config.get('short_duration_max', 58),
            step_sec=config.get('viral_window_step_sec', 5),
            candidate_durations=config.get('viral_candidate_durations', [30, 40, 50, 55]),
            num_shorts=config.get('shorts_count', 5),
            min_self_containment=config.get('viral_score_min_self_containment', 0.40),
        )
    except Exception as e:
        logger.error(f"Viral scoring failed: {e}")
        return False

    if not segments:
        print(f"  {Fore.RED}No suitable segments found.{Style.RESET_ALL}")
        return False

    _draw_viral_table(segments, config.get('show_score_breakdown', True))

    _progress_bar(75, "Extracting + smart reframing clips...")
    vertical_paths = []
    for i, seg in enumerate(segments):
        print(f"\n  {Fore.CYAN}-- Clip #{i+1}/{len(segments)} --{Style.RESET_ALL}")
        try:
            clip_path = extract_clip(video_path, seg['start'], seg['end'], i, CLIPS_DIR)

            # Smart reframe if speaker timeline available
            if speaker_timeline:
                from speaker_camera import reframe_with_speaker_tracking
                vp = reframe_with_speaker_tracking(
                    clip_path, speaker_timeline, seg['start'],
                    os.path.join(CLIPS_DIR, f"clip_{i:02d}_smart_reframe.mp4"),
                    smooth_factor=config.get('speaker_camera_smooth_factor', 0.85),
                    face_vertical_pos=config.get('speaker_face_vertical_position', 0.30),
                )
            else:
                vp = reframe_to_vertical(clip_path, CLIPS_DIR, config.get('background_blur', True))

            vertical_paths.append((i, seg, vp))
        except Exception as e:
            logger.error(f"Clip {i} failed: {e}")

    if not vertical_paths:
        print(f"  {Fore.RED}All clips failed.{Style.RESET_ALL}")
        return False

    _progress_bar(88, "Rendering captions + motion graphics...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_outputs = []
    for i, seg, vp in vertical_paths:
        print(f"\n  {Fore.CYAN}-- Rendering #{i+1} --{Style.RESET_ALL}")
        try:
            captioned = os.path.join(CLIPS_DIR, f"clip_{i:02d}_captioned.mp4")
            render_captions_on_clip(vp, word_timestamps, seg['start'], config, captioned)
            faded = os.path.join(CLIPS_DIR, f"clip_{i:02d}_faded.mp4")
            _apply_audio_fades(captioned, faded)
            out = os.path.join(OUTPUT_DIR, f"short_{i+1:02d}.mp4")
            compose_final_short(vp, faded, config, i, out)
            final_outputs.append(out)
        except Exception as e:
            logger.error(f"Render {i+1} failed: {e}")
            print(f"    {Fore.RED}Skipped.{Style.RESET_ALL}")

    _progress_bar(100, "Done!")
    n = len(final_outputs)
    if n == 0:
        return False
    print(f"\n{Fore.GREEN}{n} Short(s) created!{Style.RESET_ALL}")
    for f in final_outputs:
        print(f"  {os.path.basename(f)} ({os.path.getsize(f)/(1024*1024):.1f} MB)")

    if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
        print(f"\n{Fore.CYAN}  Uploading to Telegram...{Style.RESET_ALL}")
        try:
            upload_shorts_to_telegram(OUTPUT_DIR, config['telegram_bot_token'], config['telegram_chat_id'])
        except Exception as e:
            logger.error(f"Upload failed: {e}")
    return True


# ═══════════════════════════════════════════════════════════════════════
# v2 PIPELINE (INTERACTIVE MODE)
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline():
    from downloader import download_video, get_video_metadata
    from transcriber import transcribe_video, save_transcription
    from viral_scorer import score_all_segments
    from clip_extractor import extract_clip, reframe_to_vertical
    from caption_renderer import render_captions_on_clip
    from motion_graphics import compose_final_short
    from telegram_uploader import upload_shorts_to_telegram, validate_bot_token

    config = load_config()

    # ── HuggingFace token setup (one-time) ──
    if config.get('speaker_camera_enabled') and not config.get('hf_token'):
        print(f"\n{Fore.CYAN}{'=' * 55}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  SPEAKER DETECTION SETUP (one-time){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 55}{Style.RESET_ALL}")
        print(f"  Speaker tracking uses pyannote.audio (free, open-source).")
        print(f"  It requires a free HuggingFace token.\n")
        print(f"  HOW TO GET YOUR FREE TOKEN (1 minute):")
        print(f"  1. Go to https://huggingface.co and create a free account")
        print(f"  2. Visit: https://hf.co/settings/tokens")
        print(f"  3. Click 'New token' -> name it 'shorts-factory' -> Role: Read")
        print(f"  4. Copy the token (starts with hf_...)")
        print(f"  5. Visit this page and click 'Agree':")
        print(f"     https://hf.co/pyannote/speaker-diarization-community-1\n")

        token = input(f"  {Fore.YELLOW}Paste your HuggingFace token (or press Enter to skip): {Style.RESET_ALL}").strip()
        if token:
            _save_to_env('HF_TOKEN', token)
            config['hf_token'] = token
            if _DOTENV_LOADED:
                os.environ['HF_TOKEN'] = token
            print(f"  {Fore.GREEN}Speaker detection ready (saved to .env).{Style.RESET_ALL}")
        else:
            print(f"  {Fore.YELLOW}Skipped. Speaker camera will use center-crop fallback.{Style.RESET_ALL}")

    # ── STEP 1: YouTube URL ──
    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  STEP 1 OF 2 \u2014 VIDEO INPUT{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")

    while True:
        url = input(f"  {Fore.YELLOW}URL: {Style.RESET_ALL}").strip()
        if not is_valid_youtube_url(url):
            print(f"  {Fore.RED}Invalid URL.{Style.RESET_ALL}")
            continue
        try:
            metadata = get_video_metadata(url)
        except Exception:
            print(f"  {Fore.RED}Cannot access video.{Style.RESET_ALL}")
            continue
        break

    print(f"\n  Found: {Fore.WHITE}'{metadata['title']}'{Style.RESET_ALL}")
    print(f"  Channel: {Fore.WHITE}{metadata['channel']}{Style.RESET_ALL}")
    print(f"  Duration: {Fore.WHITE}{_fmt_time(metadata['duration'])}{Style.RESET_ALL}")

    proceed = input(f"\n  {Fore.YELLOW}Proceed? (y/n): {Style.RESET_ALL}").strip().lower()
    if proceed != 'y':
        print(f"  {Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
        return

    # ── Download ──
    _progress_bar(10, "Downloading video...")
    try:
        video_info = download_video(url, DOWNLOADS_DIR)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return
    video_path = video_info['path']
    video_id = video_info['video_id']

    # ── Transcribe ──
    _progress_bar(25, "Transcribing with Whisper...")
    try:
        srt_text, word_timestamps = transcribe_video(video_path,
            model_size=config.get('whisper_model', 'base'), language=config.get('language', 'en'))
        save_transcription(srt_text, word_timestamps, video_id, CAPTIONS_DIR)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return
    if not word_timestamps:
        print(f"  {Fore.RED}No speech detected.{Style.RESET_ALL}")
        return

    # ── Speaker camera ──
    speaker_timeline = []
    if config.get('speaker_camera_enabled') and config.get('hf_token'):
        _progress_bar(40, "Running speaker diarization (pyannote)...")
        try:
            from speaker_camera import run_full_speaker_pipeline
            speaker_timeline = run_full_speaker_pipeline(
                video_path, config['hf_token'], video_id, config)
        except Exception as e:
            logger.warning(f"Speaker camera failed: {e}")
            _progress_bar(50, "Speaker camera failed \u2014 using fallback")
    else:
        _progress_bar(50, "Speaker camera disabled \u2014 using center-crop")

    # ── Viral scoring ──
    _progress_bar(60, "Scoring all segments for virality...")
    try:
        segments = score_all_segments(
            video_path, word_timestamps, audio_path=video_path,
            min_sec=config.get('short_duration_min', 30) - 5,
            max_sec=config.get('short_duration_max', 58),
            step_sec=config.get('viral_window_step_sec', 5),
            candidate_durations=config.get('viral_candidate_durations', [30, 40, 50, 55]),
            num_shorts=config.get('shorts_count', 5),
            min_self_containment=config.get('viral_score_min_self_containment', 0.40),
        )
    except Exception as e:
        logger.error(f"Viral scoring failed: {e}")
        return
    if not segments:
        print(f"  {Fore.RED}No suitable segments found.{Style.RESET_ALL}")
        return

    _draw_viral_table(segments, config.get('show_score_breakdown', True))

    # ── Extract + Smart Reframe ──
    _progress_bar(75, "Extracting + smart reframing clips...")
    vertical_paths = []
    for i, seg in enumerate(segments):
        print(f"\n  {Fore.CYAN}-- Clip #{i+1}/{len(segments)} --{Style.RESET_ALL}")
        try:
            clip_path = extract_clip(video_path, seg['start'], seg['end'], i, CLIPS_DIR)
            if speaker_timeline:
                from speaker_camera import reframe_with_speaker_tracking
                vp = reframe_with_speaker_tracking(
                    clip_path, speaker_timeline, seg['start'],
                    os.path.join(CLIPS_DIR, f"clip_{i:02d}_smart_reframe.mp4"),
                    smooth_factor=config.get('speaker_camera_smooth_factor', 0.85),
                    face_vertical_pos=config.get('speaker_face_vertical_position', 0.30),
                )
            else:
                vp = reframe_to_vertical(clip_path, CLIPS_DIR, config.get('background_blur', True))
            vertical_paths.append((i, seg, vp))
        except Exception as e:
            logger.error(f"Clip {i} failed: {e}")
            print(f"    {Fore.RED}Skipped.{Style.RESET_ALL}")

    if not vertical_paths:
        print(f"\n  {Fore.RED}All clips failed.{Style.RESET_ALL}")
        return

    # ── Render captions + motion graphics ──
    _progress_bar(88, "Rendering captions + motion graphics...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_outputs = []
    for i, seg, vp in vertical_paths:
        print(f"\n  {Fore.CYAN}-- Rendering #{i+1} --{Style.RESET_ALL}")
        try:
            captioned = os.path.join(CLIPS_DIR, f"clip_{i:02d}_captioned.mp4")
            render_captions_on_clip(vp, word_timestamps, seg['start'], config, captioned)
            faded = os.path.join(CLIPS_DIR, f"clip_{i:02d}_faded.mp4")
            _apply_audio_fades(captioned, faded)
            out = os.path.join(OUTPUT_DIR, f"short_{i+1:02d}.mp4")
            compose_final_short(vp, faded, config, i, out)
            final_outputs.append(out)
        except Exception as e:
            logger.error(f"Render {i+1} failed: {e}")
            print(f"    {Fore.RED}Skipped.{Style.RESET_ALL}")

    _progress_bar(100, "Done!")
    n = len(final_outputs)
    if n == 0:
        print(f"\n  {Fore.RED}No shorts created.{Style.RESET_ALL}")
        return

    print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{n} Short(s) created successfully!{Style.RESET_ALL}")
    print(f"  Location: {Fore.WHITE}{OUTPUT_DIR}/{Style.RESET_ALL}")
    for f in final_outputs:
        print(f"    {os.path.basename(f)} ({os.path.getsize(f)/(1024*1024):.1f} MB)")
    print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")

    # ── Telegram upload ──
    if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
        print(f"\n{Fore.CYAN}  Telegram credentials found. Uploading...{Style.RESET_ALL}")
        try:
            upload_shorts_to_telegram(OUTPUT_DIR, config['telegram_bot_token'], config['telegram_chat_id'])
            _print_done(n)
            return
        except Exception as e:
            print(f"  {Fore.RED}Upload failed: {e}{Style.RESET_ALL}")

    # ── Prompt for Telegram ──
    print(f"\n{Fore.YELLOW}{'=' * 55}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  STEP 2 OF 2 \u2014 SEND TO TELEGRAM{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 55}{Style.RESET_ALL}")
    print(f"  HOW TO GET BOT TOKEN: Telegram @BotFather -> /newbot")
    print(f"  HOW TO GET CHAT ID: Send a msg to your bot, check")
    print(f"  https://api.telegram.org/bot<TOKEN>/getUpdates\n")

    bot_token = None
    for attempt in range(3):
        bot_token = input(f"  {Fore.YELLOW}Bot Token: {Style.RESET_ALL}").strip()
        if not bot_token:
            continue
        try:
            validate_bot_token(bot_token)
            break
        except ValueError as e:
            print(f"  {Fore.RED}{e}{Style.RESET_ALL}")
            if attempt == 2:
                _print_done(n)
                return

    chat_id = input(f"\n  {Fore.YELLOW}Chat ID: {Style.RESET_ALL}").strip()
    if not chat_id:
        _print_done(n)
        return

    # Save credentials to .env (never to config.json)
    _save_to_env('TELEGRAM_BOT_TOKEN', bot_token)
    _save_to_env('TELEGRAM_CHAT_ID', chat_id)
    print(f"  {Fore.GREEN}Credentials saved to .env (never committed to git).{Style.RESET_ALL}")

    try:
        upload_shorts_to_telegram(OUTPUT_DIR, bot_token, chat_id)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        print(f"  {Fore.RED}Upload failed: {e}{Style.RESET_ALL}")
    _print_done(n)


def _save_to_env(key, value):
    """Append or update a key in the .env file."""
    env_path = os.path.join(BASE_DIR, '.env')
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith(f'{key}='):
                    lines.append(f'{key}={value}\n')
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f'{key}={value}\n')
    with open(env_path, 'w') as f:
        f.writelines(lines)


def _print_done(n):
    print(f"\n{Fore.GREEN}{'=' * 55}{Style.RESET_ALL}")
    print(f"  ALL DONE! {n} shorts are ready.")
    print(f"  Next time: credentials are saved. Just paste a URL.")
    print(f"  Run: {Fore.CYAN}python main.py{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 55}{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def print_banner():
    print(f"""
{Fore.CYAN}===================================================
  SHORTS FACTORY v2  -  GLM-5.2
  Speaker Camera + Viral Score Engine
  Powered by Z.ai x Your AI Agent
==================================================={Style.RESET_ALL}
""")


def main():
    print_banner()
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check':
            success = run_dependency_check()
            sys.exit(0 if success else 1)
        if sys.argv[1] == '--ci' and len(sys.argv) > 2:
            url = sys.argv[2]
            try:
                ok = run_pipeline_ci(url)
                sys.exit(0 if ok else 1)
            except Exception as e:
                logger.critical(f"CI failed: {e}", exc_info=True)
                sys.exit(1)
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print(f"\n\n  {Fore.YELLOW}Interrupted.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        print(f"\n  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()