#!/usr/bin/env python3
"""
main.py — Shorts Factory Orchestrator.
Usage:
    python main.py              # Run the full pipeline
    python main.py --check      # Verify all dependencies
    python main.py --ci <URL>   # CI mode: no prompts, uses config.json for everything
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
CAPTIONS_DIR = os.path.join(BASE_DIR, "captions")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
    return logging.getLogger("shorts_factory")


logger = setup_logging()


def _progress_bar(pct: float, label: str, bar_width: int = 30):
    filled = int(bar_width * pct / 100)
    bar = '\u2588' * filled + '\u2591' * (bar_width - filled)
    print(f"\n  {Fore.GREEN}[{bar}]{Style.RESET_ALL} {pct:5.1f}% \u2014 {Fore.WHITE}{label}{Style.RESET_ALL}")


def run_dependency_check():
    print(f"\n{Fore.CYAN}  Running dependency check...{Style.RESET_ALL}\n")
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
        ("numpy", "numpy"),
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

    config_ok = os.path.exists(CONFIG_PATH)
    sym = f"{Fore.GREEN}\u2713{Style.RESET_ALL}" if config_ok else f"{Fore.RED}\u2717{Style.RESET_ALL}"
    print(f"  {sym} config.json")
    checks.append(config_ok)
    print()

    if all(checks):
        print(f"  {Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}  ALL CHECKS PASSED! Shorts Factory is ready.{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        return True
    else:
        failed = sum(1 for c in checks if not c)
        print(f"  {Fore.RED}{'=' * 50}{Style.RESET_ALL}")
        print(f"  {Fore.RED}  {failed} check(s) failed.{Style.RESET_ALL}")
        print(f"  {Fore.RED}{'=' * 50}{Style.RESET_ALL}")
        return False


def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def save_config(config: dict):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def is_valid_youtube_url(url: str) -> bool:
    return any(url.startswith(p) for p in [
        "https://www.youtube.com/", "https://youtu.be/",
        "https://m.youtube.com/", "http://www.youtube.com/", "http://youtu.be/"])


def _fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _draw_segment_table(segments: list):
    if not segments:
        return
    col_widths = [7, 10, 10, 7, 45]
    header = ["Short", "Start", "End", "Dur", "Preview"]
    print(f"\n  {'+'.join('-' * w for w in col_widths)}")
    print(f"  {'|'.join(f' {h:<{w-2}} ' for h, w in zip(header, col_widths))}|")
    print(f"  {'+'.join('-' * w for w in col_widths)}")
    for i, seg in enumerate(segments):
        dur = int(seg['end'] - seg['start'])
        preview = seg.get('transcript_excerpt', '')[:40]
        if len(seg.get('transcript_excerpt', '')) > 40:
            preview += "..."
        row = [f"  #{i+1}", _fmt_time(seg['start']), _fmt_time(seg['end']), f"  {dur}s", f"  {preview}"]
        print(f"  {'|'.join(f'{r:<{w}}' for r, w in zip(row, col_widths))}|")
    print(f"  {'+'.join('-' * w for w in col_widths)}\n")


def _apply_audio_fades(input_path: str, output_path: str):
    import ffmpeg as ff
    (ff.input(input_path).filter('afade', t='in', st=0, d=0.05)
     .filter('afade', t='out', st=-0.05, d=0.05)
     .output(output_path, c='copy', loglevel='error').run(overwrite_output=True))


def run_pipeline_ci(youtube_url: str):
    """CI mode — no prompts, uses config.json for all settings including Telegram."""
    from downloader import download_video, get_video_metadata
    from transcriber import transcribe_video, save_transcription, find_best_short_segments
    from clip_extractor import extract_clip, reframe_to_vertical
    from caption_renderer import render_captions_on_clip
    from motion_graphics import compose_final_short
    from telegram_uploader import upload_shorts_to_telegram

    config = load_config()
    print(f"\n{Fore.CYAN}  CI MODE — Processing: {youtube_url}{Style.RESET_ALL}")

    try:
        metadata = get_video_metadata(youtube_url)
    except Exception as e:
        logger.error(f"Metadata fetch failed: {e}")
        print(f"  {Fore.RED}Cannot access video: {e}{Style.RESET_ALL}")
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

    _progress_bar(30, "Transcribing audio (Whisper)...")
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

    _progress_bar(50, "Finding best segments...")
    segments = find_best_short_segments(word_timestamps, metadata,
        num_shorts=config.get('shorts_count', 5),
        min_sec=config.get('short_duration_min', 30),
        max_sec=config.get('short_duration_max', 58))

    if not segments:
        print(f"  {Fore.RED}No suitable segments found.{Style.RESET_ALL}")
        return False

    print(f"\n  {Fore.GREEN}Found {len(segments)} segments:{Style.RESET_ALL}")
    _draw_segment_table(segments)

    _progress_bar(70, "Extracting & reframing...")
    clip_paths, vertical_paths = [], []
    for i, seg in enumerate(segments):
        print(f"\n  {Fore.CYAN}-- Clip #{i+1}/{len(segments)} --{Style.RESET_ALL}")
        try:
            cp = extract_clip(video_path, seg['start'], seg['end'], i, CLIPS_DIR)
            vp = reframe_to_vertical(cp, CLIPS_DIR, config.get('background_blur', True))
            clip_paths.append(cp)
            vertical_paths.append(vp)
        except Exception as e:
            logger.error(f"Clip {i} failed: {e}")
            clip_paths.append(None)
            vertical_paths.append(None)

    valid_indices = [i for i, vp in enumerate(vertical_paths) if vp is not None]
    if not valid_indices:
        print(f"  {Fore.RED}All clips failed.{Style.RESET_ALL}")
        return False

    _progress_bar(85, "Rendering captions + motion graphics...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_outputs = []
    for i in valid_indices:
        print(f"\n  {Fore.CYAN}-- Rendering #{i+1} --{Style.RESET_ALL}")
        try:
            captioned_path = os.path.join(CLIPS_DIR, f"clip_{i:02d}_captioned.mp4")
            render_captions_on_clip(vertical_paths[i], word_timestamps, segments[i]['start'], config, captioned_path)
            faded_path = os.path.join(CLIPS_DIR, f"clip_{i:02d}_faded.mp4")
            _apply_audio_fades(captioned_path, faded_path)
            output_path = os.path.join(OUTPUT_DIR, f"short_{i+1:02d}.mp4")
            compose_final_short(vertical_paths[i], faded_path, config, i, output_path)
            final_outputs.append(output_path)
        except Exception as e:
            logger.error(f"Render {i+1} failed: {e}")
            print(f"    {Fore.RED}Skipped: {e}{Style.RESET_ALL}")

    _progress_bar(100, "Done!")
    n = len(final_outputs)
    if n == 0:
        print(f"\n  {Fore.RED}No shorts created.{Style.RESET_ALL}")
        return False

    print(f"\n{Fore.GREEN}{n} Short(s) created!{Style.RESET_ALL}")
    for f in final_outputs:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")

    # Upload to Telegram if credentials exist
    if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
        print(f"\n{Fore.CYAN}  Uploading to Telegram...{Style.RESET_ALL}")
        try:
            upload_shorts_to_telegram(OUTPUT_DIR, config['telegram_bot_token'], config['telegram_chat_id'])
        except Exception as e:
            logger.error(f"Telegram upload failed: {e}")
            print(f"  {Fore.RED}Upload failed: {e}{Style.RESET_ALL}")
    else:
        print(f"\n  {Fore.YELLOW}No Telegram credentials — skipping upload.{Style.RESET_ALL}")

    return True


def run_pipeline():
    """Interactive mode — prompts user for URL and Telegram credentials."""
    from downloader import download_video, get_video_metadata
    from transcriber import transcribe_video, save_transcription, find_best_short_segments
    from clip_extractor import extract_clip, reframe_to_vertical
    from caption_renderer import render_captions_on_clip
    from motion_graphics import compose_final_short
    from telegram_uploader import upload_shorts_to_telegram, validate_bot_token

    config = load_config()

    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  STEP 1 OF 2 \u2014 VIDEO INPUT{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"  Paste your YouTube video URL below.\n")

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

    _progress_bar(10, "Downloading video...")
    try:
        video_info = download_video(url, DOWNLOADS_DIR)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return
    video_path = video_info['path']
    video_id = video_info['video_id']

    _progress_bar(30, "Transcribing audio (Whisper)...")
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

    _progress_bar(50, "Finding best segments...")
    segments = find_best_short_segments(word_timestamps, metadata,
        num_shorts=config.get('shorts_count', 5),
        min_sec=config.get('short_duration_min', 30),
        max_sec=config.get('short_duration_max', 58))
    if not segments:
        print(f"  {Fore.RED}No suitable segments found.{Style.RESET_ALL}")
        return
    print(f"\n  {Fore.GREEN}Found {len(segments)} great segments:{Style.RESET_ALL}")
    _draw_segment_table(segments)

    _progress_bar(70, "Extracting & reframing clips to 9:16...")
    clip_paths, vertical_paths = [], []
    for i, seg in enumerate(segments):
        print(f"\n  {Fore.CYAN}-- Processing Short #{i+1}/{len(segments)} --{Style.RESET_ALL}")
        try:
            cp = extract_clip(video_path, seg['start'], seg['end'], i, CLIPS_DIR)
            vp = reframe_to_vertical(cp, CLIPS_DIR, config.get('background_blur', True))
            clip_paths.append(cp)
            vertical_paths.append(vp)
        except Exception as e:
            logger.error(f"Clip {i} failed: {e}")
            clip_paths.append(None)
            vertical_paths.append(None)

    valid_indices = [i for i, vp in enumerate(vertical_paths) if vp is not None]
    if not valid_indices:
        print(f"\n  {Fore.RED}All clips failed.{Style.RESET_ALL}")
        return

    _progress_bar(85, "Rendering captions + motion graphics...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_outputs = []
    for i in valid_indices:
        print(f"\n  {Fore.CYAN}-- Rendering Short #{i+1} --{Style.RESET_ALL}")
        try:
            captioned_path = os.path.join(CLIPS_DIR, f"clip_{i:02d}_captioned.mp4")
            render_captions_on_clip(vertical_paths[i], word_timestamps, segments[i]['start'], config, captioned_path)
            faded_path = os.path.join(CLIPS_DIR, f"clip_{i:02d}_faded.mp4")
            _apply_audio_fades(captioned_path, faded_path)
            output_path = os.path.join(OUTPUT_DIR, f"short_{i+1:02d}.mp4")
            compose_final_short(vertical_paths[i], faded_path, config, i, output_path)
            final_outputs.append(output_path)
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

    # Telegram upload
    if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
        print(f"\n{Fore.CYAN}  Telegram credentials found. Uploading...{Style.RESET_ALL}")
        try:
            upload_shorts_to_telegram(OUTPUT_DIR, config['telegram_bot_token'], config['telegram_chat_id'])
            _print_done(n)
            return
        except Exception as e:
            print(f"  {Fore.RED}Upload failed: {e}{Style.RESET_ALL}")

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

    config['telegram_bot_token'] = bot_token
    config['telegram_chat_id'] = chat_id
    save_config(config)
    print(f"  {Fore.GREEN}Credentials saved to config.json.{Style.RESET_ALL}")

    try:
        upload_shorts_to_telegram(OUTPUT_DIR, bot_token, chat_id)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        print(f"  {Fore.RED}Upload failed: {e}{Style.RESET_ALL}")
    _print_done(n)


def _print_done(n: int):
    print(f"\n{Fore.GREEN}{'=' * 55}{Style.RESET_ALL}")
    print(f"  ALL DONE! {n} shorts are ready.")
    print(f"  Next time: credentials are saved. Just paste a URL.")
    print(f"  Run: {Fore.CYAN}python main.py{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 55}{Style.RESET_ALL}")


def print_banner():
    print(f"""
{Fore.CYAN}===================================================
  SHORTS FACTORY  -  GLM-5.2
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
                logger.critical(f"CI pipeline failed: {e}", exc_info=True)
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