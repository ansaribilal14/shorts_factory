#!/usr/bin/env python3
"""
downloader.py — YouTube video downloader using yt-dlp.
Downloads the best quality video (up to 1080p) and merges with audio.
"""

import os
import re
import json
from pathlib import Path

import yt_dlp
from colorama import Fore, Style, init

init(autoreset=True)


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = name[:80]
    return name


def download_video(url: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'outtmpl': os.path.join(output_dir, '%(id)s_%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [_progress_hook],
        'noplaylist': True,
        'max_filesize': 10 * 1024 * 1024 * 1024,
    }
    print(f"\n{Fore.CYAN}  Downloading video...{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}URL: {url}{Style.RESET_ALL}\n")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', 'unknown')
            title = info.get('title', 'Untitled')
            duration = int(info.get('duration', 0))
            sanitized = _sanitize_filename(title)
            filename = f"{video_id}_{sanitized}.mp4"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                for f in os.listdir(output_dir):
                    if f.startswith(video_id) and f.endswith('.mp4'):
                        filepath = os.path.join(output_dir, f)
                        break
            if not os.path.exists(filepath):
                raise FileNotFoundError("Downloaded file not found")
            print(f"\n{Fore.GREEN}  Download complete!{Style.RESET_ALL}")
            print(f"  File: {Fore.WHITE}{filepath}{Style.RESET_ALL}")
            print(f"  Duration: {Fore.WHITE}{duration}s ({duration // 60}:{duration % 60:02d}){Style.RESET_ALL}")
            return {"path": filepath, "title": title, "duration": duration, "video_id": video_id}
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if 'private' in error_msg:
            print(f"\n{Fore.RED}  ERROR: This video is private.{Style.RESET_ALL}")
        elif 'age' in error_msg or 'sign in' in error_msg:
            print(f"\n{Fore.RED}  ERROR: This video is age-restricted.{Style.RESET_ALL}")
        elif 'unavailable' in error_msg or 'not found' in error_msg:
            print(f"\n{Fore.RED}  ERROR: Video unavailable or removed.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}  ERROR: {e}{Style.RESET_ALL}")
        raise
    except Exception as e:
        print(f"\n{Fore.RED}  Unexpected error: {e}{Style.RESET_ALL}")
        raise


def get_video_metadata(url: str) -> dict:
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': False, 'skip_download': True}
    print(f"{Fore.CYAN}  Fetching video metadata...{Style.RESET_ALL}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            chapters = info.get('chapters', [])
            formatted_chapters = [{"title": ch.get('title', ''), "start": ch.get('start_time', 0), "end": ch.get('end_time', 0)} for ch in chapters] if chapters else []
            metadata = {
                "title": info.get('title', 'Untitled'),
                "description": info.get('description', '')[:500],
                "duration": int(info.get('duration', 0)),
                "chapters": formatted_chapters,
                "upload_date": info.get('upload_date', 'Unknown'),
                "channel": info.get('uploader', 'Unknown'),
                "video_id": info.get('id', 'unknown'),
                "thumbnail": info.get('thumbnail', ''),
            }
            print(f"  {Fore.GREEN}Title:{Style.RESET_ALL} {metadata['title']}")
            print(f"  {Fore.GREEN}Channel:{Style.RESET_ALL} {metadata['channel']}")
            print(f"  {Fore.GREEN}Duration:{Style.RESET_ALL} {metadata['duration']}s ({metadata['duration'] // 60}:{metadata['duration'] % 60:02d})")
            return metadata
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e).lower()
        if 'private' in error_msg:
            print(f"{Fore.RED}  ERROR: This video is private.{Style.RESET_ALL}")
        elif 'unavailable' in error_msg or 'not found' in error_msg:
            print(f"{Fore.RED}  ERROR: Video unavailable.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}  ERROR: {e}{Style.RESET_ALL}")
        raise
    except Exception as e:
        print(f"{Fore.RED}  Error: {e}{Style.RESET_ALL}")
        raise


def _progress_hook(d: dict):
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        downloaded = d.get('downloaded_bytes', 0)
        if total > 0:
            pct = downloaded / total * 100
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
            speed = d.get('speed', 0)
            speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed else "N/A"
            print(f"\r  {Fore.GREEN}[{bar}]{Style.RESET_ALL} {pct:5.1f}% | {speed_str}    ", end='', flush=True)
    elif d['status'] == 'finished':
        print(f"\r  {Fore.GREEN}  Download finished. Merging...{Style.RESET_ALL}{' ' * 40}")