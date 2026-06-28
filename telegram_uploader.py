#!/usr/bin/env python3
"""
telegram_uploader.py — Upload finished shorts to Telegram via a bot.
"""

import os
import asyncio
import logging
from pathlib import Path
import requests
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)
TELEGRAM_MAX_FILE_SIZE = 50 * 1024 * 1024


def validate_bot_token(token: str) -> str:
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('ok'):
            bot_username = data['result'].get('username', 'unknown')
            print(f"  {Fore.GREEN}Bot validated: @{bot_username}{Style.RESET_ALL}")
            return bot_username
        raise ValueError(f"Invalid bot token: {data.get('description', 'Unknown error')}")
    except requests.RequestException as e:
        raise ValueError(f"Network error validating token: {e}")


async def _upload_single_video(bot, chat_id: int, video_path: str, index: int, total: int):
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    if file_size > TELEGRAM_MAX_FILE_SIZE:
        await bot.send_message(chat_id=chat_id, text=f"Short #{index + 1} is {file_size_mb:.1f} MB (exceeds 50 MB limit). Skipping.")
        return False
    progress_msg = await bot.send_message(chat_id=chat_id, text=f"Short #{index + 1} uploading... ({file_size_mb:.1f} MB)")
    try:
        caption = f"Short {index + 1}/{total} ready!\nSave to YouTube Shorts, TikTok, or Reels.\n\nMade with GLM-5.2 Shorts Factory"
        with open(video_path, 'rb') as vf:
            await bot.send_video(chat_id=chat_id, video=vf, caption=caption,
                                 supports_streaming=True, width=1080, height=1920)
        try:
            await progress_msg.delete()
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error(f"Upload failed for {video_path}: {e}")
        await bot.send_message(chat_id=chat_id, text=f"Error uploading short #{index + 1}: {str(e)[:100]}")
        return False


async def _upload_all_async(output_dir: str, bot_token: str, chat_id: int):
    from telegram import Bot
    bot = Bot(token=bot_token)
    mp4_files = sorted(Path(output_dir).glob("*.mp4"), key=lambda p: p.name)
    if not mp4_files:
        print(f"  {Fore.YELLOW}No MP4 files in {output_dir}{Style.RESET_ALL}")
        return
    total = len(mp4_files)
    print(f"\n  {Fore.CYAN}Uploading {total} shorts to Telegram...{Style.RESET_ALL}\n")
    success_count = 0
    for i, vp in enumerate(mp4_files):
        print(f"  {Fore.WHITE}[{i + 1}/{total}] {vp.name}...{Style.RESET_ALL}")
        if await _upload_single_video(bot, chat_id, str(vp), i, total):
            success_count += 1
            print(f"  {Fore.GREEN}  Uploaded.{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}  Failed.{Style.RESET_ALL}")
    await bot.send_message(chat_id=chat_id, text=f"All {total} shorts delivered! ({success_count}/{total} successful)")
    print(f"\n  {Fore.GREEN}Upload complete: {success_count}/{total}{Style.RESET_ALL}")


def upload_shorts_to_telegram(output_dir: str, bot_token: str, chat_id: str) -> None:
    chat_id_int = int(chat_id)
    validate_bot_token(bot_token)
    try:
        asyncio.get_event_loop().run_until_complete(_upload_all_async(output_dir, bot_token, chat_id_int))
    except RuntimeError:
        asyncio.run(_upload_all_async(output_dir, bot_token, chat_id_int))