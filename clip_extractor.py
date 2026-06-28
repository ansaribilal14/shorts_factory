#!/usr/bin/env python3
"""
clip_extractor.py — Extracts and reframes video clips for YouTube Shorts.
Converts 16:9 horizontal video to 9:16 vertical with blurred background fill.
"""

import os
import logging

import ffmpeg
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)


def extract_clip(source_video_path: str, start_sec: float, end_sec: float,
                 clip_index: int, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"clip_{clip_index:02d}_{int(start_sec)}s_{int(end_sec)}s.mp4")
    duration = end_sec - start_sec
    print(f"    {Fore.WHITE}Extracting clip {clip_index + 1}: {_fmt_time(start_sec)} -> {_fmt_time(end_sec)} ({duration:.1f}s){Style.RESET_ALL}")
    try:
        (ffmpeg.input(source_video_path, ss=start_sec, to=end_sec)
         .output(output_path, c='copy', loglevel='error').run(overwrite_output=True))
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            logger.warning(f"Stream copy failed for clip {clip_index}, re-encoding...")
            (ffmpeg.input(source_video_path, ss=start_sec, to=end_sec)
             .output(output_path, vcodec='libx264', acodec='aac', preset='ultrafast', crf='23', loglevel='error')
             .run(overwrite_output=True))
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    {Fore.GREEN}  Saved: {output_path} ({file_size_mb:.1f} MB){Style.RESET_ALL}")
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error extracting clip {clip_index}: {e.stderr}")
        raise RuntimeError(f"Failed to extract clip {clip_index}") from e


def reframe_to_vertical(clip_path: str, output_dir: str, blur_background: bool = True) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(clip_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_vertical.mp4")
    print(f"    {Fore.CYAN}  Reframing to 9:16 vertical...{Style.RESET_ALL}")
    try:
        clip = VideoFileClip(clip_path)
        w, h = clip.size
        target_w, target_h = 1080, 1920
        aspect = w / h
        if aspect < 1.0 or (0.55 < aspect < 0.65):
            print(f"    {Fore.WHITE}  Already vertical ({w}x{h}), resizing...{Style.RESET_ALL}")
            resized = clip.resize((target_w, target_h)).set_audio(clip.audio)
            resized.write_videofile(output_path, codec='libx264', audio_codec='aac',
                                    bitrate='8000k', audio_bitrate='192k', preset='medium', threads=4, logger=None)
            clip.close()
            resized.close()
            return output_path
        print(f"    {Fore.WHITE}  Horizontal ({w}x{h}), applying blurred background...{Style.RESET_ALL}")
        if blur_background:
            scale_factor = target_h / h
            bg_width = int(w * scale_factor)
            bg_path = os.path.join(output_dir, f"{base_name}_bg_blurred.mp4")
            (ffmpeg.input(clip_path)
             .filter('scale', bg_width, target_h)
             .filter('crop', target_w, target_h)
             .filter('boxblur', 30, 30)
             .output(bg_path, vcodec='libx264', acodec='aac', preset='ultrafast', crf='28', loglevel='error')
             .run(overwrite_output=True))
            bg_clip = VideoFileClip(bg_path)
            clip_width = target_w
            clip_height = int(h * (target_w / w))
            if clip_height > target_h:
                clip_height = target_h
                clip_width = int(w * (target_h / h))
            resized_clip = clip.resize((clip_width, clip_height))
            y_pos = (target_h - clip_height) // 2
            final = CompositeVideoClip([bg_clip, resized_clip.set_position((0, y_pos))], size=(target_w, target_h))
            final = final.set_audio(clip.audio).set_duration(clip.duration)
            final.write_videofile(output_path, codec='libx264', audio_codec='aac',
                                  bitrate='8000k', audio_bitrate='192k', preset='medium', threads=4, logger=None)
            bg_clip.close()
            if os.path.exists(bg_path):
                os.remove(bg_path)
        else:
            resized_clip = clip.resize((target_w, int(h * (target_w / w))))
            y_pos = (target_h - resized_clip.size[1]) // 2
            bg = ColorClip(size=(target_w, target_h), color=(0, 0, 0))
            final = CompositeVideoClip([bg, resized_clip.set_position((0, y_pos))], size=(target_w, target_h))
            final = final.set_audio(clip.audio).set_duration(clip.duration)
            final.write_videofile(output_path, codec='libx264', audio_codec='aac',
                                  bitrate='8000k', audio_bitrate='192k', preset='medium', threads=4, logger=None)
            bg.close()
        clip.close()
        final.close()
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    {Fore.GREEN}  Vertical clip saved: {output_path} ({file_size_mb:.1f} MB){Style.RESET_ALL}")
        return output_path
    except Exception as e:
        logger.error(f"Error reframing clip: {e}")
        raise RuntimeError(f"Failed to reframe: {e}") from e


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"