#!/usr/bin/env python3
"""
motion_graphics.py — Motion graphics overlays for YouTube Shorts polish.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, VideoClip
from colorama import Fore, Style, init

init(autoreset=True)
TARGET_W, TARGET_H = 1080, 1920


def _load_font(font_path: str, size: int):
    try:
        return ImageFont.truetype(font_path, size)
    except (OSError, IOError):
        for sf in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(sf, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()


def _ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def add_intro_animation(clip, duration: float = 1.5):
    def make_frame(t):
        frame = clip.get_frame(t)
        if t < duration:
            progress = t / duration
            alpha = int(255 * (1 - _ease_out_cubic(progress)))
            overlay = Image.new('RGBA', (TARGET_W, TARGET_H), (0, 0, 0, alpha))
            overlay_np = np.array(overlay)
            frame_rgb = frame[:, :, :3].astype(np.float32)
            overlay_alpha = (overlay_np[:, :, 3:4].astype(np.float32) / 255.0)
            blended = frame_rgb * (1 - overlay_alpha)
            return blended.astype(np.uint8)
        return frame
    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip = new_clip.set_audio(clip.audio)
    return new_clip


def add_outro_animation(clip, duration: float = 1.0):
    def make_frame(t):
        frame = clip.get_frame(t)
        t_from_end = clip.duration - t
        if t_from_end < duration:
            progress = 1 - (t_from_end / duration)
            alpha = int(255 * _ease_out_cubic(progress))
            overlay = Image.new('RGBA', (TARGET_W, TARGET_H), (0, 0, 0, alpha))
            overlay_np = np.array(overlay)
            frame_rgb = frame[:, :, :3].astype(np.float32)
            overlay_alpha = (overlay_np[:, :, 3:4].astype(np.float32) / 255.0)
            blended = frame_rgb * (1 - overlay_alpha)
            frame = blended.astype(np.uint8)
            if t_from_end < 0.8:
                sub_progress = 1 - (t_from_end / 0.8)
                text_alpha = int(255 * min(1.0, sub_progress * 3))
                pulse = 1.0 + 0.05 * math.sin(t * 4 * math.pi)
                font_size = int(64 * pulse)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                font = _load_font("assets/fonts/Montserrat-ExtraBold.ttf", font_size)
                text = "SUBSCRIBE"
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx, ty = (TARGET_W - tw) // 2, (TARGET_H - th) // 2
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font,
                          stroke_width=3, stroke_fill=(255, 0, 0))
                frame = np.array(img)
        return frame
    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip = new_clip.set_audio(clip.audio)
    return new_clip


def add_progress_bar(clip, color: str = "#FF0000", height: int = 6):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    def make_frame(t):
        frame = clip.get_frame(t)
        progress = t / clip.duration if clip.duration > 0 else 0
        bar_width = int(TARGET_W * progress)
        y_start = TARGET_H - height
        if bar_width > 0:
            frame[y_start:TARGET_H, :bar_width, 0] = r
            frame[y_start:TARGET_H, :bar_width, 1] = g
            frame[y_start:TARGET_H, :bar_width, 2] = b
        return frame
    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip = new_clip.set_audio(clip.audio)
    return new_clip


def add_subscribe_bug(clip, position: str = "top_right", appear_at: float = 3.0):
    def make_frame(t):
        frame = clip.get_frame(t)
        if t < appear_at:
            return frame
        fade_progress = min(1.0, (t - appear_at) / 0.5)
        alpha = int(255 * fade_progress)
        pulse_time = (t - appear_at) % 1.5
        if pulse_time < 0.75:
            scale = 1.0 + 0.05 * (pulse_time / 0.75)
        else:
            scale = 1.0 + 0.05 * (1.0 - (pulse_time - 0.75) / 0.75)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, 'RGBA')
        font_size = int(36 * scale)
        font = _load_font("assets/fonts/Montserrat-ExtraBold.ttf", font_size)
        text = "SUBSCRIBE"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        padding_x, padding_y = 24, 12
        pill_w, pill_h = tw + padding_x * 2, th + padding_y * 2
        px = TARGET_W - pill_w - 20 if position == "top_right" else 20
        py = 60
        draw.rounded_rectangle([(px, py), (px + pill_w, py + pill_h)],
                               radius=20, fill=(220, 40, 40, alpha))
        draw.text((px + padding_x, py + padding_y), text, fill=(255, 255, 255, alpha), font=font)
        return np.array(img)
    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip = new_clip.set_audio(clip.audio)
    return new_clip


def add_chapter_watermark(clip, chapter_title: str, clip_index: int):
    label = chapter_title.strip()[:30] if chapter_title and chapter_title.strip() else f"Part {clip_index + 1}"
    def make_frame(t):
        frame = clip.get_frame(t)
        fade_alpha = min(1.0, t / 0.5) if t < 0.5 else 1.0
        alpha = int(200 * fade_alpha)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, 'RGBA')
        font = _load_font("assets/fonts/Montserrat-Bold.ttf", 32)
        draw.text((30, TARGET_H - 80), label, fill=(255, 255, 255, alpha), font=font)
        return np.array(img)
    new_clip = VideoClip(make_frame, duration=clip.duration)
    new_clip = new_clip.set_audio(clip.audio)
    return new_clip


def compose_final_short(vertical_clip_path: str, captioned_clip_path: str,
                        config: dict, clip_index: int, output_path: str) -> str:
    print(f"    {Fore.CYAN}  Composing final short with motion graphics...{Style.RESET_ALL}")
    clip = VideoFileClip(captioned_clip_path)
    if config.get('add_progress_bar', True):
        clip = add_progress_bar(clip, color=config.get('progress_bar_color', '#FF0000'), height=6)
        print(f"      {Fore.WHITE}+ Progress bar{Style.RESET_ALL}")
    if config.get('add_subscribe_bug', True):
        clip = add_subscribe_bug(clip, position="top_right", appear_at=3.0)
        print(f"      {Fore.WHITE}+ Subscribe badge{Style.RESET_ALL}")
    clip = add_chapter_watermark(clip, "", clip_index)
    if config.get('add_intro_animation', True):
        clip = add_intro_animation(clip, duration=1.5)
        print(f"      {Fore.WHITE}+ Intro animation{Style.RESET_ALL}")
    if config.get('add_outro_animation', True):
        clip = add_outro_animation(clip, duration=1.0)
        print(f"      {Fore.WHITE}+ Outro animation{Style.RESET_ALL}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac',
                         bitrate=config.get('video_bitrate', '8000k'),
                         audio_bitrate=config.get('audio_bitrate', '192k'),
                         preset='slow', threads=4, logger=None)
    clip.close()
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    {Fore.GREEN}  Final: {output_path} ({file_size_mb:.1f} MB){Style.RESET_ALL}")
    return output_path