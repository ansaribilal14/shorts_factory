#!/usr/bin/env python3
"""
caption_renderer.py — Premium word-by-word caption rendering system.
Renders animated captions matching viral YouTube Shorts style.
"""

import os
import math
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, VideoClip
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)

TARGET_W, TARGET_H = 1080, 1920
WORDS_PER_GROUP = 4
CAPTION_Y_FRACTION = 0.65
ANIM_DURATION = 0.08


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except (OSError, IOError):
        for sf in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
            try:
                return ImageFont.truetype(sf, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()


def _text_dimensions(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def generate_caption_frame(words_in_group: list, current_word_idx: int,
                           frame_width: int, frame_height: int, config: dict) -> Image.Image:
    img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = config.get('caption_font_size', 72)
    font_path = config.get('caption_font', 'assets/fonts/Montserrat-ExtraBold.ttf')
    highlight_color = config.get('caption_highlight_color', '#FFD700')
    caption_color = config.get('caption_color', '#FFFFFF')
    outline_color = config.get('caption_outline_color', '#000000')
    outline_thickness = config.get('caption_outline_thickness', 4)
    font = _load_font(font_path, font_size)
    texts = []
    for i, w in enumerate(words_in_group):
        texts.append((w['word'], i == current_word_idx, font))
    total_width = 0
    text_items = []
    for text, is_highlight, fnt in texts:
        tw, th = _text_dimensions(draw, text, fnt)
        text_items.append((text, total_width, tw, th, is_highlight, fnt))
        total_width += tw + 20
    max_text_width = frame_width - 100
    if total_width > max_text_width:
        scale = max_text_width / total_width
        new_size = max(32, int(font_size * scale))
        font = _load_font(font_path, new_size)
        total_width = 0
        text_items = []
        for text, is_highlight, fnt in texts:
            tw, th = _text_dimensions(draw, text, font)
            text_items.append((text, total_width, tw, th, is_highlight, font))
            total_width += tw + 20
    x_start = (frame_width - total_width) // 2
    y_center = int(frame_height * CAPTION_Y_FRACTION)
    if text_items:
        max_height = max(item[3] for item in text_items)
        y_text = y_center - max_height // 2
    else:
        return img
    shadow_offset = 3
    shadow_color = (0, 0, 0, 150)
    shadow_img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_img)
    for text, x_off, tw, th, is_highlight, fnt in text_items:
        x = x_start + x_off + shadow_offset
        y = y_text + shadow_offset
        shadow_draw.text((x, y), text, fill=shadow_color, font=fnt,
                         stroke_width=outline_thickness, stroke_fill=shadow_color)
    img = Image.alpha_composite(img, shadow_img)
    draw = ImageDraw.Draw(img)
    for text, x_off, tw, th, is_highlight, fnt in text_items:
        x = x_start + x_off
        y = y_text
        fill_color = highlight_color if is_highlight else caption_color
        draw.text((x, y), text, fill=fill_color, font=fnt,
                  stroke_width=outline_thickness, stroke_fill=outline_color)
    return img


def render_captions_on_clip(clip_path: str, word_timestamps: list,
                            clip_start_offset: float, config: dict, output_path: str) -> str:
    print(f"    {Fore.CYAN}  Rendering captions (word-by-word)...{Style.RESET_ALL}")
    clip = VideoFileClip(clip_path)
    clip_duration = clip.duration
    clip_end = clip_start_offset + clip_duration
    filtered_words = [w for w in word_timestamps if clip_start_offset - 0.5 <= w['start'] <= clip_end + 0.5]
    if not filtered_words:
        logger.warning(f"No word timestamps for clip offset {clip_start_offset}")
        clip.close()
        import shutil
        shutil.copy2(clip_path, output_path)
        return output_path
    adjusted_words = []
    for w in filtered_words:
        adj_start = max(0, min(w['start'] - clip_start_offset, clip_duration))
        adj_end = max(0, min(w['end'] - clip_start_offset, clip_duration))
        if adj_end > adj_start:
            adjusted_words.append({'word': w['word'], 'start': adj_start, 'end': adj_end})
    groups = [adjusted_words[i:i + WORDS_PER_GROUP] for i in range(0, len(adjusted_words), WORDS_PER_GROUP)]
    frame_w, frame_h = clip.size

    def make_caption_frame(t):
        active_group_idx = -1
        for gi, group in enumerate(groups):
            if group[0]['start'] <= t <= group[-1]['end'] + 0.5:
                active_group_idx = gi
                break
            if t < group[0]['start']:
                break
        if active_group_idx < 0:
            return np.array(Image.new('RGBA', (frame_w, frame_h), (0, 0, 0, 0)))
        group = groups[active_group_idx]
        current_word_idx = 0
        for i, w in enumerate(group):
            if w['start'] <= t <= w['end']:
                current_word_idx = i
                break
            if t > w['end']:
                current_word_idx = i + 1
        current_word_idx = min(current_word_idx, len(group) - 1)
        caption_img = generate_caption_frame(group, current_word_idx, frame_w, frame_h, config)
        current_word = group[current_word_idx]
        time_since_start = t - current_word['start']
        if 0 <= time_since_start <= ANIM_DURATION:
            progress = time_since_start / ANIM_DURATION
            scale = 0.85 + 0.15 * (1 - (1 - progress) ** 3)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            caption_img = caption_img.resize((new_w, new_h), Image.LANCZOS)
            final_img = Image.new('RGBA', (frame_w, frame_h), (0, 0, 0, 0))
            final_img.paste(caption_img, ((frame_w - new_w) // 2, (frame_h - new_h) // 2))
            caption_img = final_img
        return np.array(caption_img)

    caption_clip = VideoClip(make_caption_frame, duration=clip_duration, ismask=False)
    final = CompositeVideoClip([clip, caption_clip], size=(frame_w, frame_h))
    final = final.set_audio(clip.audio).set_duration(clip_duration)
    final.write_videofile(output_path, codec='libx264', audio_codec='aac',
                          bitrate='8000k', audio_bitrate='192k', preset='medium', threads=4, logger=None)
    clip.close()
    caption_clip.close()
    final.close()
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    {Fore.GREEN}  Captioned: {output_path} ({file_size_mb:.1f} MB){Style.RESET_ALL}")
    return output_path