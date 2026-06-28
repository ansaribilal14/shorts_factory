#!/usr/bin/env python3
"""
speaker_camera.py — Speaker-Focused Smart Camera for Shorts Factory v2.
Combines pyannote.audio diarization + MediaPipe face detection to create
a smooth, speaker-locked vertical reframe that follows the active speaker.
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)
logger = logging.getLogger(__name__)

TARGET_W, TARGET_H = 1080, 1920


# ═══════════════════════════════════════════════════════════════════════
# PART 1: SPEAKER DIARIZATION (who speaks when)
# ═══════════════════════════════════════════════════════════════════════

def run_speaker_diarization(video_path: str, hf_token: str,
                           output_dir: str = None,
                           video_id: str = "video") -> list:
    """
    Run speaker diarization using pyannote.audio community-1.

    Returns:
        List of speaker turns: [{"speaker": str, "start": float, "end": float}]
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captions")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  {Fore.CYAN}  Running speaker diarization (pyannote)...{Style.RESET_ALL}")

    # Extract audio to temp WAV
    temp_wav = os.path.join(output_dir, f"{video_id}_audio.wav")
    try:
        subprocess.run(
            ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16k', '-ar', '16000',
             '-ac', '1', '-y', temp_wav],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        raise RuntimeError(f"Failed to extract audio for diarization") from e

    # Run pyannote
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=hf_token
        )
        # Use CPU to avoid GPU memory issues
        diarization = pipeline(temp_wav, num_speakers=None)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "speaker": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            })

        # Save to JSON
        diar_path = os.path.join(output_dir, f"{video_id}_diarization.json")
        with open(diar_path, 'w') as f:
            json.dump(turns, f, indent=2)
        print(f"  {Fore.GREEN}  Diarization saved: {diar_path}{Style.RESET_ALL}")

        # Print stats
        speakers = set(t["speaker"] for t in turns)
        print(f"  {Fore.GREEN}  {len(speakers)} speaker(s) detected: {', '.join(sorted(speakers))}{Style.RESET_ALL}")
        for sp in sorted(speakers):
            total_time = sum(t["end"] - t["start"] for t in turns if t["speaker"] == sp)
            print(f"    {sp}: {total_time:.1f}s speaking time")

        return turns

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        print(f"  {Fore.RED}  Diarization failed: {e}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}  Will use center-crop fallback.{Style.RESET_ALL}")
        raise
    finally:
        # Cleanup temp WAV
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def identify_primary_speaker(diarization_turns: list) -> str:
    """
    Identify the speaker with the most cumulative speaking time.

    Returns:
        Speaker label string (e.g. "SPEAKER_00").
    """
    if not diarization_turns:
        return "SPEAKER_00"

    speaker_time = {}
    for turn in diarization_turns:
        sp = turn["speaker"]
        duration = turn["end"] - turn["start"]
        speaker_time[sp] = speaker_time.get(sp, 0) + duration

    primary = max(speaker_time, key=speaker_time.get)
    print(f"  {Fore.CYAN}  Primary speaker: {primary} "
          f"({speaker_time[primary]:.1f}s){Style.RESET_ALL}")

    return primary


# ═══════════════════════════════════════════════════════════════════════
# PART 2: FACE DETECTION + TRACKING (where is the speaker)
# ═══════════════════════════════════════════════════════════════════════

def build_face_track(video_path: str, video_id: str = "video",
                     sample_fps: int = 2,
                     output_dir: str = None) -> list:
    """
    Detect faces in sampled frames using MediaPipe.

    Returns:
        List of face track entries:
        [{"time": float, "faces": [{"x", "y", "w", "h", "confidence"}]}]
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captions")
    os.makedirs(output_dir, exist_ok=True)

    import cv2
    import mediapipe as mp

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / source_fps if source_fps > 0 else 1

    # Sample every Nth frame
    frame_interval = max(1, int(source_fps / sample_fps))
    total_samples = total_frames // frame_interval

    face_track = []
    frame_idx = 0

    print(f"  {Fore.CYAN}  Building face track ({sample_fps} fps, ~{total_samples} frames)...{Style.RESET_ALL}")

    with tqdm(total=total_samples, desc="  Face detection", unit="frame",
              bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / source_fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb_frame.shape[:2]

                results = detector.process(rgb_frame)

                faces = []
                if results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        fx = int(bbox.xmin * w)
                        fy = int(bbox.ymin * h)
                        fw = int(bbox.width * w)
                        fh = int(bbox.height * h)
                        conf = det.score[0] if det.score else 0.0
                        faces.append({
                            "x": fx, "y": fy,
                            "w": fw, "h": fh,
                            "confidence": round(conf, 3)
                        })

                face_track.append({"time": round(timestamp, 3), "faces": faces})
                pbar.update(1)

            frame_idx += 1

    cap.release()
    detector.close()

    # Save
    track_path = os.path.join(output_dir, f"{video_id}_face_track.json")
    with open(track_path, 'w') as f:
        json.dump(face_track, f, indent=2)
    print(f"  {Fore.GREEN}  Face track saved: {track_path} ({len(face_track)} samples){Style.RESET_ALL}")

    return face_track


# ═══════════════════════════════════════════════════════════════════════
# PART 3: MATCH SPEAKER TO FACE
# ═══════════════════════════════════════════════════════════════════════

def match_speaker_to_face(diarization_turns: list, face_track: list,
                          primary_speaker_label: str) -> list:
    """
    Match audio speaker turns with detected face positions.

    Returns:
        Smoothed speaker position timeline:
        [{"time": float, "speaker": str,
          "face_bbox": {"x", "y", "w", "h"} or None}]
    """
    if not diarization_turns or not face_track:
        return []

    print(f"  {Fore.CYAN}  Matching speakers to faces...{Style.RESET_ALL}")

    # Build a timeline at face track timestamps
    timeline = []

    for ft_entry in face_track:
        t = ft_entry["time"]
        faces = ft_entry["faces"]

        # Find which speaker is active at this time
        active_speaker = None
        for turn in diarization_turns:
            if turn["start"] <= t <= turn["end"]:
                active_speaker = turn["speaker"]
                break

        # Select the best face for this frame
        best_face = None
        if faces:
            if len(faces) == 1:
                best_face = faces[0]
            else:
                # Multiple faces: pick largest (closest to camera = main subject)
                best_face = max(faces, key=lambda f: f["w"] * f["h"])

        timeline.append({
            "time": t,
            "speaker": active_speaker,
            "face_bbox": best_face,
        })

    # Smooth the timeline — interpolate gaps and remove jumps
    smoothed = _smooth_timeline(timeline)

    active_count = sum(1 for e in smoothed if e["face_bbox"] is not None)
    print(f"  {Fore.GREEN}  Speaker-face match: {active_count}/{len(smoothed)} "
          f"frames with detected face{Style.RESET_ALL}")

    return smoothed


def _smooth_timeline(timeline: list, max_jump_ratio: float = 0.20) -> list:
    """
    Apply smoothing to the face position timeline:
    1. Fill gaps (use last known position)
    2. Interpolate jumps > 20% of frame width
    """
    if not timeline:
        return []

    smoothed = []
    last_known_bbox = None
    prev_bbox = None

    for entry in timeline:
        bbox = entry["face_bbox"]

        if bbox is not None:
            # Check for a jump from previous position
            if prev_bbox is not None and last_known_bbox is not None:
                frame_w_est = 1920  # estimate source width
                jump_threshold = frame_w_est * max_jump_ratio

                dx = abs(bbox["x"] - prev_bbox["x"])
                if dx > jump_threshold:
                    # Interpolate: blend towards new position (60/40)
                    bbox = {
                        "x": int(0.6 * prev_bbox["x"] + 0.4 * bbox["x"]),
                        "y": int(0.6 * prev_bbox["y"] + 0.4 * bbox["y"]),
                        "w": bbox["w"],
                        "h": bbox["h"],
                        "confidence": bbox.get("confidence", 0.5),
                    }

            last_known_bbox = bbox
            prev_bbox = bbox
        else:
            # No face detected — use last known position
            bbox = last_known_bbox

        smoothed.append({
            "time": entry["time"],
            "speaker": entry["speaker"],
            "face_bbox": bbox,
        })

    return smoothed


# ═══════════════════════════════════════════════════════════════════════
# PART 4: SMART VERTICAL REFRAME (the "AI camera")
# ═══════════════════════════════════════════════════════════════════════

def reframe_with_speaker_tracking(source_clip_path: str,
                                 speaker_timeline: list,
                                 clip_start_offset: float,
                                 output_path: str,
                                 target_w: int = 1080,
                                 target_h: int = 1920,
                                 smooth_factor: float = 0.85,
                                 face_vertical_pos: float = 0.30) -> str:
    """
    Create a vertical reframe that tracks the speaker's face.

    The crop follows the speaker with smooth EMA-based camera movement.
    Falls back to center-crop if no face data is available.
    """
    import cv2
    from moviepy.editor import VideoFileClip, CompositeVideoClip, VideoClip, ColorClip

    print(f"    {Fore.CYAN}  Smart reframing with speaker tracking...{Style.RESET_ALL}")

    clip = VideoFileClip(source_clip_path)
    src_w, src_h = clip.size
    clip_duration = clip.duration

    if not speaker_timeline:
        print(f"    {Fore.YELLOW}  No speaker timeline — using center crop.{Style.RESET_ALL}")
        clip.close()
        # Fallback to basic vertical reframe
        from clip_extractor import reframe_to_vertical
        return reframe_to_vertical(source_clip_path,
                                   os.path.dirname(output_path), blur_background=True)

    # Offset timeline to clip-local time
    local_timeline = []
    for entry in speaker_timeline:
        local_t = entry["time"] - clip_start_offset
        if -1.0 <= local_t <= clip_duration + 1.0:
            local_timeline.append({
                "time": local_t,
                "speaker": entry["speaker"],
                "face_bbox": entry["face_bbox"],
            })

    # Determine if source needs upscaling to fill target_w
    scale = 1.0
    if src_w < target_w:
        scale = target_w / src_w
    effective_w = int(src_w * scale)
    effective_h = int(src_h * scale)

    # Pre-compute blurred background for areas outside the crop
    bg_path = output_path.replace(".mp4", "_bg.mp4")
    import ffmpeg as ff
    (ff.input(source_clip_path)
     .filter('scale', effective_w, effective_h)
     .filter('boxblur', 30, 30)
     .output(bg_path, vcodec='libx264', acodec='aac',
             preset='ultrafast', crf='28', loglevel='error')
     .run(overwrite_output=True))
    bg_clip = VideoFileClip(bg_path)

    # Open source video with cv2 for frame-by-frame processing
    cap = cv2.VideoCapture(source_clip_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # State for EMA smoothing
    smooth_crop_x = effective_w / 2
    smooth_crop_y = effective_h / 2
    no_face_counter = 0
    fallback_logged = set()

    def get_crop_position(t: float):
        """Get the smoothed crop center for time t."""
        nonlocal smooth_crop_x, smooth_crop_y, no_face_counter

        # Find closest timeline entries
        best_entry = None
        min_dt = float('inf')
        for entry in local_timeline:
            dt = abs(entry["time"] - t)
            if dt < min_dt:
                min_dt = dt
                best_entry = entry

        target_x = effective_w / 2
        target_y = effective_h / 2
        face_found = False

        if best_entry and best_entry["face_bbox"] is not None:
            bbox = best_entry["face_bbox"]
            # Scale bbox coordinates if we upscaled
            face_cx = (bbox["x"] + bbox["w"] / 2) * scale
            face_cy = (bbox["y"] + bbox["h"] / 2) * scale

            # Target: center crop on face, position face in upper portion
            target_x = face_cx
            target_y = face_cy - (target_h * face_vertical_pos - target_h / 2)
            face_found = True
            no_face_counter = 0

            # Log fallback usage
            t_key = int(t * 10)
            if t_key in fallback_logged:
                fallback_logged.discard(t_key)
        else:
            no_face_counter += 1

        # EMA smoothing
        alpha = 1.0 - smooth_factor
        smooth_crop_x = smooth_factor * smooth_crop_x + alpha * target_x
        smooth_crop_y = smooth_factor * smooth_crop_y + alpha * target_y

        # Clamp to source bounds
        half_w = target_w / 2
        half_h = target_h / 2
        smooth_crop_x = max(half_w, min(effective_w - half_w, smooth_crop_x))
        smooth_crop_y = max(half_h, min(effective_h - half_h, smooth_crop_y))

        return int(smooth_crop_x), int(smooth_crop_y), face_found

    def make_frame(t):
        ret, frame = cap.read()
        if not ret:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Resize if needed
        if scale != 1.0:
            frame = cv2.resize(frame, (effective_w, effective_h))

        cx, cy, face_found = get_crop_position(t)

        # Calculate crop window
        x1 = max(0, cx - target_w // 2)
        y1 = max(0, cy - target_h // 2)
        x2 = min(effective_w, x1 + target_w)
        y2 = min(effective_h, y1 + target_h)

        # Adjust if we hit edges
        if x2 - x1 < target_w:
            x1 = max(0, x2 - target_w)
        if y2 - y1 < target_h:
            y1 = max(0, y2 - target_h)

        cropped = frame[y1:y2, x1:x2]

        # If crop is smaller than target (source too small), pad and use bg
        if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            bg_frame = bg_clip.get_frame(t)
            result = bg_frame.copy()
            # Center the small crop
            oy = (target_h - cropped.shape[0]) // 2
            ox = (target_w - cropped.shape[1]) // 2
            result[oy:oy + cropped.shape[0], ox:ox + cropped.shape[1]] = cropped
        else:
            result = cv2.resize(cropped, (target_w, target_h))

        # Convert BGR -> RGB for moviepy
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Build output video
    import moviepy.editor as mpy
    output_clip = mpy.VideoClip(make_frame, duration=clip_duration)
    output_clip = output_clip.set_audio(clip.audio)

    output_clip.write_videofile(
        output_path, codec='libx264', audio_codec='aac',
        bitrate='8000k', audio_bitrate='192k', preset='medium',
        threads=4, logger=None,
    )

    clip.close()
    bg_clip.close()
    output_clip.close()
    cap.close()

    # Cleanup temp bg
    if os.path.exists(bg_path):
        os.remove(bg_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    {Fore.GREEN}  Smart reframe saved: {output_path} ({size_mb:.1f} MB){Style.RESET_ALL}")
    return output_path


def run_full_speaker_pipeline(video_path: str, hf_token: str,
                              video_id: str = "video",
                              config: dict = None) -> list:
    """
    Run the full speaker camera pipeline:
    1. Diarization
    2. Face tracking
    3. Speaker-face matching

    Returns:
        Smoothed speaker timeline list, or empty list on failure.
    """
    if config is None:
        config = {}

    captions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captions")
    sample_fps = config.get("speaker_camera_sample_fps", 2)

    try:
        # Step 1: Diarization
        diarization_turns = run_speaker_diarization(video_path, hf_token, captions_dir, video_id)

        # Step 2: Identify primary speaker
        primary = identify_primary_speaker(diarization_turns)

        # Step 3: Face tracking
        face_track = build_face_track(video_path, video_id, sample_fps, captions_dir)

        # Step 4: Match speakers to faces
        speaker_timeline = match_speaker_to_face(diarization_turns, face_track, primary)

        return speaker_timeline

    except Exception as e:
        logger.warning(f"Speaker camera pipeline failed, will use fallback: {e}")
        print(f"  {Fore.YELLOW}  Speaker camera unavailable — will use center-crop fallback.{Style.RESET_ALL}")
        return []