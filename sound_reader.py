#!/usr/bin/env python
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
# import onnx
import onnxruntime as ort

from file_utils import get_bundle_filepath

SAMPLE_RATE = 32000

is_windows = False

current_platform = platform.system()
if current_platform == "Windows":
    ffmpeg_path = r".\ffmpeg\windows\ffmpeg.exe"
    is_windows = True
elif current_platform == "Darwin":  # macOS
    ffmpeg_path = r"./ffmpeg/osx/ffmpeg"
else:  # Linux
    ffmpeg_path = r"./ffmpeg/linux/ffmpeg"

ffmpeg_path = get_bundle_filepath(ffmpeg_path)

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def subsample(frame: np.ndarray, scale_factor: int) -> np.ndarray:
    subframe = frame[:len(frame) - (len(frame) % scale_factor)].reshape(
        -1, scale_factor)
    subframe_mean = subframe.max(axis=1)

    subsample = subframe_mean

    if len(frame) % scale_factor != 0:
        residual_frame = frame[len(frame) - (len(frame) % scale_factor):]
        residual_mean = residual_frame.max()
        subsample = np.append(subsample, residual_mean)

    return subsample


def get_segments(scores: np.ndarray, precision: int, threshold: float,
                 offset: int) -> list:

    seq_iter = iter(np.where(scores > threshold)[0])
    try:
        seq = next(seq_iter)
        pred = scores[seq]
        segment = {'start': seq, 'end': seq, 'pred': pred}
    except StopIteration:
        return

    for seq in seq_iter:
        pred = scores[seq]
        if seq - 1 == segment['end']:
            segment['end'] = seq
            segment['pred'] = max(segment['pred'], pred)
        else:
            segment['start'] = segment['start']
            segment['end'] = segment['end']

            yield segment
            segment = {'start': seq, 'end': seq, 'pred': pred}

    yield segment


def compute_timestamps(
    framewise_output: np.ndarray,
    precision: int,
    threshold: float,
    focus_idx: int,
    offset: int,
):
    focus = framewise_output[:, focus_idx]
    # precision in the amount of milliseconds per timestamp sample (higher values will result in less precise timestamps)

    subsampled_scores = subsample(focus, precision)
    segments = map(
        lambda segment: {
            'start': segment['start'] * precision / 100 + offset,
            'end': segment['end'] * precision / 100 + offset + 1,
            'pred': round(float(segment['pred']), 6)
        }, get_segments(subsampled_scores, precision, threshold, offset))
    return segments


def load_audio(file: str, sr: int, frame_count: int):
    cmd = [
        ffmpeg_path, '-hide_banner', '-loglevel', 'warning', '-i', file,
        '-filter_complex', '[0:a]asetpts=PTS-STARTPTS[audio]', '-map',
        '[audio]', '-ac', '1', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar',
        str(sr), '-'
    ]

    # Specify subprocess options to suppress the command prompt on Windows
    subprocess_options = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }

    if is_windows:
        subprocess_options['creationflags'] = subprocess.CREATE_NO_WINDOW

    ffmpeg_process = subprocess.run(cmd, **subprocess_options)

    return np.frombuffer(ffmpeg_process.stdout, dtype=np.int16).reshape(
        1, -1).astype(np.float32) / (2**15)


def get_timestamps(file, precision=100, block_size=600, threshold=0.90, focus_idx=58, model="bdetectionmodel_05_01_23"):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = ort.InferenceSession(model,
                                       sess_options,
                                       providers=ort.get_available_providers())

    offset = 0
    blocks = load_audio(file, SAMPLE_RATE, SAMPLE_RATE * block_size)
    np.array_split(blocks, block_size)

    info = {'filename': file, 'timestamps': []}

    for block in blocks:
        block = block[np.newaxis, :]
        if block.shape[1] >= SAMPLE_RATE:
            ort_inputs = {'input': block}
            framewise_output = ort_session.run(['output'], ort_inputs)[0]

            preds = framewise_output[0]
            info['timestamps'].extend(
                compute_timestamps(preds, precision, threshold,
                                   focus_idx, offset))

        offset += block_size

    return info
