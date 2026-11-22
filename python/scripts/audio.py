"""Audio input for Whisper: WAV decoding, resampling to 16 kHz, and log-mel features.

The feature extraction follows Hugging Face's WhisperFeatureExtractor, so the output matches
what Whisper was trained on: 16 kHz mono, a 400-sample Hann window with hop 160, 80 Slaney-scaled
mel bins, log10 power, clipped to 8 dB below the peak, then scaled to roughly [-1, 1].
Only WAV (PCM) is decoded here, because the app must not depend on ffmpeg.
"""
from __future__ import annotations

import io
import math
import wave
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 480000
N_FRAMES = CHUNK_SAMPLES // HOP              # 3000 frames per 30 s


class AudioError(ValueError):
    pass


def read_wav(data: bytes) -> np.ndarray:
    """Decode WAV bytes to mono float32 samples at 16 kHz."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError("could not read the audio; send a PCM WAV file") from exc

    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        x = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioError(f"unsupported WAV sample width: {width * 8} bits")

    if channels > 1:
        x = x[: len(x) - len(x) % channels].reshape(-1, channels).mean(axis=1)
    return resample(x.astype(np.float32), rate, SAMPLE_RATE)


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(x) == 0:
        return x
    n_out = max(1, round(len(x) * dst_rate / src_rate))
    positions = np.linspace(0, len(x) - 1, n_out)
    return np.interp(positions, np.arange(len(x)), x).astype(np.float32)


def split_chunks(samples: np.ndarray) -> list[np.ndarray]:
    """Split audio into consecutive 30 s windows. Short audio gives one window."""
    if len(samples) == 0:
        raise AudioError("the audio is empty")
    return [samples[i:i + CHUNK_SAMPLES] for i in range(0, len(samples), CHUNK_SAMPLES)]


@lru_cache(maxsize=1)
def _mel_filters() -> np.ndarray:
    """Slaney-scaled, Slaney-normalized triangular filters: shape (N_FFT // 2 + 1, N_MELS)."""
    def hz_to_mel(freq):
        min_log_hz, min_log_mel = 1000.0, 15.0
        logstep = math.log(6.4) / 27.0
        mels = 3.0 * freq / 200.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels):
        min_log_hz, min_log_mel = 1000.0, 15.0
        logstep = math.log(6.4) / 27.0
        freq = 200.0 * mels / 3.0
        mask = mels >= min_log_mel
        freq = np.where(mask, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freq)
        return freq

    n_bins = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, n_bins)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), N_MELS + 2)
    filter_freqs = mel_to_hz(mel_points)
    diffs = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diffs[:-1]
    up = slopes[:, 2:] / diffs[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS])
    return (filters * enorm[None, :]).astype(np.float32)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    n = np.arange(N_FFT)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / N_FFT)).astype(np.float32)  # periodic Hann


def log_mel(chunk: np.ndarray) -> np.ndarray:
    """One chunk of up to 30 s of audio -> log-mel features of shape (80, 3000)."""
    x = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    n = min(len(chunk), CHUNK_SAMPLES)
    x[:n] = chunk[:n]

    padded = np.pad(x, N_FFT // 2, mode="reflect")
    n_frames = 1 + CHUNK_SAMPLES // HOP            # 3001; the last one is dropped below
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * _window()[None, :]
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2   # (3001, 201)
    mel = (power[:-1] @ _mel_filters()).T                        # (80, 3000)

    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
