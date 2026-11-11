"""Sliding-window and timepoint EEG feature construction."""

from __future__ import annotations

import numpy as np


def make_sliding_features(
    data: np.ndarray,
    time: np.ndarray,
    *,
    step_ms: float = 10.0,
    span_ms: float = 10.0,
    window_ms: float = 100.0,
    toi: tuple[float, float] = (-0.05, 1.2),
    mode: str = "sliding-window",
) -> tuple[np.ndarray, np.ndarray]:
    """Build decoding features on CPU.

    data: EEG values, shape (trial, channel, time).
    time: sample times in seconds, shape (time,).
    step_ms: spacing between window centres in milliseconds.
    span_ms: duration of each averaging segment in milliseconds.
    window_ms: total sliding-window duration in milliseconds.
    toi: time range (start, stop) in seconds, using the same reference as time.
    mode: "sliding-window" demeans each window, averages segments and flattens them;
          "timepoint" selects raw samples without windowing, demeaning or averaging.

    returns: features (trial, feature, output time) and times (output time,).
        sliding-window: feature = channel x segment; times are window centres.
        timepoint: feature = channel; times are selected sample times.
        step_ms, span_ms and window_ms apply only to sliding-window mode.
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)

    hz = round(1.0 / np.median(np.diff(time)))
    time_range_indices = np.flatnonzero((time >= toi[0]) & (time <= toi[1]))

    if mode == "timepoint":
        return np.asarray(data[:, :, time_range_indices]), time[time_range_indices]

    window_centres_s = np.arange(
        time[time_range_indices[0]],
        time[time_range_indices[-1]] + step_ms / 2000,
        step_ms / 1000,
    )
    window_sample_indices = np.array([
        np.argmin(np.abs(time - centre)) for centre in window_centres_s
    ])
    window_sample_indices = np.unique(window_sample_indices)

    window_samples = round(window_ms * hz / 1000)
    span_samples = round(span_ms * hz / 1000)
    samples_before = window_samples // 2
    samples_after = window_samples - samples_before
    window_sample_indices = window_sample_indices[
        (window_sample_indices >= samples_before)
        & (window_sample_indices + samples_after <= time.size)
    ]

    n_trials, n_channels, _ = data.shape
    n_segments = window_samples // span_samples
    features = np.empty(
        (n_trials, n_channels * n_segments, window_sample_indices.size), dtype=data.dtype
    )

    for output_index, sample_index in enumerate(window_sample_indices):
        window = data[
            :, :, sample_index - samples_before : sample_index + samples_after
        ]
        window = window - window.mean(axis=2, keepdims=True)
        window = window.reshape(n_trials, n_channels, n_segments, span_samples)
        window = window.mean(axis=3)
        features[:, :, output_index] = window.reshape(
            n_trials, n_channels * n_segments
        )
    return features, time[window_sample_indices]
