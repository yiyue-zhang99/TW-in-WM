"""Fixed spatiotemporal features matching Wolff's Fig. 6/8 MATLAB script."""
import numpy as np


def make_fixed_features(data, time, sfreq, onset, relative_window=(.100, .400)):
    """Return trial x (channel*time) x 1; select (start, stop] (default 100–400 ms).

    Demean each trial/channel within the selected window, apply MATLAB
    movmean(..., span, 'Endpoints','discard'), and retain every span-th
    sample. No prestimulus baseline, temporal score smoothing, or scaling.
    Inputs follow the fixed pipelines' trial x channel x time format.
    """
    data = np.asarray(data, dtype=np.float32)
    time = np.asarray(time, dtype=float).reshape(-1)
    span = round(sfreq * .010)
    aligned = np.round(time * sfreq) / sfreq
    lo, hi = map(float, relative_window)
    start, stop = onset + lo, onset + hi
    # Tolerance prevents representation error from changing boundary inclusion.
    indices = np.flatnonzero((aligned > start + 1e-9) & (aligned <= stop + 1e-9))
    window = data[:, :, indices].copy()
    window -= window.mean(axis=2, keepdims=True)
    windows = np.lib.stride_tricks.sliding_window_view(window, span, axis=2)
    reduced = windows.mean(axis=-1)[:, :, ::span]
    features = reduced.reshape(data.shape[0], -1, 1)
    return features, np.array([start, stop])
