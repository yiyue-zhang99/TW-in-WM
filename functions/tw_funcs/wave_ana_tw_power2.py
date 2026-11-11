"""
Travelling-wave power calculation utilities.

Main input format
-----------------
the_data : numpy.ndarray, shape (n_channels, n_times, n_trials)
    EEG signal data. Axis 0 is eeg channel, axis 1 is time sample,
    and axis 2 is trial.

electrode_lines : list of list of int, shape (n_lines, n_channels_per_line)
    Data format: [[line1_channel_indices], [line2_channel_indices], ...].
    Each inner list contains the channel indices for one line, in spatial order
    along that line (posterior to anterior).
    For example:
        electrode_lines = [
            [0, 3, 7, 10],
            [2, 5, 9, 14],
            [20, 24, 28, 32],
        ]
    Each line is analyzed separately, and the indices must match the channel
    order in the first axis of the_data and in pos.

chn_dist_lines : list of 1D array-like, shape (n_lines, n_channels_per_line)
    Only required for nonuniform NUFFT. 
    Data format:[line1_distances, line2_distances, ...]. It must have the same number of
    lines as electrode_lines, and each distance vector must match the number
    of channels in the corresponding electrode line. Usually this is created
    with build_chn_dist_lines(electrode_lines, pos).

pos : array-like, shape (n_channels, 3)
    Cartesian electrode coordinates in the same channel order as the first
    axis of the_data. Pass this to build_chn_dist_lines to create
    chn_dist_lines for the nonuniform method.

sfreq : float, default 500
    Sampling frequency in Hz.

window_size : int, default 250
    Length (in samples) of each sliding analysis window.

step : int, default 10
    Hop size (in samples) between consecutive window starts. Window wi
    starts at sample wi*step; there are n_win = 1 + (n_times - window_size)//step
    windows.

shuffle_reps : int, default 10
    Number of phase-shuffled surrogates generated per window to build the
    null baseline. Only used when baseline_mode="surr".

freq_lo, freq_hi : float, default 1.0, 40.0
    Temporal frequency band in Hz kept in the output; ff (and the frequency
    axis of every *Max array) is restricted to [freq_lo, freq_hi].

baseline_mode : {"surr", "mean"}, default "surr"
    "surr": baseline is the travelling-wave amplitude averaged over the
    phase-shuffled surrogates (returned in fwMaxSS / bwMaxSS); FFTf is None.
    "mean": baseline is the temporal 1D-FFT mean (returned in FFTf), and
    fwMaxSS / bwMaxSS alias it.

max_batch : int, default 1000
    Maximum number of trials (and surrogates) pushed through the FFT/NUFFT
    in one chunk. Caps peak working memory; results are identical regardless
    of its value.

rng : numpy.random.Generator or None, default None
    Random generator used for surrogate generation. None -> a fresh
    np.random.default_rng().

eps : float, default 1e-9
    NUFFT accuracy tolerance. Only used by wave_trigger_computer_nonuniform.

spatial_demean : bool, default False
    If True, subtract the across-electrode mean at each time point (within
    each window) before the FFT, removing the spatially uniform component.

return_dict : bool, default False
    If True, return the full result dict; if False, return the 7-tuple
    described in the "Outputs" section below.

Typical calls
-------------
chn_dist_lines = build_chn_dist_lines(electrode_lines, pos)
wave_trigger_computer_nonuniform(the_data, electrode_lines, chn_dist_lines)
    Use this when electrodes along a line are not equally spaced.

wave_trigger_computer_uniform(the_data, electrode_lines)
    Use this when electrodes along each line can be treated as equally spaced.

Outputs
-------
With return_dict=True a dict is returned; with return_dict=False the same
values come back as a 7-tuple in the order below. The travelling-wave
direction comes from the sign of the spatial frequency (wavenumber) in the
space x time 2D-FFT: for each temporal frequency the peak amplitude is taken
over the forward wavenumbers (fy < 0) and, separately, the backward
wavenumbers (fy > 0).

fwMax : ndarray, shape (n_lines, n_freqs, n_win, n_trials)
    Forward travelling-wave amplitude (peak over forward wavenumbers).
bwMax : ndarray, same shape as fwMax
    Backward travelling-wave amplitude (peak over backward wavenumbers).
fwMaxSS : ndarray, same shape as fwMax
    Baseline for fwMax. In baseline_mode="surr" it is the forward amplitude
    averaged over the `shuffle_reps` phase-shuffled surrogates (null level);
    in baseline_mode="mean" it holds the temporal 1D-FFT mean baseline
    (identical to FFTf).
bwMaxSS : ndarray, same shape as fwMax
    Baseline for bwMax (surrogate mean). In "mean" mode it aliases fwMaxSS / FFTf.
FFTf : ndarray (same shape) or None
    Temporal 1D-FFT mean baseline; populated only when baseline_mode="mean",
    otherwise None (surrogate mode does not use it).
starts : ndarray, shape (n_win,)
    Start sample index of each sliding window; window wi spans samples
    [starts[wi], starts[wi] + window_size).
ff : ndarray, shape (n_freqs,)
    Temporal frequency axis in Hz, restricted to [freq_lo, freq_hi].
"""

import numpy as np
try:
    import finufft
except ImportError:  # Uniform FFT does not require the optional NUFFT package.
    finufft = None


def _fft_baseline_1d_batch(data, fmask):
    """
    data: (n_elec, n_times, n_batch)

    Returns
    -------
    fft_f : (n_freq, n_batch)
        1D temporal FFT amplitude averaged over electrodes for each trial/batch.
    """
    n_elec = data.shape[0]
    fft_amp = np.abs(np.fft.fftshift(np.fft.fft(data, axis=1), axes=1))
    fft_f = fft_amp.mean(axis=0)[fmask, :]
    return fft_f


def _make_surrogate_data(seg_data, shuffle_reps, rng):
    n_elec, n_times, n_trials = seg_data.shape
    surr_data = np.empty(
        (n_elec, n_times, shuffle_reps * n_trials),
        dtype=seg_data.dtype,
    )
    for tt in range(n_trials):
        for kk in range(shuffle_reps):
            out_idx = kk * n_trials + tt
            surr_data[:, :, out_idx] = seg_data[rng.permutation(n_elec), :, tt]
    return surr_data


def _nufft2d_batch(data, dist_line, sfreq, *,
                         freq_lo=1.0, freq_hi=40.0, eps=1e-9):
    """
    Regular 2D NUFFT Type-1.

    space: nonuniform normalized coordinate in [-pi, pi]
    time : uniform normalized coordinate in [-pi, pi]
    modes: integer Fourier bins
    """
    if finufft is None:
        raise ImportError(
            "finufft is required for the nonuniform method; "
            "install finufft or use wave_trigger_computer_uniform."
        )

    n_elec, n_times, n_batch = data.shape
    dist_line = np.asarray(dist_line, dtype=np.float64)
    span = float(dist_line.max() - dist_line.min())
    # Add one mean electrode spacing so the last electrode does not wrap onto the first (-pi == pi).
    span = span + span / (n_elec - 1)

    s_norm =(-np.pi + (dist_line - dist_line.min()) / span * 2 * np.pi).astype(np.float64)
    # Same for time: n_times points over [-pi, pi), so the last sample does not wrap onto the first.
    t_norm = -np.pi + 2 * np.pi * np.arange(n_times, dtype=np.float64) / n_times

    xt, xs = np.meshgrid(t_norm, s_norm)
    xs_flat = np.ascontiguousarray(xs.ravel(order="F"))
    xt_flat = np.ascontiguousarray(xt.ravel(order="F"))

    c = np.asfortranarray(data).reshape(n_elec * n_times, n_batch, order="F").T
    c = np.ascontiguousarray(c.astype(np.complex128))

    fhat = finufft.nufft2d1(
        xs_flat, xt_flat, c,
        n_modes=(n_elec, n_times),
        isign=+1,
        eps=eps,
    )
    amp = np.abs(fhat)

    fx = np.fft.fftshift(np.fft.fftfreq(n_times, d=1.0 / sfreq))
    fmask = (fx >= freq_lo) & (fx <= freq_hi)
    ff = fx[fmask]
    fy = np.fft.fftshift(np.fft.fftfreq(n_elec)) * n_elec
    amp_sel = amp[:, :, fmask]

    fw = amp_sel[:, fy < 0, :].max(axis=1).T
    bw = amp_sel[:, fy > 0, :].max(axis=1).T
    fft_f = _fft_baseline_1d_batch(data, fmask)
    return fw, bw, fft_f, ff, fy

def _uniform_fft2_batch(data, sfreq, *,
                        freq_lo=1.0, freq_hi=40.0):
    """Standard 2D FFT for equally spaced electrodes."""
    n_elec, n_times, _n_batch = data.shape

    fx = np.fft.fftshift(np.fft.fftfreq(n_times, d=1.0 / sfreq))
    fmask = (fx >= freq_lo) & (fx <= freq_hi)
    ff = fx[fmask]
    fy = np.fft.fftshift(np.fft.fftfreq(n_elec)) * n_elec

    fhat = np.fft.fft(data, axis=0)
    fhat = np.fft.fft(fhat, axis=1)
    amp = np.abs(np.fft.fftshift(fhat, axes=(0, 1)))
    amp_sel = amp[:, fmask, :]

    fw = amp_sel[fy < 0, :, :].max(axis=0)
    bw = amp_sel[fy > 0, :, :].max(axis=0)
    fft_f = _fft_baseline_1d_batch(data, fmask)
    return fw, bw, fft_f, ff, fy


def _wave_trigger_template(the_data, electrode_lines, batch_func, *,
                           chn_dist_lines=None,
                           sfreq=500,
                           window_size=250,
                           step=10,
                           shuffle_reps=10,
                           freq_lo=1.0,
                           freq_hi=40.0,
                           baseline_mode="surr",  # "surr": shuffled baseline; "mean": temporal FFT mean baseline
                           max_batch=1000,
                           spatial_demean=False,
                           rng=None,
                           **batch_kwargs):
    """
    max_batch caps how many trials are ever pushed through ``batch_func`` (and
    through surrogate generation) in one call. Trials are sliced into chunks of
    at most ``max_batch`` and processed one chunk at a time, so peak memory for
    the FFT/NUFFT working arrays stays bounded no matter how large n_trials is
    - only the final (n_lines, n_freqs, n_win, n_trials) result arrays scale
    with n_trials.
    """
    mode = baseline_mode.lower()
    rng = np.random.default_rng() if rng is None else rng
    max_batch = int(max_batch)
    if max_batch <= 0:
        raise ValueError("max_batch must be a positive integer")

    _n_ch, n_times, n_trials = the_data.shape
    starts = np.array(list(range(0, n_times - window_size + 1, step)))
    n_lines = len(electrode_lines)
    n_win = len(starts)
    trial_chunks = list(range(0, n_trials, max_batch))

    idx0 = np.asarray(electrode_lines[0], dtype=int)
    data0 = the_data[idx0, :window_size, :1]
    if spatial_demean:
        data0 = data0 - data0.mean(axis=0, keepdims=True)
    if chn_dist_lines is None:
        fw0, _bw0, fft0, ff, fy0 = batch_func(
            data0, sfreq,
            freq_lo=freq_lo, freq_hi=freq_hi,
            **batch_kwargs,
        )
    else:
        fw0, _bw0, fft0, ff, fy0 = batch_func(
            data0, np.asarray(chn_dist_lines[0], dtype=float), sfreq,
            freq_lo=freq_lo, freq_hi=freq_hi,
            **batch_kwargs,
        )
    n_freqs = fw0.shape[0]

    fwMax = np.empty((n_lines, n_freqs, n_win, n_trials), dtype=float)
    bwMax = np.empty_like(fwMax)
    fwMaxSS = np.empty_like(fwMax)
    
    if mode == "mean":
        bwMaxSS = fwMaxSS
        FFT_f = fwMaxSS
    else:
        bwMaxSS = np.empty_like(fwMax)
        FFT_f = None
    fy_lines = []

    for wi, st in enumerate(starts):
        seg = slice(st, st + window_size)
        for mm, line in enumerate(electrode_lines):
            idx_line = np.asarray(line, dtype=int)
            dist_line = None if chn_dist_lines is None else np.asarray(chn_dist_lines[mm], dtype=float)

            for t0 in trial_chunks:
                t1 = min(t0 + max_batch, n_trials)
                seg_data = the_data[idx_line, seg, t0:t1]
                if spatial_demean:
                    seg_data = seg_data - seg_data.mean(axis=0, keepdims=True)
                n_chunk = t1 - t0

                if chn_dist_lines is None:
                    fw, bw, fft_f, _ff, fy = batch_func(
                        seg_data, sfreq,
                        freq_lo=freq_lo, freq_hi=freq_hi,
                        **batch_kwargs,
                    )
                else:
                    fw, bw, fft_f, _ff, fy = batch_func(
                        seg_data, dist_line, sfreq,
                        freq_lo=freq_lo, freq_hi=freq_hi,
                        **batch_kwargs,
                    )

                fwMax[mm, :, wi, t0:t1] = fw
                bwMax[mm, :, wi, t0:t1] = bw

                if wi == 0 and t0 == 0:
                    fy_lines.append(fy)

                if mode == "mean":
                    fwMaxSS[mm, :, wi, t0:t1] = fft_f
                    continue

                reps_per_chunk = max(1, max_batch // n_chunk)
                fw_sum = np.zeros((n_freqs, n_chunk), dtype=float)
                bw_sum = np.zeros((n_freqs, n_chunk), dtype=float)
                reps_done = 0
                while reps_done < shuffle_reps:
                    reps_now = min(reps_per_chunk, shuffle_reps - reps_done)
                    surr_data = _make_surrogate_data(seg_data, reps_now, rng)
                    if chn_dist_lines is None:
                        fw_s, bw_s, _fft_s, _ff, _fy = batch_func(
                            surr_data, sfreq,
                            freq_lo=freq_lo, freq_hi=freq_hi,
                            **batch_kwargs,
                        )
                    else:
                        fw_s, bw_s, _fft_s, _ff, _fy = batch_func(
                            surr_data, dist_line, sfreq,
                            freq_lo=freq_lo, freq_hi=freq_hi,
                            **batch_kwargs,
                        )

                    fw_s = fw_s.reshape(n_freqs, reps_now, n_chunk)
                    bw_s = bw_s.reshape(n_freqs, reps_now, n_chunk)
                    fw_sum += fw_s.sum(axis=1)
                    bw_sum += bw_s.sum(axis=1)
                    reps_done += reps_now

                fwMaxSS[mm, :, wi, t0:t1] = fw_sum / shuffle_reps
                bwMaxSS[mm, :, wi, t0:t1] = bw_sum / shuffle_reps

    return {
        "fwMax": fwMax,
        "bwMax": bwMax,
        "fwMaxSS": fwMaxSS,
        "bwMaxSS": bwMaxSS,
        "FFTf": FFT_f,
        "starts": starts,
        "ff": ff,
        "fy": fy_lines,
        "baseline_mode": mode,
        "max_batch": max_batch,
        "spatial_demean": bool(spatial_demean),
    }


def _format_result(result, return_dict):
    """Return the result as the full dict, or as the 7-tuple documented in the
    module-level "Outputs" section (fwMax, bwMax, fwMaxSS, bwMaxSS, FFTf,
    starts, ff)."""
    if return_dict:
        return result
    return (
        result["fwMax"],
        result["bwMax"],
        result["fwMaxSS"],
        result["bwMaxSS"],
        result["FFTf"],
        result["starts"],
        result["ff"],
    )


def wave_trigger_computer_nonuniform(the_data,
                                     electrode_lines,
                                     chn_dist_lines,
                                     sfreq=500,
                                     window_size=250,
                                     step=10,
                                     shuffle_reps=10,
                                     freq_lo=1.0,
                                     freq_hi=40.0,
                                     baseline_mode="surr",
                                     max_batch=1000,
                                     rng=None,
                                     eps=1e-9,
                                     spatial_demean=False,
                                     return_dict=False):
    """Sliding-window TW power using regular 2D NUFFT Type-1."""
    result = _wave_trigger_template(
        the_data, electrode_lines, _nufft2d_batch,
        chn_dist_lines=chn_dist_lines,
        sfreq=sfreq,
        window_size=window_size,
        step=step,
        shuffle_reps=shuffle_reps,
        freq_lo=freq_lo,
        freq_hi=freq_hi,
        baseline_mode=baseline_mode,
        max_batch=max_batch,
        spatial_demean=spatial_demean,
        rng=rng,
        eps=eps,
    )
    return _format_result(result, return_dict)


def wave_trigger_computer_uniform(the_data,
                                  electrode_lines,
                                  sfreq=500,
                                  window_size=250,
                                  step=10,
                                  shuffle_reps=10,
                                  freq_lo=1.0,
                                  freq_hi=40.0,
                                  baseline_mode="surr",
                                  max_batch=1000,
                                  rng=None,
                                  spatial_demean=False,
                                  return_dict=False):
    """
    Sliding-window TW power using standard 2D FFT for equally spaced lines.
    """
    result = _wave_trigger_template(
        the_data, electrode_lines, _uniform_fft2_batch,
        chn_dist_lines=None,
        sfreq=sfreq,
        window_size=window_size,
        step=step,
        shuffle_reps=shuffle_reps,
        freq_lo=freq_lo,
        freq_hi=freq_hi,
        baseline_mode=baseline_mode,
        max_batch=max_batch,
        spatial_demean=spatial_demean,
        rng=rng,
    )
    return _format_result(result, return_dict)


def _elec_distance(pos, xo=0.0, yo=0.0, zo=0.0):
    """
    Compute pairwise spherical arc lengths between electrodes.

    Parameters
    ----------
    pos : array-like, shape (n_elec, 3)
        Electrode Cartesian coordinates.
    xo, yo, zo : float
        Sphere center coordinates.

    Returns
    -------
    dist : ndarray, shape (n_elec, n_elec)
        Pairwise arc-length distance matrix.
    """
    pos = np.asarray(pos, dtype=float)
    A = pos - np.array([xo, yo, zo])
    lengths = np.linalg.norm(A, axis=1)

    n = len(lengths)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r     = (lengths[i] + lengths[j]) / 2.0
            denom = lengths[i] * lengths[j]
            if denom == 0.0:
                dist[i, j] = np.linalg.norm(A[i] - A[j])
                continue
            cos_t = np.dot(A[i], A[j]) / denom
            cos_t = np.clip(cos_t, -1.0, 1.0)
            theta = np.arccos(cos_t)
            dist[i, j] = r * theta
    return dist



def build_chn_dist_lines(electrode_lines, pos, xo=0.0, yo=0.0, zo=0.0):
    """
    Build cumulative electrode distances along each electrode line.

    Parameters
    ----------
    electrode_lines : list of lists
        Channel indices for each line.
    pos : array-like, shape (n_total_channels, 3)
        Cartesian coordinate for every channel in data order.

    Returns
    -------
    chn_dist_lines : list of ndarray
        One cumulative distance vector per electrode line.
    """
    pos = np.asarray(pos, dtype=float)
    dist_matrix = _elec_distance(pos, xo, yo, zo)

    chn_dist_lines = []
    for line in electrode_lines:
        idx = np.asarray(line, dtype=int)
        dist_line = np.zeros(len(idx))
        for kk in range(1, len(idx)):
            dist_line[kk] = dist_line[kk - 1] + dist_matrix[idx[kk - 1], idx[kk]]
        chn_dist_lines.append(dist_line)
    return chn_dist_lines
