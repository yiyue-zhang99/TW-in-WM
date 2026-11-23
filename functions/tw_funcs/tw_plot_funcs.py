"""Travelling-wave (TW) power plots shared by the Guven, Michael and Mingmin notebooks.

Pipeline:
    average_maps          per condition: one power map per subject and line group
    plot_tf_maps          time-frequency maps with cluster outlines
    plot_band_timeseries  band-power time courses with 95% CIs and cluster bars
"""
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

MEASURES = ('fw', 'bw')
DIRECTION_TITLES = {'fw': 'Forward TW', 'bw': 'Backward TW'}
MEASURE_COLORS = {'fw': '#1f77b4', 'bw': '#ff7f0e'}
# Every TW file has 11 electrode lines in this order: C5 … C1, M, I1 … I5 (indices 0–10).
LINE_GROUPS = {'contra': [0, 1, 2, 3, 4], 'ipsi': [6, 7, 8, 9, 10], 'midline': [5],
               'all_lines': list(range(11))}


# ---------------------------------------------------------------- data

def _read_power(path):
    """Read one session's TW result file: time (s), frequencies (Hz), fw / bw power in dB.
    """
    with open(path, 'rb') as stream:
        data = pickle.load(stream)
    # Guven files keep the TW function's own field names; rename them to the Michael / Mingmin names.
    guven_to_common = {'fwMax': 'fwmax', 'fwMaxSS': 'fwssmax', 'bwMax': 'bwmax', 'bwMaxSS': 'bwssmax',
                       'starts': 'time'}
    data = {guven_to_common.get(name, name): value for name, value in data.items()}

    time = -1.2 + (np.asarray(data['time'])+125)/500
    power = {'fw': 10*np.log10(data['fwmax']/data['fwssmax']),
             'bw': 10*np.log10(data['bwmax']/data['bwssmax'])}
    return time, np.asarray(data['ff']), power


def _session_means(power, trials, flip):
    """Average one session's selected trials, then the lines of each group in LINE_GROUPS.

    Args:
        power: {measure: lines × frequencies × times × trials} of one session.
        trials: the selected trials of that session (rows of the trials table; uses the
            `trial` and flip columns).
        flip: name of the True / False flip column, or None.

    Returns:
        {(measure, group): frequencies × times}.
    """
    means = {}
    for measure, measure_power in power.items():
        selected = measure_power[..., trials['trial'].to_numpy()].copy()
        if flip:
            flipped = trials[flip].to_numpy()
            selected[..., flipped] = selected[::-1, ..., flipped]  # reverse the line axis
        line_means = selected.mean(axis=-1)  # lines × frequencies × times
        for group, lines in LINE_GROUPS.items():
            means[measure, group] = line_means[lines].mean(axis=0)
    return means


def _subject_mean(sessions, key):
    """Mean of a subject's session means; every session counts the same."""
    return sum(means[key] for means in sessions) / len(sessions)


def average_maps(trials, *, split_by=None, flip=None):
    """Average trials into one map per subject, condition and line group.

    A subject's map is the mean of its session means, so every session counts the same. In
    Michael each session has one cue side, so this weights both cue sides equally.

    Args:
        trials: rows of the notebook's trials table (made by read_trials), one per trial to use:

                file                                subject  trial  block  early_left
                .../mingmin_fft/subj01_sess1.pkl          1      6  fixed       False
                .../mingmin_fft/subj01_sess1.pkl          1     13  fixed        True
                .../mingmin_fft/subj01_sess2.pkl          1      3  random      False

            `file` is the session's TW result file, `trial` the trial's index in it.
        split_by: 'rotation' (Guven), 'block' (Mingmin), 'speed' (Michael, Mingmin),
            'hand_target' (Michael), or None for one condition 'all'.
        flip: 'cue_left' (Guven, Michael), 'early_left' / 'late_left' (Michael, Mingmin),
            or None; True trials get the line order reversed.

    Returns:
        {condition: maps}, one key per value of the split_by column ('all' if None), where
        maps = {'subjects': subject ids, 'time': s, 'frequencies': Hz,
                'values': {'fw' / 'bw': {group: subjects × frequencies × times, dB}}}.
    """
    trials = trials.assign(condition=trials[split_by] if split_by else 'all')
    session_maps = defaultdict(lambda: defaultdict(list))
    for file, file_trials in trials.groupby('file', sort=False):
        time, frequencies, power = _read_power(file)
        subject = file_trials['subject'].iloc[0]  # one file = one session of one subject
        for condition, condition_trials in file_trials.groupby('condition', sort=False):
            session_maps[condition][subject].append(_session_means(power, condition_trials, flip))

    result = {}
    for condition in sorted(session_maps):
        subjects = sorted(session_maps[condition])
        values = {}
        for measure in MEASURES:
            values[measure] = {}
            for group in LINE_GROUPS:
                subject_maps = [_subject_mean(session_maps[condition][s], (measure, group))
                                for s in subjects]
                values[measure][group] = np.stack(subject_maps)  # subjects × frequencies × times
            values[measure]['difference'] = values[measure]['contra'] - values[measure]['ipsi']
        result[condition] = dict(subjects=np.array(subjects), time=time, frequencies=frequencies,
                                 values=values)
    return result


# ---------------------------------------------------------------- statistics

def cluster_test(values, *, n_perm, seed, alpha, correction=1):
    """Test subject values against zero with a two-sided cluster permutation test.

    Args:
        values: subjects × times, or subjects × frequencies × times.
        n_perm: number of permutations.
        seed: random seed of the permutations.
        alpha: significance level; also sets the t threshold that forms clusters.
        correction: Bonferroni correction factor (number of tests); 1: no correction.

    Returns:
        (mask, details): mask has the shape of one subject's values and is True inside
        significant clusters; details is [{'indices', 'p'}] for every cluster.
    """
    threshold = stats.t.ppf(1-alpha/2, len(values)-1)  # t value that forms clusters
    _, clusters, p_values, _ = permutation_cluster_1samp_test(
        values, threshold=threshold, n_permutations=n_perm, tail=0,
        n_jobs=1, seed=seed, out_type='indices', verbose=False)
    mask = np.zeros(values.shape[1:], dtype=bool)
    details = []
    for indices, p in zip(clusters, p_values):
        if p*correction < alpha:
            mask[indices] = True
        details.append({'indices': indices, 'p': float(p)})
    return mask, details


def print_clusters(details, time, *, time_zero, alpha, frequencies=None, correction=1):
    """Print the time / frequency range and p of each significant cluster, or of the lowest-p one.

    Args:
        details: cluster details from cluster_test.
        time: times (s) of the tested values.
        time_zero: time (s) printed as 0 ms.
        alpha: significance level.
        frequencies: frequencies (Hz) of the tested values; None for time courses.
        correction: Bonferroni correction factor; if > 1, corrected p-values are printed too.
    """
    selected = [item for item in details if item['p']*correction < alpha]
    if not selected:
        if not details:
            print('No clusters found; no cluster time window or p-value to report.')
            return
        selected = [min(details, key=lambda item: item['p'])]
        print('No significant clusters. Minimum-p cluster (not significant):')
    for item in selected:
        indices = item['indices']
        times = (time[indices[-1]]-time_zero)*1000
        text = f'time=[{times.min():.1f}, {times.max():.1f}] ms'
        if frequencies is not None:
            ff = frequencies[indices[0]]
            text += f', freq=[{ff.min():.1f}, {ff.max():.1f}] Hz'
        text += f", p={item['p']:.4g}"
        if correction > 1:
            text += f", Bonferroni p={min(item['p']*correction, 1.):.4g}"
        print(text)


def _mean_ci95(values):
    """Mean over subjects (axis 0) and the half-width of its 95% t confidence interval."""
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1)/np.sqrt(len(values))
    return mean, stats.t.ppf(.975, len(values)-1)*sem


# ---------------------------------------------------------------- drawing helpers

def _style_axis(ax, *, time_zero):
    """Show the time axis in ms relative to time_zero and apply the shared axis style."""
    start, stop = ax.get_xlim()
    low, high = (start-time_zero)*1000, (stop-time_zero)*1000
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(low, high)
    ticks = ticks[(ticks >= low) & (ticks <= high)]
    ax.set_xticks(time_zero+ticks/1000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{(value-time_zero)*1000:g}'))
    ax.tick_params(labelsize=12, width=1.2)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.yaxis.label.set(fontsize=13, fontweight='bold')


def _draw_events(ax, *, events, t_start, t_stop, rt_lines, labels, label_y=-.22, fontsize=10):
    """Draw the task events and RT lines on one axis and set its time window (y-range unchanged).

    Args:
        ax: the axis.
        events: [(onset s, duration s, label)]: a dashed line at the onset, a gray bar below
            the axis for the duration, and the label below the axis if `labels`.
        t_start, t_stop: time window (s); events outside it are skipped.
        rt_lines: [(time s, color)]: colored dashed lines, e.g. mean RTs.
        labels: True to write the event labels (only the bottom row of a figure needs them).
        label_y: height of the labels in axis units (negative: below the axis).
        fontsize: font size of the labels.
    """
    for onset, _, label in events:
        if not t_start <= onset <= t_stop:
            continue
        ax.axvline(onset, color='.3', ls='--', lw=.9, alpha=.45)
        if labels:
            at_edge = np.isclose(onset, t_start) or np.isclose(onset, t_stop)
            ax.text(onset, label_y, label, transform=ax.get_xaxis_transform(),
                    ha='left' if at_edge else 'center', va='top',
                    fontsize=fontsize, fontweight='bold', clip_on=False)
    bars = []  
    for onset, duration, _ in events:
        left, right = max(onset, t_start), min(onset+duration, t_stop)
        if right > left:
            bars.append((left, right-left))
    y_limits = ax.get_ylim()
    ax.broken_barh(bars, (-.05, .05), facecolors='gray', alpha=.4,
                   transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylim(y_limits)  
    for index, (onset, color) in enumerate(rt_lines):
        if t_start <= onset <= t_stop:
            ax.axvline(onset, color=color, ls=(3*(index % 2), (3, 3)),
                       lw=2, zorder=5, dash_capstyle='butt')
    ax.set_xlim(t_start, t_stop)


def _sig_lane(ax, mask, time, *, lane, color):
    """Draw a bar above the axis where mask is True; each test (lane) gets its own height."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    for start, stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        ax.plot([time[start], time[stop-1]], [1.035+.07*lane]*2,
                color=color, lw=3, transform=ax.get_xaxis_transform(), clip_on=False)


def _colorbar_extend(values, limit):
    """Which colorbar ends get an arrow because values lie beyond ±limit."""
    below, above = np.any(values < -limit), np.any(values > limit)
    return 'both' if below and above else 'min' if below else 'max' if above else 'neither'


def _save(fig, output_dir, name, *, t_start, t_stop, time_zero):
    """Save the figure as output_dir/<name>_<time window>.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window = f'{t_start:g}-{t_stop:g}s_zero-{time_zero:g}s'.replace('.', 'p')
    path = output_dir/f'{name}_{window}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'[saved] {path}')


# ---------------------------------------------------------------- figures

def plot_tf_maps(maps, *, color_limit, t_start, t_stop, time_zero, events, output_dir, name,
                 n_perm, seed, alpha, rt_lines=()):
    """Plot and save contra − ipsi TF maps up to 30 Hz (fw left, bw right); print the clusters.

    Args:
        maps: {'subjects': subject ids, 'time': s, 'frequencies': Hz,
                'values': {'fw' / 'bw': {group: subjects × frequencies × times, dB}}}.
        color_limit: color bar.
        t_start, t_stop: time window to plot and test (s).
        time_zero: time (s) shown as 0 ms on the time axis.
        events: [(onset s, duration s, label)] task events.
        output_dir: folder for the figure.
        name: <name>_fw_bw_tf_difference_<time window>.png.
        n_perm, seed, alpha: cluster test settings.
        rt_lines: [(time s, color)] colored dashed lines.
    """
    in_window = (maps['time'] >= t_start) & (maps['time'] <= t_stop)
    in_range = maps['frequencies'] <= 30  # Hz
    time, frequencies = maps['time'][in_window], maps['frequencies'][in_range]
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(12.8, 4.8))
    for ax, measure in zip(axes, MEASURES):
        # subjects × frequencies × times, cut to ≤ 30 Hz and the time window
        values = maps['values'][measure]['difference'][:, in_range][:, :, in_window]
        mean = values.mean(axis=0)  # average over subjects (axis 0): frequencies × times
        image = ax.pcolormesh(time, frequencies, mean, shading='auto', cmap='coolwarm',
                              vmin=-color_limit, vmax=color_limit)
        colorbar = fig.colorbar(image, ax=ax, pad=.025, fraction=.045, aspect=22,
                                extend=_colorbar_extend(mean, color_limit))
        colorbar.set_label('Mean power (dB)', fontsize=10)
        colorbar.ax.tick_params(labelsize=9)

        significant, details = cluster_test(values, n_perm=n_perm, seed=seed, alpha=alpha)
        if significant.any():  # contour() warns when there is nothing to outline
            ax.contour(time, frequencies, significant.astype(float), levels=[.5],
                       colors='black', linewidths=1.5)
        print(f'[{name} | {measure} | difference]')
        print_clusters(details, time, time_zero=time_zero, alpha=alpha, frequencies=frequencies)

        ax.set_title(DIRECTION_TITLES[measure], fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Frequency (Hz)')
        _draw_events(ax, events=events, t_start=t_start, t_stop=t_stop, rt_lines=rt_lines,
                     labels=True, label_y=-.14)
        _style_axis(ax, time_zero=time_zero)
    width, height = fig.get_size_inches()
    fig.subplots_adjust(left=.7/width, right=1-.75/width, bottom=.8/height, top=1-.45/height,
                        wspace=.45)
    _save(fig, output_dir, f'{name}_fw_bw_tf_difference', t_start=t_start, t_stop=t_stop,
          time_zero=time_zero)
    plt.show()


def plot_band_timeseries(conditions, *, group, freq_bands, ylim, t_start, t_stop, time_zero,
                         events, output_dir, name, n_perm, seed, alpha, tests=None, bonferroni=False,
                         layout='rows', legend_title=None, colors=None,
                         panel_size=None, event_fontsize=10, rt_lines=()):
    """Plot and save band-power time courses (mean ± 95% CI) with cluster bars; print the clusters.

    Args:
        conditions: {condition: maps} from average_maps; one curve per
            condition. For a single condition pass {'all': maps}.
        group: 'difference' (contra − ipsi), 'midline' or 'all_lines' .
        freq_bands: FREQ_BANDS or some of its bands.
        ylim: (bottom, top) of the power axis, in dB.
        t_start, t_stop: time window to plot and test (s).
        time_zero: time (s) shown as 0 ms on the time axis.
        events: [(onset s, duration s, label)] task events.
        output_dir: output folder for the figure.
        name:  <name>_fw_bw_bands_<time window>.png.
        n_perm, seed, alpha: cluster test settings.
        tests: (condition, None): condition vs 0; (condition, condition): difference vs 0;
            None: every condition vs 0.
        bonferroni: True : bonferroni correction; False: no correction.
        layout: 'rows': fw / bw in rows, bands in columns; 'columns': the other way round.
        legend_title: title of the condition legend, or None.
        colors: {condition: curve color, (condition, condition): difference bar color}.
        panel_size: (width, height) of one panel in inches; None: chosen from layout.
        event_fontsize: font size of the event labels.
        rt_lines: [(time s, color)] colored dashed lines, e.g. mean RTs.
    """
    if tests is None:
        tests = [(condition, None) for condition in conditions]
    correction = len(tests) if bonferroni else 1
    single = len(conditions) == 1

    directions_in_columns = layout == 'columns'
    nrows, ncols = ((len(freq_bands), len(MEASURES)) if directions_in_columns
                    else (len(MEASURES), len(freq_bands)))
    width, height = panel_size or ((6.4, 4.8) if directions_in_columns else (4.8, 4.2))
    fig, grid = plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows),
                             squeeze=False, sharex=True)
    bottom_row = list(grid[-1])  # only these panels get event labels
    axes = grid.T if directions_in_columns else grid  # axes[direction_index, band_index]

    first_maps = list(conditions.values())[0]  # every condition has the same time and frequencies
    in_window = (first_maps['time'] >= t_start) & (first_maps['time'] <= t_stop)
    time = first_maps['time'][in_window]
    for direction_index, measure in enumerate(MEASURES):
        for band_index, (low, high, band_name) in enumerate(freq_bands):
            band = f'{band_name} ({low}–{high} Hz)'
            ax = axes[direction_index, band_index]
            in_band = (first_maps['frequencies'] >= low) & (first_maps['frequencies'] <= high)
            band_power = {}
            for condition, maps in conditions.items():
                # subjects × frequencies × times → subjects × times 
                values = maps['values'][measure][group][:, in_band][:, :, in_window].mean(axis=1)
                band_power[condition] = values
                mean, half_width = _mean_ci95(values)  # values: subjects × times
                color = MEASURE_COLORS[measure] if single else colors[condition]
                ax.plot(time, mean, color=color, lw=3)
                ax.fill_between(time, mean-half_width, mean+half_width, color=color,
                                alpha=.32, lw=0, zorder=1)

            for test_index, (first, second) in enumerate(tests):
                values = band_power[first] if second is None else band_power[first]-band_power[second]
                significant, details = cluster_test(values, n_perm=n_perm, seed=seed,
                                                    alpha=alpha, correction=correction)
                label = str(first) if second is None else f'{first} − {second}'
                print(f'[{name} | {measure} | {band} | {label}]')
                print_clusters(details, time, time_zero=time_zero, alpha=alpha,
                               correction=correction)
                if second is None:
                    color = MEASURE_COLORS[measure] if single else colors[first]
                else:
                    color = colors.get((first, second), 'black')
                _sig_lane(ax, significant, time, lane=test_index, color=color)

            ax.axhline(0, color='.5', ls='--', lw=.8)
            ax.set_ylim(*ylim)
            ax.set_ylabel('Power (dB)')
            title = f'{DIRECTION_TITLES[measure]}\n{band}' if directions_in_columns and band_index == 0 else band
            ax.set_title(title, pad=25+10*(len(tests)-1), fontsize=11, fontweight='bold')
            _draw_events(ax, events=events, t_start=t_start, t_stop=t_stop, rt_lines=rt_lines,
                         labels=ax in bottom_row, fontsize=event_fontsize)
    for ax in axes.flat:
        _style_axis(ax, time_zero=time_zero)

    # Space (in inches) above the panels for legend and direction titles.
    header = 1.0 if directions_in_columns and single else 1.55
    fig_width, fig_height = fig.get_size_inches()
    fig.subplots_adjust(bottom=.8/fig_height, top=1-header/fig_height, right=.9,
                        hspace=1.1, wspace=.3)

    if not single:
        # One legend entry per condition (curve color), then one per comparison (bar color).
        labels = [str(condition) for condition in conditions]
        line_colors = [colors[condition] for condition in conditions]
        comparisons = [(first, second) for first, second in tests if second is not None]
        for first, second in comparisons:
            labels.append('Difference' if len(comparisons) == 1 else f'{first} vs {second}')
            line_colors.append(colors.get((first, second), 'black'))
        handles = [Line2D([], [], color=color, lw=3) for color in line_colors]
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1-.25/fig_height),
                   ncol=len(labels), frameon=False, fontsize=9,
                   title=legend_title, title_fontsize=9)

    if not directions_in_columns:
        # Direction titles above each row; the first sits at the top of the figure.
        for direction_index, measure in enumerate(MEASURES):
            y = .98 if direction_index == 0 else (axes[direction_index, 0].get_position().y1
                                                  + (55+10*(len(tests)-1))/72/fig_height)
            fig.text(.5, y, DIRECTION_TITLES[measure], ha='center', va='bottom',
                     fontsize=13, fontweight='bold')

    _save(fig, output_dir, f'{name}_fw_bw_bands', t_start=t_start, t_stop=t_stop,
          time_zero=time_zero)
    plt.show()
