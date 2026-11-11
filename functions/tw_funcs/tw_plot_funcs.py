"""Shared TW plots: input times in seconds, displayed times in milliseconds."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

DIRECTION_TITLES = {'fw': 'Forward TW', 'bw': 'Backward TW'}

def normalize_power(fw, fw_surrogate, bw, bw_surrogate):
    arrays = {'fw': 10 * np.log10(fw / fw_surrogate), 'bw': 10 * np.log10(bw / bw_surrogate)}
    return arrays


def collect_maps(sessions, *, groups, align, weighting='session'):
    """Average nonempty trial selections within subjects.

    weighting='session' weights sessions equally; 'trial' pools trials.
    """
    sums, weights, counts, response_times = {}, {}, {}, {}
    for session in sessions:
        time, frequencies = session['time'], session['frequencies']
        for condition, keep in session['selections'].items():
            n = int(keep.sum())
            # A session may contain only one experimental condition.
            # An absent condition contributes neither a mean nor a session weight.
            if n == 0:
                continue
            key = (condition, session['subject'])
            weight = 1 if weighting == 'session' else n
            weights[key] = weights.get(key, 0) + weight
            counts[key] = counts.get(key, 0) + n
            for name, rt in session['rt'].items():
                response_times.setdefault((condition, session['subject'], name), []).append(rt[keep])
            for measure, power in session['power'].items():
                selected = power[..., keep].copy()
                if align:
                    flip = session['flip'][keep]
                    selected[..., flip] = selected[::-1, ..., flip]
                line_means = selected.mean(axis=-1)
                for group, indices in groups.items():
                    value = line_means[indices].mean(axis=0)
                    entry = (condition, session['subject'], measure, group)
                    sums[entry] = sums.get(entry, 0) + weight*value
    if not weights:
        raise ValueError('No valid trials remain in any selected condition.')
    conditions = sorted({key[0] for key in weights}, key=str)
    measures = sorted({key[2] for key in sums})
    result = {}
    for condition in conditions:
        subjects = np.array(sorted(subject for c, subject in weights if c == condition))
        values = {measure: {group: np.stack([
            sums[condition, subject, measure, group]/weights[condition, subject]
            for subject in subjects]) for group in groups} for measure in measures}
        if 'contra' in groups and 'ipsi' in groups:
            for measure in measures:
                values[measure]['difference'] = values[measure]['contra']-values[measure]['ipsi']
        names = {key[2] for key in response_times if key[0] == condition}
        rt = {name: {subject: np.concatenate(response_times[condition, subject, name])
                     for subject in subjects} for name in names}
        result[condition] = dict(subjects=subjects, time=time, frequencies=frequencies, values=values, counts={subject: counts[condition, subject] for subject in subjects}, rt=rt)
    return result


def cluster_test(values, *, alpha=.05, n_perm=5000, seed=42):
    """Two-sided one-sample cluster test on subjects × [frequencies ×] times."""
    threshold = stats.t.ppf(1-alpha/2, len(values)-1)
    _, clusters, p_values, _ = permutation_cluster_1samp_test(
        values, threshold=threshold, n_permutations=n_perm, tail=0,
        n_jobs=1, seed=seed, out_type='indices', verbose=False)
    mask = np.zeros(values.shape[1:], dtype=bool)
    details = []
    for indices, p in zip(clusters, p_values):
        if p < alpha:
            mask[indices] = True
        details.append({'indices': indices, 'p': float(p)})
    return mask, details


def print_clusters(details, time, *, time_zero, frequencies=None, alpha=.05, correction=1):
    """Print significant clusters, or the minimum-p cluster labeled non-significant."""
    selected = [item for item in details if min(item['p']*correction, 1.) < alpha]
    if not selected:
        if not details:
            print('No clusters found; no cluster time window or p-value to report.')
            return
        selected = [min(details, key=lambda item: item['p'])]
        print('No significant clusters. Minimum-p cluster (not significant):')
    for item in selected:
        corrected_p = min(item['p']*correction, 1.)
        indices = item['indices']
        times = (time[indices[-1]]-time_zero)*1000
        text = f'time=[{times.min():.1f}, {times.max():.1f}] ms'
        if frequencies is not None:
            ff = frequencies[indices[0]]
            text += f', freq=[{ff.min():.1f}, {ff.max():.1f}] Hz'
        text += f", p={item['p']:.4g}"
        if correction > 1:
            text += f', Bonferroni p={corrected_p:.4g}'
        print(text)


def _style(fig, axes, *, time_zero):
    for ax in np.asarray(axes).flat:
        start, stop = ax.get_xlim()
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(
            (start-time_zero)*1000, (stop-time_zero)*1000)
        ticks = ticks[(ticks >= (start-time_zero)*1000) & (ticks <= (stop-time_zero)*1000)]
        ax.set_xticks(time_zero+ticks/1000)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{(value-time_zero)*1000:g}'))
        ax.tick_params(labelsize=12, width=1.2)
        ax.spines[['top', 'right']].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.title.set(fontsize=11, fontweight='bold')
        ax.xaxis.label.set(fontsize=13, fontweight='bold')
        ax.yaxis.label.set(fontsize=13, fontweight='bold')
    if fig._suptitle is not None:
        fig._suptitle.set(fontsize=13, fontweight='bold')


def _events(ax, *, events, stimulus_bars, t_start, t_stop, labels, hide_right_label, rt_lines,
            label_rows=1, label_y=-.22, label_fontsize=10):
    label_index = 0
    for onset, label in events:
        if t_start <= onset <= t_stop:
            ax.axvline(onset, color='.3', ls='--', lw=.9, alpha=.45)
            if labels and not (hide_right_label and np.isclose(onset, t_stop)):
                alignment = 'left' if np.isclose(onset, t_start) or np.isclose(onset, t_stop) else 'center'
                ax.annotate(label, (onset, label_y), xycoords=ax.get_xaxis_transform(),
                        xytext=(0, -16*(label_index % label_rows)), textcoords='offset points',
                        ha=alignment, va='top', rotation=0, fontsize=label_fontsize,
                        fontweight='bold', annotation_clip=False)
                label_index += 1
    bars = [(max(start, t_start), min(start+width, t_stop)-max(start, t_start))
            for start, width in stimulus_bars if min(start+width, t_stop)>max(start, t_start)]
    ax.broken_barh(bars, (-.05, .05), facecolors='gray', alpha=.4,
                   transform=ax.get_xaxis_transform(), clip_on=False)
    for index, (onset, color, label) in enumerate(rt_lines):
        if t_start <= onset <= t_stop:
            # Alternate dash positions so coincident RT lines remain visible.
            ax.axvline(onset, color=color, ls=(3*(index % 2), (3, 3)),
                       lw=2, label='_nolegend_', zorder=5, dash_capstyle='butt')
    ax.set_xlim(t_start, t_stop)


def _save(fig, output_dir, name, *, t_start, t_stop, time_zero, output_format='png'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{t_start:g}-{t_stop:g}s_zero-{time_zero:g}s'.replace('.', 'p')
    path = output_dir/f'{name}_{tag}.{output_format}'
    if output_format == 'svg':
        # Matplotlib rasterizes colorbar meshes by default; keep them vector too.
        for artist in fig.findobj():
            if artist.get_rasterized():
                artist.set_rasterized(False)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'[saved] {path}')
    return path


def _mean_ci95(values):
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1)/np.sqrt(len(values))
    width = stats.t.ppf(.975, len(values)-1)*sem
    return mean, width


def _sig_lane(ax, mask, time, *, lane, color):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    for start, stop in zip(np.where(edges==1)[0], np.where(edges==-1)[0]):
        ax.plot([time[start], time[stop-1]], [1.035+.07*lane]*2,
                color=color, lw=3, transform=ax.get_xaxis_transform(), clip_on=False)


def plot_tf_maps(data, *, measures, t_start, t_stop, time_zero, events,
                 stimulus_bars, output_dir, name, fmax=30,
                 limits, diff_limits, cmap='coolwarm',
                 combine_measures=False, panel_size=(6.4, 4.8), show_sides=True,
                 rt_lines=(), run_cluster=True, n_perm=5000, seed=42, alpha=.05,
                 output_format='png'):
    """TF maps: groups in rows, combined directions in columns.

    show_sides=False plots/tests only contra-minus-ipsi.
    panel_size is (width, height) in inches; combined result key is 'fw_bw'.
    """
    time_mask = (data['time']>=t_start)&(data['time']<=t_stop)
    frequency_mask = data['frequencies']<=fmax
    time, frequencies = data['time'][time_mask], data['frequencies'][frequency_mask]
    results = {}
    figures = [list(measures)] if combine_measures else [[measure] for measure in measures]
    figure_groups = [(('difference',), 'difference')]
    if show_sides:
        figure_groups.insert(0, (('contra', 'ipsi', 'mid'), 'sides'))
    for figure_measures in figures:
        tag = '_'.join(figure_measures)
        results[tag] = {}
        for groups, suffix in figure_groups:
            nrows, ncols = len(groups), len(figure_measures)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(panel_size[0]*ncols, panel_size[1]*nrows),
                                     sharex=True, sharey=True, squeeze=False)
            details_by_measure = {measure: {} for measure in figure_measures}
            panels = [(row, col, group, measure) for row, group in enumerate(groups)
                      for col, measure in enumerate(figure_measures)]
            for row, col, group, measure in panels:
                ax = axes[row,col]
                values = data['values'][measure][group][:, frequency_mask][:, :, time_mask]
                scale = diff_limits[measure] if group=='difference' else limits[measure]
                mean = values.mean(axis=0)
                image = ax.pcolormesh(time, frequencies, mean,
                    shading='auto', cmap=cmap, vmin=-scale, vmax=scale)
                below, above = np.any(mean < -scale), np.any(mean > scale)
                extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
                colorbar = fig.colorbar(image, ax=ax, pad=.025, fraction=.045,
                                       aspect=22, extend=extend)
                colorbar.set_label('Mean power (dB)', fontsize=10)
                colorbar.ax.tick_params(labelsize=9)
                details = []
                if run_cluster:
                    sig, details = cluster_test(values, n_perm=n_perm, seed=seed, alpha=alpha)
                    if sig.any() and not sig.all():
                        ax.contour(time,frequencies,sig.astype(float),levels=[.5],colors='black',linewidths=1.5)
                    print(f'[{name} | {measure} | {group}]')
                    print_clusters(details,time,time_zero=time_zero,frequencies=frequencies,alpha=alpha)
                details_by_measure[measure][group] = details
                title = {'contra':'Contra', 'ipsi':'Ipsi', 'mid':'Midline', 'difference':''}[group]
                title = DIRECTION_TITLES[measure] + (f' · {title}' if title else '')
                ax.set(title=title, ylabel='Frequency (Hz)')
                frequency_limits = ax.get_ylim()
                _events(ax,events=events,stimulus_bars=stimulus_bars,t_start=t_start,t_stop=t_stop,
                        labels=row==nrows-1, hide_right_label=False, rt_lines=rt_lines,
                        label_y=-.14)
                ax.set_ylim(frequency_limits)
                ax.tick_params(labelleft=True)
            _style(fig,axes,time_zero=time_zero)
            for ax in axes.flat:
                ax.set_title(ax.get_title(), fontsize=13, fontweight='bold', pad=12)
            fig.subplots_adjust(left=.7/fig.get_figwidth(), right=1-.75/fig.get_figwidth(),
                                bottom=.8/fig.get_figheight(), top=1-.45/fig.get_figheight(),
                                hspace=.28, wspace=.45)
            path = _save(fig,output_dir,f'{name}_{tag}_tf_{suffix}',t_start=t_start,t_stop=t_stop,time_zero=time_zero,output_format=output_format)
            details = (details_by_measure[figure_measures[0]] if len(figure_measures)==1
                       else details_by_measure)
            results[tag][suffix] = dict(figure=fig,axes=axes,path=path,clusters=details)
            plt.show()
    return results


def plot_band_timeseries(data, *, group, measures, freq_bands, t_start,
                         t_stop, time_zero, events, stimulus_bars, output_dir,
                         name, ylims, tests=None, split_measures=True,
                         correction=1, colors=None, legend_title=None,
                         measure_linestyles=None, bands_as_rows=False,
                         combine_measures=False, measure_layout='vertical', line_width=3,
                         panel_size=None,
                         legend_layout='top', comparison_colors=None,
                         rt_lines=(), hide_right_event_label=False, event_label_rows=1,
                         event_label_fontsize=10,
                         run_cluster=True, n_perm=5000, seed=42, alpha=.05,
                         output_format='png'):
    """Band time courses with pointwise 95% CIs; paired inputs must share subject and time ordering.

    tests: (condition, None) against zero, or two paired conditions.
    combine_measures uses measure_layout='vertical' or 'horizontal'.
    panel_size is in inches; correction applies across the requested tests.
    """
    if combine_measures:
        split_measures = False
        bands_as_rows = {'vertical': False, 'horizontal': True}[measure_layout]
    direction_columns = combine_measures and measure_layout == 'horizontal'
    conditions = {'all': data} if 'subjects' in data else data
    if tests is None:
        tests = [(condition, None) for condition in conditions]
    for first, second in tests:
        if first not in conditions or (second is not None and second not in conditions):
            raise ValueError(f'No valid data for requested comparison: {first}, {second}')
        if second is not None:
            for key in ('subjects', 'time', 'frequencies'):
                if not np.array_equal(conditions[first][key], conditions[second][key]):
                    raise ValueError(f'Paired conditions must have matching {key}: {first}, {second}')
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    condition_colors = {key: palette[i%len(palette)] for i,key in enumerate(conditions)}
    if colors is not None:
        condition_colors.update(colors)
    measure_colors = {'fw': '#1f77b4', 'bw': '#ff7f0e'}
    figures = [[measure] for measure in measures] if split_measures else [list(measures)]
    result = {}
    for figure_measures in figures:
        shape = (len(freq_bands),len(figure_measures)) if bands_as_rows else (len(figure_measures),len(freq_bands))
        side_legend = legend_layout == 'right'
        height = 4.8 if side_legend or direction_columns else 4.2
        width = 6.4 if direction_columns else 4.8
        if panel_size is not None:
            width, height = panel_size
        fig,axes=plt.subplots(*shape,figsize=(width*shape[1]+(2.2 if side_legend else 0),height*shape[0]),squeeze=False,sharex=True)
        all_stats={}
        comparison_handles = {}
        for mi,measure in enumerate(figure_measures):
            for bi,(lo,hi,band_name) in enumerate(freq_bands):
                ax=axes[bi,mi] if bands_as_rows else axes[mi,bi]
                band_data={}
                times={}
                for condition,data in conditions.items():
                    tm=(data['time']>=t_start)&(data['time']<=t_stop)
                    fm=(data['frequencies']>=lo)&(data['frequencies']<=hi)
                    values=data['values'][measure][group][:,fm][:,:,tm].mean(axis=1)
                    time=data['time'][tm]
                    times[condition]=time
                    band_data[condition]=values
                    mean,width=_mean_ci95(values)
                    color=measure_colors[measure] if len(conditions)==1 else condition_colors[condition]
                    linestyle='-' if measure_linestyles is None else measure_linestyles[measure]
                    ax.plot(time,mean,color=color,ls=linestyle,lw=line_width,
                            label=str(condition) if len(conditions)>1 else '_nolegend_')
                    ax.fill_between(time,mean-width,mean+width,color=color,alpha=.32,lw=0, zorder=1)
                panel_stats=[]
                for ti,(first,second) in enumerate(tests):
                    if not run_cluster:
                        break
                    data=conditions[first]
                    values=band_data[first]
                    subject_ids=data['subjects']
                    if second is not None:
                        values=values-band_data[second]
                    _,details=cluster_test(values,n_perm=n_perm,seed=seed+1000*measures.index(measure)+100*bi+ti,alpha=alpha)
                    label=str(first) if second is None else f'{first} − {second}'
                    print(f'[{name} | {measure} | {band_name} | {label}]')
                    print_clusters(details,times[first],time_zero=time_zero,alpha=alpha,correction=correction)
                    sig=np.zeros(len(times[first]),bool)
                    for item in details:
                        if item['p']*correction<alpha:
                            sig[item['indices'][0]]=True
                    color=measure_colors[measure] if len(conditions)==1 else condition_colors[first]
                    if second is not None:
                        color = 'black' if comparison_colors is None else comparison_colors[ti % len(comparison_colors)]
                        unit = '°' if legend_title == 'Rotation (°)' else ''
                        comparison = f'{first}{unit} vs {second}{unit}'
                        comparison_handles.setdefault(comparison, Line2D(
                            [], [], color=color, lw=3, label=comparison))
                    _sig_lane(ax,sig,times[first],lane=ti,color=color)
                    panel_stats.append(dict(test=(first,second),subjects=subject_ids,clusters=details,correction=correction))
                all_stats[measure,band_name]=panel_stats
                ax.axhline(0,color='.5',ls='--',lw=.8)
                ax.set_ylim(*ylims[measure])
                ax.set_ylabel('Power (dB)')
                title = f'{DIRECTION_TITLES[measure]}\n{band_name}' if direction_columns and bi==0 else band_name
                ax.set_title(title, pad=25+10*(len(tests)-1))
                _events(ax,events=events,stimulus_bars=stimulus_bars,t_start=t_start,t_stop=t_stop,
                        labels=(bi==shape[0]-1 if bands_as_rows else mi==shape[0]-1),
                        hide_right_label=hide_right_event_label,rt_lines=rt_lines,
                        label_rows=event_label_rows, label_fontsize=event_label_fontsize)
        if len(figure_measures) == 1 and not direction_columns:
            fig.suptitle(DIRECTION_TITLES[figure_measures[0]])
        _style(fig,axes,time_zero=time_zero)
        header_height = 1.35 if side_legend else (1.55 if len(figure_measures)>1 or legend_title else 1.25)
        if direction_columns and not side_legend:
            header_height = 1.55 if len(conditions)>1 or legend_title else 1.0
        fig.subplots_adjust(bottom=(.8+(event_label_rows-1)*16/72)/(height*shape[0]),
                            top=1-header_height/(height*shape[0]),
                            right=1-2.2/fig.get_figwidth() if side_legend else .9,
                            hspace=1.1,wspace=.3)
        if len(conditions)>1:
            legend_entries = {}
            for axis in axes.flat:
                handles, labels = axis.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    legend_entries.setdefault(label, handle)
            if not side_legend and comparison_handles:
                if len(comparison_handles) == 1:
                    legend_entries['Difference'] = next(iter(comparison_handles.values()))
                else:
                    legend_entries.update(comparison_handles)
            if legend_entries:
                fig.legend(legend_entries.values(), legend_entries.keys(),
                           loc='upper left' if side_legend else 'upper center',
                           bbox_to_anchor=(1-2.05/fig.get_figwidth(), .86) if side_legend else (.5, .94),
                           ncol=1 if side_legend else min(len(legend_entries), 4), frameon=False, fontsize=9,
                           title=legend_title, title_fontsize=9)
        if side_legend and comparison_handles:
            fig.legend(handles=list(comparison_handles.values()),
                       loc='upper left', bbox_to_anchor=(1-2.05/fig.get_figwidth(), .58),
                       ncol=1, frameon=False,
                       fontsize=9, title='Difference',
                       title_fontsize=9)
        if len(figure_measures) > 1 and not direction_columns:
            for index, measure in enumerate(figure_measures):
                axis = axes[0,index] if bands_as_rows else axes[index,0]
                x = axis.get_position().x0 + axis.get_position().width/2 if bands_as_rows else .5
                y = axis.get_position().y1 + (55+10*(len(tests)-1))/72/fig.get_figheight()
                if bands_as_rows or index == 0:
                    y = .98
                fig.text(x,y,DIRECTION_TITLES[measure],ha='center',va='bottom',
                         fontsize=13,fontweight='bold')
        tag='_'.join(figure_measures)
        path=_save(fig,output_dir,f'{name}_{tag}_bands',t_start=t_start,t_stop=t_stop,time_zero=time_zero,output_format=output_format)
        result[tag]=dict(figure=fig,axes=axes,path=path,statistics=all_stats)
        plt.show()
    return result
