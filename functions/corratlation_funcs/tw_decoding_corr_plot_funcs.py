"""Figures for fixed-window TW–decoding correlations."""

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

STYLE = {'font.size':12, 'pdf.fonttype':42, 'axes.labelsize':13,
         'axes.labelweight':'bold', 'axes.linewidth':1.2,
         'axes.titlesize':12, 'axes.titleweight':'bold',
         'xtick.labelsize':12, 'ytick.labelsize':12,
         'xtick.major.width':1.2, 'ytick.major.width':1.2}

COLORS = {'BW':'#2563a6', 'FW':'#d47825'}

DECODING_TITLES = {
    'early': 'Early item decoding',
    'late': 'Late item decoding',
    'late_minus_early': 'Late−early decoding',
    'cue': 'cue item decoding',
    'uncue': 'uncue item decoding',
    'cue_minus_uncue': 'cue-uncue item decoding',
}

def plot_correlation_boxes(subjects, *, panels, out, name,
                           ylabel='Partial correlation (r)', show_r=True):
    """Panels are (target, side, title); FW left, BW right; mean r ± 1 SEM."""
    from .correlation_statistics import ttest_bayes_summary
    results = ttest_bayes_summary(subjects)
    colors = {'FW': '#1f77b4', 'BW': '#ff7f0e'}
    values = np.concatenate([subjects.loc[
        (subjects.target == target) & subjects.measure.isin(
            [f'FW_{side}', f'BW_{side}']), 'r'].to_numpy()
        for target, side, _ in panels])
    low, high = min(values.min(), 0), max(values.max(), 0)
    spread = max(high-low, .02)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.25*len(panels), 4.6),
                                 sharey=True, squeeze=False, layout='constrained')
        rng = np.random.default_rng(42)
        for ax, (target, side, title) in zip(axes.flat, panels):
            for position, direction in enumerate(('FW', 'BW')):
                measure = direction+'_'+side
                table = subjects[(subjects.target == target) & (subjects.measure == measure)].sort_values('subject')
                row = results[(results.target == target) & (results.measure == measure)].iloc[0]
                color = colors[direction]
                ax.boxplot([table.r.to_numpy()], positions=[position], widths=.48,
                    showfliers=False, patch_artist=True, manage_ticks=False,
                    boxprops=dict(facecolor=color, alpha=.16, edgecolor=color, linewidth=1.5),
                    medianprops=dict(color=color, linewidth=1.8),
                    whiskerprops=dict(color=color, linewidth=1.3),
                    capprops=dict(color=color, linewidth=1.3))
                ax.scatter(position+rng.uniform(-.12, .12, len(table)), table.r,
                    color=color, s=30, alpha=.8, edgecolors='none', zorder=3)
                ax.errorbar(position, row.mean_r, yerr=row.sem_r, fmt='o', color='black',
                    capsize=5, markersize=5, elinewidth=1.7, capthick=1.7, zorder=4)
                p_label = '$p$ < .001' if row.p_raw < .001 else f'$p$ = {row.p_raw:.3f}'
                r_label = f'$r$ = {row.r:.3f}\n' if show_r else ''
                ax.text(position, .98, r_label+p_label+f'\n$BF_{{10}}$ = {row.BF10:.3g}',
                    transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=11)
            ax.axhline(0, color='.5', ls=':', lw=1.2)
            ax.set(xticks=[0, 1], xticklabels=['FW', 'BW'], xlim=(-.5, 1.5),
                ylim=(low-.1*spread, high+.55*spread), title=title)
            ax.spines[['top', 'right']].set_visible(False)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
        axes[0, 0].set_ylabel(ylabel)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out/f'{name}.png', dpi=200)
        plt.show()
        plt.close(fig)
    return results

def _save(fig, out, filename):
    fig.savefig(Path(out)/f'{filename}.png', dpi=180)
    plt.show()
    plt.close(fig)

def plot_partial_summary(results, subjects, *, target, out, prefix,
                         ylabel='Partial correlation (r)'):
    values=np.r_[subjects.r, results.ci_low, results.ci_high, 0.]
    low,high=values.min(),values.max()
    spread=max(high-low,.02)
    layouts=[(('contra','ipsi'),('Contra','Ipsi'),(8.5,4.5),f'{prefix}_{target}'),
             (('contra_minus_ipsi',),(DECODING_TITLES[target],),(4.8,4.5),f'{prefix}_lateralization_effect_{target}')]
    with plt.rc_context(STYLE):
        for sides,titles,figsize,filename in layouts:
            fig,axes=plt.subplots(1,len(sides),figsize=figsize,sharey=True,squeeze=False,layout='constrained')
            for side,title,ax in zip(sides,titles,axes.flat):
                for position,direction in enumerate(('BW','FW')):
                    measure=direction+'_'+side
                    table=subjects[subjects.measure==measure].sort_values('subject')
                    row=results[results.measure==measure].iloc[0]
                    ax.scatter(position+np.linspace(-.13,.13,len(table)),table.r,
                               s=25,alpha=.6,color=COLORS[direction],edgecolors='none',zorder=3)
                    ax.errorbar(position,row.r,yerr=[[row.r-row.ci_low],[row.ci_high-row.r]],
                        fmt='o',color='black',capsize=5,markersize=6,elinewidth=1.2,capthick=1.2,zorder=4)
                    ax.text(position,.97,f'r={row.r:.3f}\np={row.p_raw:.4g}\nBF10={row.BF10:.3g}',
                        transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=10)
                ax.axhline(0,color='gray',ls=':',lw=1.2)
                ax.set(xticks=[0,1],xticklabels=['BW','FW'],xlim=(-.4,1.4),
                       ylim=(low-.12*spread,high+.38*spread),title=title)
                ax.spines[['top','right']].set_visible(False)
            axes[0,0].set_ylabel(ylabel)
            _save(fig,out,filename)
