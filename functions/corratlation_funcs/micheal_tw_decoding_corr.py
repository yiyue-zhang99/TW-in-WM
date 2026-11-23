"""Michael: TW lateralization vs fixed impulse-2 decoding, within-session partial correlations.

Each session's trials are residualized for the session intercept and the original trial order;
session Fisher z values are averaged with equal weight per subject.

"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from functions.corratlation_funcs import correlation_statistics
from functions.corratlation_funcs.partial_correlation import residual

SIDES = ('contra', 'ipsi', 'contra_minus_ipsi')
MEASURES = tuple(d + '_' + side for d in ('BW', 'FW') for side in SIDES)


def extract(dec_dir, tw_dir, tw_window, frequency_band, subjects, sessions):
    """One row per retained trial: decoding scores, RTs and contra / ipsi TW power."""
    dec_dir = Path(dec_dir)
    tw_dir = Path(tw_dir)
    tables = []
    for subject in subjects:
        with np.load(dec_dir / f'subject_{subject:02d}_voltage_torch.npz') as decoding:
            for session in sessions:
                prefix = f'sess{session}_'
                trial_ids = decoding[prefix + 'trial_ids'].astype(int)
                trial_indices = trial_ids - 1
                with (tw_dir / f'subj{subject:02d}_sess{session}.pkl').open('rb') as file:
                    tw = pickle.load(file)
                centers = np.round(-1.2 + (np.asarray(tw['time']) + 125) / 500, 6)
                time_mask = (centers >= tw_window[0]) & (centers <= tw_window[1])
                frequencies = np.asarray(tw['ff'])
                frequency_mask = (frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1])
                behavior = decoding[prefix + 'behavior']
                order = behavior[:, 2].astype(int)
                early_rt = behavior[:, 9]
                late_rt = behavior[:, 13]
                table = pd.DataFrame(dict(
                    subject=subject, session=session, trial=trial_ids,
                    source_row_1based=trial_ids, test_order=order, early_rt=early_rt, late_rt=late_rt,
                ))
                for target in ('late', 'early'):
                    table['D_' + target] = np.asarray(decoding[prefix + 'trial_dec_' + target], dtype=float)
                table['D_late_minus_early'] = table.D_late - table.D_early
                for direction in ('BW', 'FW'):
                    signal = tw[direction.lower() + 'max'][:, frequency_mask][:, :, time_mask][:, :, :, trial_indices]
                    surrogate = tw[direction.lower() + 'ssmax'][:, frequency_mask][:, :, time_mask][:, :, :, trial_indices]
                    normalized = 10 * np.log10(signal / surrogate)
                    left = normalized[:5].mean(axis=(0, 1, 2))
                    right = normalized[6:].mean(axis=(0, 1, 2))
                    # order=1: late is right -> contra=left; order=2: converse.
                    contra = np.where(order == 1, left, right)
                    ipsi = np.where(order == 1, right, left)
                    table[direction + '_contra'] = contra
                    table[direction + '_ipsi'] = ipsi
                    table[direction + '_contra_minus_ipsi'] = contra - ipsi
                    table['L_' + direction] = contra - ipsi
                bad = np.asarray(tw['is_bad_epoch'], bool)[trial_indices]
                keep = (np.isfinite(early_rt) & (early_rt > 0)
                        & np.isfinite(late_rt) & (late_rt > 0) & (late_rt <= 1.5) & ~bad)
                tables.append(table.loc[keep].copy())
    return pd.concat(tables, ignore_index=True)


def residualize_trials(trials, columns):
    """Remove session intercept and original trial order, preserving rejected-trial gaps."""
    result = trials.copy()
    for _, table in trials.groupby(['subject', 'session']):
        trial = table['source_row_1based'].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(trial)), trial - trial.mean()])
        result.loc[table.index, columns] = residual(table[columns].to_numpy(dtype=float), design)
    return result


def residualize(trials):
    """residualize_trials for the TW lateralization, TW measures and decoding columns."""
    return residualize_trials(trials, ['L_BW', 'L_FW', *MEASURES, 'D_late', 'D_early', 'D_late_minus_early'])


def within_session_correlations(residuals, measures, decoding_column):
    """Correlate already residualized trials; average session Fisher z equally per subject."""
    rows = []
    for (subject, session), table in residuals.groupby(['subject', 'session']):
        for measure in measures:
            r = stats.pearsonr(table[measure], table[decoding_column]).statistic
            rows.append(dict(method='trial_residual', measure=measure, subject=subject,
                session=session, n_trials=len(table), r=r,
                fisher_z=np.arctanh(r)))
    sessions = pd.DataFrame(rows)
    subjects = sessions.groupby(['method', 'measure', 'subject']).agg(
        fisher_z=('fisher_z', 'mean'), n_trials=('n_trials', 'sum')).reset_index()
    subjects['r'] = np.tanh(subjects.fisher_z)
    results = [dict(method=method, measure=measure, **correlation_statistics.group_summary(table.fisher_z))
        for (method, measure), table in subjects.groupby(['method', 'measure'])]
    return pd.DataFrame(results), subjects, sessions
