"""Guven: TW lateralization vs fixed-window decoding, trial-wise partial correlations.

Each subject contributes one correlation per TW / decoding combination. Sessions are pooled
without session centering or session regressors; only an intercept and the linear global
trial order are removed from both variables.

"""
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from functions.corratlation_funcs.correlation_statistics import group_summary
from functions.corratlation_funcs.partial_correlation import partial_correlation

TARGETS = ('cue', 'uncue', 'cue_minus_uncue')
SIDES = ('contra', 'ipsi', 'contra_minus_ipsi')
DIRECTIONS = ('BW', 'FW')


def extract(dec_dir, tw_dir, tw_window, frequency_band, subjects, sessions):
    """One row per retained trial: decoding scores and contra / ipsi TW power."""
    matched_tables = []

    for subject in subjects:
        decoding_path = Path(dec_dir) / f'subject_{subject:02d}_voltage_torch.npz'
        with np.load(decoding_path) as decoding_results:
            trial_offset = 0
            for session in sessions:
                # Decoding trial IDs are 1-based rows of the full TW trial axis.
                decoding_trial_rows = decoding_results[f'sess{session}_trial_ids']
                decoding_trial_rows = decoding_trial_rows.astype(int)
                trial_indices = decoding_trial_rows - 1
                tw_path = Path(tw_dir) / f'sub{subject:02d}_session{session}.pkl'
                with tw_path.open('rb') as tw_file:
                    tw_data = pickle.load(tw_file)

                cue_loc = np.asarray(tw_data['cue_loc'])[trial_indices].astype(int)
                rt = decoding_results[f'sess{session}_rt'].astype(float)
                bad_trials = np.asarray(tw_data['is_bad_epoch'], bool)[trial_indices]

                # TW sampled at 500 Hz with 250-sample windows; use the window centre times.
                tw_centers = np.round(tw_data['epoch_tmin'] + (np.asarray(tw_data['starts']) + 125) / tw_data['sfreq'], 6)
                tw_time_mask = (tw_centers >= tw_window[0]) & (tw_centers <= tw_window[1])
                frequencies = np.asarray(tw_data['ff'])
                frequency_mask = (frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1])

                trial_table = pd.DataFrame({
                    'subject': subject,
                    'session': session,
                    'trial': decoding_trial_rows,
                    'source_row_1based': decoding_trial_rows,
                    'global_trial': trial_offset + decoding_trial_rows,
                    'cue_loc': cue_loc,
                    'rt': rt,
                })
                for target in TARGETS[:2]:
                    values = np.asarray(decoding_results[f'sess{session}_trial_dec_{target}'], float)
                    trial_table['D_' + target] = values
                trial_table['D_cue_minus_uncue'] = trial_table.D_cue - trial_table.D_uncue

                for direction in ['BW', 'FW']:
                    signal = tw_data[direction.lower() + 'Max'][:, frequency_mask][:, :, tw_time_mask][:, :, :, trial_indices]
                    surrogate = tw_data[direction.lower() + 'MaxSS'][:, frequency_mask][:, :, tw_time_mask][:, :, :, trial_indices]
                    normalized_tw = 10 * np.log10(signal / surrogate)

                    # Midline excluded; cue_loc 1 = left cue, so contra = right; cue_loc 2 the converse.
                    left_mean = normalized_tw[:5].mean(axis=(0, 1, 2))
                    right_mean = normalized_tw[6:].mean(axis=(0, 1, 2))
                    contra_mean = np.where(cue_loc == 1, right_mean, left_mean)
                    ipsi_mean = np.where(cue_loc == 1, left_mean, right_mean)
                    trial_table[direction + '_contra'] = contra_mean
                    trial_table[direction + '_ipsi'] = ipsi_mean
                    trial_table[direction + '_contra_minus_ipsi'] = contra_mean - ipsi_mean
                    del signal, surrogate, normalized_tw

                # cue, uncue and their difference always use the same cued-side lateralization.
                keep_trials = np.isfinite(rt) & (rt > 0) & ~bad_trials
                matched_tables.append(trial_table.loc[keep_trials].copy())
                # Advance by the full session, preserving gaps from rejected trials.
                trial_offset += len(tw_data['cue_loc'])
                del tw_data
                gc.collect()

    return pd.concat(matched_tables, ignore_index=True)


def analyze(trials):
    """Per subject partial r (intercept + global trial order removed); group summary of Fisher z."""
    rows = []
    for subject, table in trials.groupby('subject'):
        table = table.sort_values('global_trial')
        design = np.column_stack([np.ones(len(table)), table.global_trial])
        for direction in DIRECTIONS:
            for target in TARGETS:
                for side in SIDES:
                    x = table[direction + '_' + side].to_numpy()
                    y = table['D_' + target].to_numpy()
                    r = partial_correlation(x, y, design)
                    rows.append(dict(subject=subject, direction=direction,
                        target=target, side=side, n_trials=len(table),
                        n_sessions=table.session.nunique(), r=r, fisher_z=np.arctanh(r)))
    subjects = pd.DataFrame(rows)
    keys = ['direction', 'target', 'side']
    results = []
    for key, table in subjects.groupby(keys):
        summary = group_summary(table.fisher_z)
        results.append(dict(zip(keys, key), **summary))
    return pd.DataFrame(results), subjects
