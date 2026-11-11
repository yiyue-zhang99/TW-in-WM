"""Group tests and Bayes factors for TW–fixed-window decoding correlations."""
import numpy as np
import pandas as pd
from scipy import stats
from pingouin import bayesfactor_ttest, bayesfactor_pearson


def one_sample(values):
    """Two-sided one-sample t test against zero, 95% t CI and Bayesian t test."""
    values = np.asarray(values, float)
    n = len(values)
    mean, sem = values.mean(), stats.sem(values)
    test = stats.ttest_1samp(values, 0)
    half = stats.t.ppf(.975, n - 1) * sem
    return dict(N=n, mean=mean, sem=sem, ci_low=mean-half, ci_high=mean+half,
                t=test.statistic, df=n-1, p_raw=test.pvalue,
                BF10=bayesfactor_ttest(float(test.statistic), n, r=.707),
                significant=bool(test.pvalue < .05), bf_prior_scale=.707)


def group_summary(fisher_z):
    """Test subject Fisher z against zero; express the mean and 95% t CI as r."""
    result = one_sample(fisher_z)
    result['mean_fisher_z'] = result.pop('mean')
    result['sem_fisher_z'] = result.pop('sem')
    result['r'] = np.tanh(result['mean_fisher_z'])
    result['ci_low'], result['ci_high'] = np.tanh([result['ci_low'], result['ci_high']])
    result['test'] = 'two-sided one-sample t-test on Fisher z'
    result['ci_method'] = '95% t CI on mean Fisher z, transformed to r'
    return result


def ttest_bayes_summary(subjects):
    """Summarize each target/measure; add descriptive mean r and SEM for plots."""
    return pd.DataFrame([
        dict(target=target, measure=measure, **group_summary(table.fisher_z),
             mean_r=table.r.mean(), sem_r=stats.sem(table.r))
        for (target, measure), table in subjects.groupby(['target', 'measure'], sort=False)
    ])


def pair_result(x, y):
    """Between-subject Pearson test (df=N-2) and Bayesian correlation, not a mean test."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    test = stats.pearsonr(x, y)
    r, p = test.statistic, test.pvalue
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r*r))
    ci = test.confidence_interval()
    return dict(N=n, r=r, t=t, df=n-2, p_raw=p,
                ci_low=ci.low, ci_high=ci.high,
                BF10=bayesfactor_pearson(float(r), n, kappa=1.),
                significant=bool(p < .05), test='two-sided Pearson correlation',
                bf_prior_kappa=1.)
