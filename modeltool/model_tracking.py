#########################################################
# Description: Production model tracking tables
# Author: Xinyue He
# Created date: 2026.09
#########################################################

import os
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from .tracking_report import build_report

EPS = 1e-6      # floor for empty-bin shares in PSI / IV
MISSING = -1    # bin id for missing values
PSI_COLS = ['bin', 'bin_range', 'bmk_ratio', 'ratio', 'psi']
GAINS_COLS = ['bin', 'bin_range', 'Total_Int', 'Min_Score', 'Mean_Score', 'Max_Score', 'Min_Odds', 'Responder',
              'PercResponder', 'CumPercResponder', 'CumPercNonResponder', 'ResponseRate', 'KS', 'lift', 'cum_lift']

# One benchmark column: interior cut points, share of rows per bin, IV on its labelled rows
Bmk = namedtuple('Bmk', 'cut ratio iv')


def cuts(x, nbins):
    '''Interior quantile cut points of x; the outer bins are open-ended (-inf, c0] ... (cN, inf).'''
    x = x.dropna()
    return np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)[1:-1])) if len(x) else np.array([])


def assign(x, cut):
    '''Bin id of each value: 0..len(cut) for the right-closed bins, MISSING for NaN.'''
    return np.where(x.isna(), MISSING, np.searchsorted(cut, x))


def bin_labels(cut):
    edges = np.r_[-np.inf, cut, np.inf]
    return {MISSING: 'missing', **{i: f'({lo:.4g}, {hi:.4g}]' for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))}}


def iv(ids, y):
    '''Information value of bin ids against a 0/1 target.'''
    grp = pd.Series(y).groupby(ids).agg(['sum', 'size'])
    bad, good = grp['sum'], grp['size'] - grp['sum']
    if bad.sum() == 0 or good.sum() == 0:
        return np.nan
    b, g = (bad / bad.sum()).clip(lower=EPS), (good / good.sum()).clip(lower=EPS)
    return ((b - g) * np.log(b / g)).sum()


def psi_table(x, bmk):
    '''Share of x in each benchmark bin next to the benchmark share, with each bin's PSI contribution.'''
    t = pd.DataFrame({'bmk_ratio': bmk.ratio,
                      'ratio': pd.Series(assign(x, bmk.cut)).value_counts(normalize=True)}).fillna(0).sort_index()
    a, e = t['ratio'].clip(lower=EPS), t['bmk_ratio'].clip(lower=EPS)
    t['psi'] = (a - e) * np.log(a / e)
    t.insert(0, 'bin_range', t.index.map(bin_labels(bmk.cut)))
    return t.rename_axis('bin').reset_index()


def auc_ks(y, prob):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    fpr, tpr, _ = roc_curve(y, prob)
    return auc(fpr, tpr), (tpr - fpr).max()


def gains_table(y, prob, score, cut):
    '''Gains table with the model_report.KS columns on prob bins, riskiest (highest prob) bin first.'''
    ids = assign(prob, cut)
    d = pd.DataFrame({'y': y, 'prob': prob.values, 'score': score.values, 'bin': ids})
    t = d.groupby('bin').agg(Total_Int=('y', 'size'), Min_Score=('prob', 'min'), Mean_Score=('prob', 'mean'),
                             Max_Score=('prob', 'max'), Min_Odds=('score', 'min'), Responder=('y', 'sum'))
    t = t.reindex(np.union1d(np.arange(len(cut) + 1), ids)[::-1])
    t[['Total_Int', 'Responder']] = t[['Total_Int', 'Responder']].fillna(0)
    total, bad = t['Total_Int'].sum(), t['Responder'].sum()
    t['PercResponder'] = t['Responder'] / bad
    t['CumPercResponder'] = t['Responder'].cumsum() / bad
    t['CumPercNonResponder'] = (t['Total_Int'] - t['Responder']).cumsum() / (total - bad)
    t['ResponseRate'] = t['Responder'] / t['Total_Int']
    t['KS'] = t['CumPercResponder'] - t['CumPercNonResponder']
    t['lift'] = t['ResponseRate'] / (bad / total)
    t['cum_lift'] = t['Responder'].cumsum() / t['Total_Int'].cumsum() / (bad / total)
    t.insert(0, 'bin_range', t.index.map(bin_labels(cut)))
    return t.rename_axis('bin').reset_index()


def stack(frames, keys, columns):
    '''{key: frame} -> one long table with the key(s) as leading columns; empty-safe.'''
    if not frames:
        return pd.DataFrame(columns=keys + columns)
    return pd.concat(frames, names=keys).reset_index(level=keys).reset_index(drop=True)[keys + columns]


def save(tables, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(os.path.join(outdir, f'{name}.csv'), index=False)


class model_tracker():
    '''
    Daily (frontend) and weekly (backend) monitoring tables for a production model.

    Benchmark: the first month of data, [first date, first date + 1 month). Every bin cut
    (score and attributes) comes from it. A day or week that falls inside that month is
    compared against the rest of the month, never against itself, so tracking works
    from the second day even while the first month is still filling up.

    Input:
        df: DataFrame or csv path
            one row per scored application, with the model attributes, prob, score and y
        model_name: string
            written into every output table
        attrs: list of string
            model attribute columns (numeric)
        date: string
            scoring date column
        y: string
            0/1 target column, NaN while not yet matured
        prob: string default probs
            model probability of y=1; score bins, AUC/KS and the gains table use it
        score: string default scr
            model score column
        nbins: int default 10
            number of quantile bins

    Output: run() returns a dict of DataFrames, also saved as <name>.csv when outdir is given
        benchmark             bmk_start, bmk_end, n_days, n, n_labelled, avg_score, avg_prob, bad_rate, auc, ks
        benchmark_gains       gains table of the whole first month, same columns as weekly_gains without week
        frontend
            daily_summary     date, n, avg_score, avg_prob, score_psi, is_bmk
            daily_score_psi   date, bin, bin_range, bmk_ratio, ratio, psi
            daily_attr_psi    date, attribute, psi
        backend (IV, AUC/KS and gains on labelled rows only)
            weekly_perf       week, n, avg_score, avg_prob, n_labelled, bad_rate, score_psi, is_bmk, auc, ks
            weekly_score_psi  week, bin, bin_range, bmk_ratio, ratio, psi
            weekly_attr       week, attribute, psi, bmk_iv, iv, iv_shift
            weekly_attr_bins  week, attribute, bin, bin_range, bmk_ratio, ratio, psi
            weekly_gains      week, bin, bin_range, + model_report gains columns
        Weeks start on Monday. bin is ordered by value (0 = lowest) and MISSING (-1) holds NaN.

    sample code:
    tracker = model_tracker('prod_scores.csv',
                            model_name='acq_cl_sms_v1',
                            attrs=xgb.Booster(model_file='model.json').feature_names,
                            date='apply_date',
                            y='fpd7_ind')
    tables = tracker.run(outdir='tracking/')
    '''
    def __init__(self, df, model_name, attrs, date, y, prob='probs', score='scr', nbins=10) -> None:
        df = df if isinstance(df, pd.DataFrame) else pd.read_csv(df)
        self.df = df[[date, prob, score, y] + list(attrs)].copy()
        self.df[[prob, score, y]] = self.df[[prob, score, y]].astype('float64')   # float32 would not survive the csv exactly
        self.df['date'] = pd.to_datetime(self.df[date]).dt.normalize()
        self.df['week'] = self.df['date'].dt.to_period('W').dt.start_time

        self.model_name = model_name
        self.attrs = list(attrs)
        self.y, self.prob, self.score, self.nbins = y, prob, score, nbins

        self.bmk_end = self.df['date'].min() + pd.DateOffset(months=1)
        self.in_bmk = self.df['date'] < self.bmk_end
        self._bmk_cache = {}

    def _benchmark(self, period=None, value=None):
        '''
        Benchmark for one day/week: the first month minus that period itself, as {column: Bmk}.
        No period gives the whole first month, which every period after the first month shares.
        None when the first month holds nothing but that period (e.g. only one day scored so far).
        '''
        in_self = self.df[period] == value if period else pd.Series(False, index=self.df.index)
        key = (period, value) if (self.in_bmk & in_self).any() else None
        if key not in self._bmk_cache:
            rows = self.df[self.in_bmk & ~in_self]
            labelled = rows[self.y].notna().values
            bmk = {}
            for col in [self.prob] + self.attrs:
                cut = cuts(rows[col], self.nbins)
                ids = assign(rows[col], cut)
                all_bins = np.union1d(np.arange(len(cut) + 1), ids)
                ratio = pd.Series(ids).value_counts(normalize=True).reindex(all_bins, fill_value=0)
                bmk[col] = Bmk(cut, ratio, iv(ids[labelled], rows[self.y].values[labelled]))
            self._bmk_cache[key] = bmk if len(rows) else None
        return self._bmk_cache[key]

    def _psi(self, period):
        '''Per-bin score and attribute PSI of every day or week, each against its own benchmark.'''
        score, attrs = {}, {}
        for value, rows in self.df.groupby(period):
            bmk = self._benchmark(period, value)
            if bmk is None:
                continue
            score[value] = psi_table(rows[self.prob], bmk[self.prob])
            attrs.update({(value, a): psi_table(rows[a], bmk[a]) for a in self.attrs})
        return stack(score, [period], PSI_COLS), stack(attrs, [period, 'attribute'], PSI_COLS)

    def benchmark(self):
        '''The first-month reference that later periods are compared against, with its gains table.'''
        rows = self.df[self.in_bmk]
        labelled = rows[rows[self.y].notna()]
        y = labelled[self.y].values
        auc_, ks = auc_ks(y, labelled[self.prob])
        gains = (gains_table(y, labelled[self.prob], labelled[self.score], self._benchmark()[self.prob].cut)
                 if len(labelled) else pd.DataFrame(columns=GAINS_COLS))
        summary = pd.DataFrame([{
            'bmk_start': rows['date'].min(), 'bmk_end': self.bmk_end - pd.Timedelta(days=1),
            'n_days': rows['date'].nunique(), 'n': len(rows), 'n_labelled': len(labelled),
            'avg_score': rows[self.score].mean(), 'avg_prob': rows[self.prob].mean(),
            'bad_rate': labelled[self.y].mean(), 'auc': auc_, 'ks': ks}])
        return {'benchmark': summary, 'benchmark_gains': gains}

    def daily(self):
        '''Frontend: scoring count, average score, score PSI by bin and attribute PSI per day.'''
        score_psi, attr_bins = self._psi('date')
        summary = self.df.groupby('date').agg(n=(self.prob, 'size'),
                                              avg_score=(self.score, 'mean'),
                                              avg_prob=(self.prob, 'mean'))
        summary['score_psi'] = score_psi.groupby('date')['psi'].sum().astype(float)
        summary['is_bmk'] = summary.index < self.bmk_end
        return {'daily_summary': summary.reset_index(),
                'daily_score_psi': score_psi,
                'daily_attr_psi': attr_bins.groupby(['date', 'attribute'], sort=False)['psi'].sum().reset_index()}

    def weekly(self):
        '''Backend: performance, score and attribute stability, IV shift and the gains table per week.'''
        score_psi, attr_bins = self._psi('week')
        perf = self.df.groupby('week').agg(n=(self.prob, 'size'),
                                           avg_score=(self.score, 'mean'),
                                           avg_prob=(self.prob, 'mean'),
                                           n_labelled=(self.y, 'count'),
                                           bad_rate=(self.y, 'mean'))
        perf['score_psi'] = score_psi.groupby('week')['psi'].sum().astype(float)
        perf['is_bmk'] = perf.index < self.bmk_end

        scores, gains, ivs = [], {}, []
        for week, rows in self.df[self.df[self.y].notna()].groupby('week'):
            y = rows[self.y].values
            scores.append((week, *auc_ks(y, rows[self.prob])))
            bmk = self._benchmark('week', week)
            if bmk is None:
                continue
            gains[week] = gains_table(y, rows[self.prob], rows[self.score], bmk[self.prob].cut)
            ivs += [(week, a, bmk[a].iv, iv(assign(rows[a], bmk[a].cut), y)) for a in self.attrs]

        # astype(float): with no labels yet these frames are empty and would otherwise come out as object columns
        perf = perf.join(pd.DataFrame(scores, columns=['week', 'auc', 'ks']).set_index('week').astype(float))
        attr = (attr_bins.groupby(['week', 'attribute'], sort=False)['psi'].sum().astype(float).to_frame()
                .join(pd.DataFrame(ivs, columns=['week', 'attribute', 'bmk_iv', 'iv']).set_index(['week', 'attribute']).astype(float)))
        attr['iv_shift'] = attr['iv'] - attr['bmk_iv']
        return {'weekly_perf': perf.reset_index(),
                'weekly_score_psi': score_psi,
                'weekly_attr': attr.reset_index(),
                'weekly_attr_bins': attr_bins,
                'weekly_gains': stack(gains, ['week'], GAINS_COLS)}

    def run(self, outdir=None):
        '''
        object entrance to build all the tables of this model
        '''
        tables = {**self.benchmark(), **self.daily(), **self.weekly()}
        for t in tables.values():
            t.insert(0, 'model', self.model_name)
        if outdir:
            save(tables, outdir)
        return tables


def track_models(models, outdir=None, report='tracking_report.html'):
    '''
    Run model_tracker for several models and stack their tables; the model column tells them apart.
    Input:
        models: list of dict
            model_tracker arguments, one dict per model. Models scored in one table can share
            the same DataFrame and point to their own prob / score / attrs columns.
        outdir: string
            if given, the stacked tables are saved there as <name>.csv next to the HTML report
        report: string
            file name of the HTML report inside outdir

    sample code:
    tables = track_models([dict(df=df_sms, model_name='acq_cl_sms_v1', attrs=attrs_sms,
                                date='apply_date', y='fpd7_ind'),
                           dict(df=df_app, model_name='acq_cl_app_v2', attrs=attrs_app,
                                date='apply_date', y='fpd7_ind', prob='app_probs', score='app_scr')],
                          outdir='tracking/')
    '''
    runs = [model_tracker(**m).run() for m in models]
    tables = {name: pd.concat([r[name] for r in runs if len(r[name])] or [runs[0][name]], ignore_index=True)
              for name in runs[0]}
    if outdir:
        save(tables, outdir)
        build_report(tables, os.path.join(outdir, report))
    return tables
