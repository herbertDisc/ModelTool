#########################################################
# Description: HTML tracking report from the model_tracker tables
# Author: Xinyue He
# Created date: 2026.09
#########################################################

'''
build_report() turns the tracking tables into one self-contained HTML file: inline SVG charts,
no network, no JS libraries. It reads nothing but the tables, so the same tables always give
the same file. Rebuild a report from a saved table folder with

    python -m modeltool.tracking_report tracking/ -o tracking/tracking_report.html

Layout: an overview tab ranking every model, then one tab per model with
    KPI tiles | volume & score | score stability | performance | attribute issues + drill-down
Every chart has its data table one click away; flag thresholds are listed in the footer.
'''

import argparse
import hashlib
import html
import json
import os
from string import Template
from types import SimpleNamespace

import numpy as np
import pandas as pd

PSI_WATCH, PSI_SHIFT = 0.10, 0.25     # PSI bands: stable < watch < shift
AUC_WATCH, AUC_SHIFT = 0.03, 0.05     # AUC drop vs benchmark
IV_DROP, IV_MIN = 0.30, 0.02          # relative IV drop, flagged only when benchmark IV >= IV_MIN
IV_WEEKS = 4                          # latest matured weeks averaged for the recent IV
RECENT = 8                            # weeks looked back for persistence
MATURE = 0.5                          # a week counts as matured once this share of it is labelled
MISSING = -1                          # bin id model_tracker gives missing values

TABLES = ['benchmark', 'benchmark_gains', 'daily_summary', 'weekly_perf', 'weekly_score_psi',
          'weekly_attr', 'weekly_attr_bins', 'weekly_gains']
STATUS = [('stable', '●'), ('watch', '▲'), ('shift', '■')]
INT, PCT, F1, F2, F3, F4 = '{:,.0f}', '{:.1%}', '{:.1f}', '{:.2f}', '{:.3f}', '{:.4f}'
GAINS_TABLE = [('bin_range', 'prob bin', None), ('Total_Int', 'Total_Int', INT), ('Min_Score', 'Min_Score', F4),
               ('Mean_Score', 'Mean_Score', F4), ('Max_Score', 'Max_Score', F4), ('Min_Odds', 'Min_Odds', INT),
               ('Responder', 'Responder', INT), ('PercResponder', 'PercResponder', PCT),
               ('CumPercResponder', 'CumPercResponder', PCT), ('CumPercNonResponder', 'CumPercNonResponder', PCT),
               ('ResponseRate', 'ResponseRate', PCT), ('KS', 'KS', F3), ('lift', 'lift', F2), ('cum_lift', 'cum_lift', F2)]

VW, WIDE = 640, 1000                  # viewBox widths of a chart / full-width heatmap
M = dict(l=52, r=12, t=14, b=26)      # chart margins


# ---------------------------------------------------------------- formatting

def esc(s):
    return html.escape(str(s))


def fmt(v, spec=F3):
    return '–' if v is None or pd.isna(v) else spec.format(v)


def psi_status(v):
    return 0 if pd.isna(v) else int(v >= PSI_WATCH) + int(v >= PSI_SHIFT)


def auc_status(drop):
    return 0 if pd.isna(drop) else int(drop >= AUC_WATCH) + int(drop >= AUC_SHIFT)


def chip(s):
    name, icon = STATUS[int(s)]
    return f'<span class="chip st{int(s)}"><i>{icon}</i>{name}</span>'


def seq_cls(v, breaks):
    '''Sequential class q1..q5 of v against 4 ascending breaks.'''
    return None if pd.isna(v) else f'q{np.searchsorted(breaks, v, side="right") + 1}'


def div_cls(v, breaks):
    '''Diverging class: d0 inside the first break, then dp1..3 above zero (blue) / dn1..3 below (red).'''
    if pd.isna(v):
        return None
    k = int(np.searchsorted(breaks, abs(v), side='right'))
    return 'd0' if k == 0 else f'd{"p" if v > 0 else "n"}{k}'


def missing_last(desc=False):
    '''Sort key for bin ids that puts the missing bin last in either direction.'''
    fill = -np.inf if desc else np.inf
    return lambda k: k.map(lambda b: fill if b == MISSING else b)


def delta(d, spec, good=None):
    '''Signed change vs benchmark; good=+1/-1 says which direction is good, None keeps it neutral.'''
    if pd.isna(d):
        return ''
    cls = '' if good is None or d == 0 else ('pos' if d * good > 0 else 'neg')
    return f'<span class="dlt {cls}">{"+" if d > 0 else ""}{spec.format(d)} vs benchmark</span>'


# ---------------------------------------------------------------- SVG charts

def svg(w, h, body, label=''):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">{"".join(body)}</svg>'


def ticks(lo, hi, n=4):
    '''Round tick values covering [lo, hi].'''
    if hi - lo < 1e-9:
        lo, hi = lo - max(abs(lo) * .1, 1e-3), hi + max(abs(hi) * .1, 1e-3)
    raw = (hi - lo) / n
    step = 10 ** np.floor(np.log10(raw))
    step *= next(m for m in (1, 2, 5, 10) if raw <= step * m)
    return np.arange(np.floor(lo / step), np.ceil(hi / step) + .5) * step


def path(xs, ys):
    '''SVG path through the finite points, broken at gaps.'''
    d, pen = [], 'M'
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            d.append(f'{pen}{x:.1f},{y:.1f}')
            pen = 'L'
        else:
            pen = 'M'
    return ''.join(d)


def bar(x, y0, w, y1):
    '''Column from baseline y0 up to y1 with a 4px rounded top.'''
    r = min(4, w / 2, y0 - y1)
    return (f'M{x:.1f},{y0:.1f}V{y1 + r:.1f}Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f}'
            f'H{x + w - r:.1f}Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f}V{y0:.1f}Z')


def y_axis(yt, Y, fmt_spec, w=VW):
    return [f'<line class="gl" x1="{M["l"]}" x2="{w - M["r"]}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
            f'<text x="{M["l"] - 6}" y="{Y(v) + 3.5:.1f}" text-anchor="end">{fmt(v, fmt_spec)}</text>' for v in yt]


def line_chart(x, series, yfmt, tfmt=None, refs=(), shade=None, zero=False, h=200, w=VW):
    '''
    Time series on one y axis.
    x: Timestamps; series: [(name, values, style)] with style '1' accent / '0' context gray;
    refs: [(label, y, css class)] horizontal lines; shade: benchmark (start, end); zero: y axis from 0;
    w: viewBox width - keep it near the rendered width so the 11px text stays 11px.
    Every x owns a full-height hover band whose tooltip lists all series there.
    '''
    x = pd.DatetimeIndex(x)
    t = np.asarray((x - x[0]).days, float)
    ys = [np.asarray(v, float) for _, v, _ in series]
    known = np.concatenate(ys + [np.array([r[1] for r in refs], float)])
    known = known[np.isfinite(known)]
    lo, hi = (known.min(), known.max()) if len(known) else (0., 1.)
    yt = ticks(min(lo, 0) if zero else lo, hi)
    top, bot = M['t'], h - M['b']
    X = lambda v: M['l'] + v / max(t[-1], 1) * (w - M['l'] - M['r'])
    Y = lambda v: bot - (v - yt[0]) / (yt[-1] - yt[0]) * (bot - top)

    out = []
    if shade is not None:
        a, b = np.clip([(pd.Timestamp(s) - x[0]).days for s in shade], 0, t[-1])
        if b > a:
            out.append(f'<rect class="band" x="{X(a):.1f}" y="{top}" width="{X(b) - X(a):.1f}" height="{bot - top}"/>'
                       f'<text x="{X(a) + 4:.1f}" y="{top + 10}">benchmark</text>')
    out += y_axis(yt, Y, tfmt or yfmt, w)
    for i in np.unique(np.linspace(0, len(x) - 1, min(len(x), max(w // 110, 3))).round().astype(int)):
        anchor = 'end' if i == len(x) - 1 and i else 'middle'
        out.append(f'<text x="{X(t[i]):.1f}" y="{h - 8}" text-anchor="{anchor}">{x[i]:%m-%d}</text>')
    for label, v, cls in refs:
        if pd.notna(v):
            out.append(f'<line class="{cls}" x1="{M["l"]}" x2="{w - M["r"]}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                       f'<text x="{w - M["r"]}" y="{Y(v) - 4:.1f}" text-anchor="end">{esc(label)}</text>')
    for (_, _, s), y in zip(series, ys):
        out.append(f'<path class="ln ln{s}" d="{path(X(t), Y(y))}"/>')
        if len(x) <= 60:
            out += [f'<circle class="dt dt{s}" cx="{X(ti):.1f}" cy="{Y(yi):.1f}" r="4"/>'
                    for ti, yi in zip(t, y) if np.isfinite(yi)]
    edges = np.r_[t[0], (t[1:] + t[:-1]) / 2, t[-1]]
    for i, xi in enumerate(x):
        tip = '\n'.join([f'{xi:%Y-%m-%d}']
                        + [f'{fmt(y[i], yfmt)}\t{name}\t{s}' for (name, _, s), y in zip(series, ys)]
                        + [f'{fmt(v, yfmt)}\t{label}' for label, v, _ in refs if pd.notna(v)])
        out.append(f'<g class="hb"><line class="xh" x1="{X(t[i]):.1f}" x2="{X(t[i]):.1f}" y1="{top}" y2="{bot}"/>'
                   f'<rect class="hit" x="{X(edges[i]):.1f}" y="{top}" width="{max(X(edges[i + 1]) - X(edges[i]), 1):.1f}" '
                   f'height="{bot - top}" data-tip="{esc(tip)}"/></g>')
    return svg(w, h, out, ', '.join(name for name, _, _ in series))


def column_chart(cats, series, yfmt, tfmt=None, tips=None, h=200):
    '''Grouped columns from a 0 baseline; cats label the x slots, tips[i] heads slot i's tooltip.'''
    n, k = len(cats), len(series)
    ys = [np.asarray(v, float) for _, v, _ in series]
    hi = np.nanmax(np.concatenate(ys)) if n else 1
    yt = ticks(0, hi if np.isfinite(hi) and hi > 0 else 1)
    top, bot = M['t'], h - M['b']
    Y = lambda v: bot - (v - yt[0]) / (yt[-1] - yt[0]) * (bot - top)
    slot = (VW - M['l'] - M['r']) / n
    bw = min(24, (slot * .7 - 2 * (k - 1)) / k)
    out = y_axis(yt, Y, tfmt or yfmt)
    for i, c in enumerate(cats):
        x0 = M['l'] + i * slot + (slot - (k * bw + 2 * (k - 1))) / 2
        for j, ((_, _, s), y) in enumerate(zip(series, ys)):
            if np.isfinite(y[i]) and Y(y[i]) < bot:
                out.append(f'<path class="b{s}" d="{bar(x0 + j * (bw + 2), bot, bw, Y(y[i]))}"/>')
        tip = '\n'.join([tips[i] if tips else str(c)]
                        + [f'{fmt(y[i], yfmt)}\t{name}\t{s}' for (name, _, s), y in zip(series, ys)])
        out.append(f'<text x="{M["l"] + (i + .5) * slot:.1f}" y="{h - 8}" text-anchor="middle">{esc(c)}</text>'
                   f'<rect class="hitc" x="{M["l"] + i * slot:.1f}" y="{top}" width="{slot:.1f}" height="{bot - top}" '
                   f'data-tip="{esc(tip)}"/>')
    return svg(VW, h, out, ', '.join(name for name, _, _ in series))


def heatmap(rows, cols, cls, tips, row_attrs=None, label_w=200, cell_h=16):
    '''
    rows x week-columns of class-coloured cells separated by a 2px surface gap.
    cls / tips: 2-D lists (None = no cell); row_attrs: extra attributes per row, e.g. a drill-down link.
    '''
    n = max(len(cols), 1)
    cw = (WIDE - label_w - 8) / n
    h = 4 + len(rows) * cell_h + 20
    out = []
    for i, r in enumerate(rows):
        y = 4 + i * cell_h
        out.append(f'<g{row_attrs[i] if row_attrs else ""}>'
                   f'<text x="{label_w - 8}" y="{y + cell_h / 2 + 3.5:.1f}" text-anchor="end">{esc(r)}</text>')
        out += [f'<rect class="cell {cls[i][j]}" x="{label_w + j * cw + 1:.1f}" y="{y + 1}" width="{cw - 2:.1f}" '
                f'height="{cell_h - 2}" rx="2" data-tip="{esc(tips[i][j])}"/>' for j in range(len(cols)) if cls[i][j]]
        out.append('</g>')
    for j in range(0, len(cols), int(np.ceil(n / 12))):
        out.append(f'<text x="{label_w + (j + .5) * cw:.1f}" y="{h - 6}" text-anchor="middle">{cols[j]:%m-%d}</text>')
    return svg(WIDE, h, out)


def sparkline(v, w=90, h=20):
    '''Weekly PSI trend for a table cell, with the watch threshold as a hairline.'''
    v = np.asarray(v, float)
    hi = max(np.nanmax(v) if np.isfinite(v).any() else 0, PSI_SHIFT)
    X = lambda i: 2 + i / max(len(v) - 1, 1) * (w - 4)
    Y = lambda y: h - 2 - y / hi * (h - 4)
    last = np.flatnonzero(np.isfinite(v))
    dot = f'<circle class="dt1" cx="{X(last[-1]):.1f}" cy="{Y(v[last[-1]]):.1f}" r="2.5"/>' if len(last) else ''
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
            f'<line class="ref-w" x1="0" x2="{w}" y1="{Y(PSI_WATCH):.1f}" y2="{Y(PSI_WATCH):.1f}"/>'
            f'<path class="ln ln1" style="stroke-width:1.5" d="{path(X(np.arange(len(v))), Y(v))}"/>{dot}</svg>')


# ---------------------------------------------------------------- HTML pieces

def legend(items, line=False):
    key = 'lk' if line else 'sw'
    return '<div class="legend">' + ''.join(f'<span><i class="{key} {c}"></i>{esc(l)}</span>' for c, l in items) + '</div>'


PSI_BREAKS = [0.02, 0.05, PSI_WATCH, PSI_SHIFT]
PSI_LEGEND = legend([('q1', '< 0.02'), ('q2', '0.02–0.05'), ('q3', '0.05–0.10'),
                     ('q4', '0.10–0.25 watch'), ('q5', '≥ 0.25 shift')])


def div_legend(breaks, spec, up, down):
    b = [spec.format(x) for x in breaks]
    return legend([('dn3', f'≤ −{b[2]} {down}'), ('dn2', f'−{b[1]}'), ('dn1', f'−{b[0]}'), ('d0', f'within ±{b[0]}'),
                   ('dp1', f'+{b[0]}'), ('dp2', f'+{b[1]}'), ('dp3', f'≥ +{b[2]} {up}')])


def card(title, sub, body, table=None, wide=False):
    tbl = f'<details><summary>Table</summary>{table}</details>' if table else ''
    return (f'<figure class="card{" wide" if wide else ""}"><figcaption><b>{esc(title)}</b>'
            f'<span>{esc(sub)}</span></figcaption>{body}{tbl}</figure>')


def html_table(df, cols, sortable=False, row_attrs=None):
    '''cols: [(column, header, spec)]; spec is a format string, a function returning html, or None for text.'''
    left = [' class=l' if callable(spec) or not pd.api.types.is_numeric_dtype(df[c]) else '' for c, _, spec in cols]
    head = ''.join(f'<th{lc}>{esc(h)}</th>' for (_, h, _), lc in zip(cols, left))
    body = []
    for i, (_, r) in enumerate(df.iterrows()):
        cells = []
        for (c, _, spec), lc in zip(cols, left):
            v = r[c]
            if callable(spec):
                txt = spec(v)
            elif spec:
                txt = fmt(v, spec)
            else:
                txt = esc(f'{v:%Y-%m-%d}' if isinstance(v, pd.Timestamp) else v)
            num = isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
            dv = f' data-v="{v if np.isfinite(v) else "-Infinity"}"' if num else ''
            cells.append(f'<td{lc}{dv}>{txt}</td>')
        body.append(f'<tr{row_attrs[i] if row_attrs else ""}>{"".join(cells)}</tr>')
    return (f'<div class="tscroll"><table{" class=sortable" if sortable else ""}><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def pivot_table(p, spec, row_name):
    '''Table view of a heatmap: rows x weeks.'''
    t = p.copy()
    t.columns = [f'{c:%Y-%m-%d}' for c in t.columns]
    t = t.rename_axis(row_name).reset_index()
    return html_table(t, [(row_name, row_name, None)] + [(c, c, spec) for c in t.columns[1:]])


def tile(label, value, sub='', note=''):
    return f'<div class="tile"><span class="lbl">{esc(label)}</span><span class="val">{value}</span>{sub}<span class="note">{note}</span></div>'


# ---------------------------------------------------------------- per-model numbers

def load_tables(src):
    '''The dict returned by model_tracker.run() / track_models(), or the folder it was saved to.'''
    if isinstance(src, dict):
        return src
    tables = {n: pd.read_csv(os.path.join(src, f'{n}.csv'), float_precision='round_trip') for n in TABLES}
    for t in tables.values():
        for c in ['date', 'week', 'bmk_start', 'bmk_end']:
            if c in t:
                t[c] = pd.to_datetime(t[c])
    return tables


def fingerprint(tables):
    '''Short hash of the report's input tables: equal fingerprints mean equal reports.'''
    h = hashlib.sha256()
    for n in TABLES:
        h.update(tables[n].to_csv(index=False).encode())
    return h.hexdigest()[:12]


def attribute_issues(attr, bins, matured):
    '''One row per attribute - latest PSI, persistence, IV change, missing rate - most severe first.'''
    psi = attr.pivot(index='attribute', columns='week', values='psi')
    latest, recent = psi.columns[-1], psi.iloc[:, -RECENT:]
    ivs = (attr[attr['week'].isin(matured[-IV_WEEKS:])].groupby('attribute')
           .agg(bmk_iv=('bmk_iv', 'last'), iv=('iv', 'mean')))
    miss = (bins[(bins['bin'] == MISSING) & (bins['week'] == latest)].set_index('attribute')
            [['bmk_ratio', 'ratio']].rename(columns={'bmk_ratio': 'miss_bmk', 'ratio': 'miss'}))
    t = pd.DataFrame({'psi': psi[latest], 'psi_max': recent.max(axis=1),
                      'weeks_flagged': (recent >= PSI_WATCH).sum(axis=1)}).join(ivs).join(miss)
    t[['miss_bmk', 'miss']] = t[['miss_bmk', 'miss']].fillna(0)
    t['iv_change'] = t['iv'] / t['bmk_iv'] - 1
    iv_drop = (t['bmk_iv'] >= IV_MIN) & (t['iv_change'] <= -IV_DROP)
    t['status'] = np.maximum(t['psi'].map(psi_status), iv_drop.astype(int))
    return t.sort_values(['status', 'psi'], ascending=False).rename_axis('attribute').reset_index()


def model_view(tables, model):
    '''Everything the report shows for one model, from its slice of the tables.'''
    get = lambda n: tables[n][tables[n]['model'] == model].drop(columns='model').reset_index(drop=True)
    v = SimpleNamespace(name=model, bmk=get('benchmark').iloc[0], daily=get('daily_summary'), perf=get('weekly_perf'),
                        spsi=get('weekly_score_psi'), attr=get('weekly_attr'), bins=get('weekly_attr_bins'),
                        gains=get('weekly_gains'), bgains=get('benchmark_gains'))
    v.last = v.perf.iloc[-1]
    v.week = v.last['week']
    v.week_days = v.daily['date'].between(v.week, v.week + pd.Timedelta(days=6)).sum()
    v.vol, v.vol_bmk = v.last['n'] / v.week_days, v.bmk['n'] / v.bmk['n_days']
    mat = v.perf[(v.perf['n_labelled'] >= MATURE * v.perf['n']) & v.perf['auc'].notna()]
    v.matured, v.mat = mat, (mat.iloc[-1] if len(mat) else None)
    v.auc_drop = v.bmk['auc'] - v.mat['auc'] if v.mat is not None else np.nan
    v.status = max(psi_status(v.last['score_psi']), auc_status(v.auc_drop))
    v.issues = attribute_issues(v.attr, v.bins, list(mat['week'])) if len(v.attr) else pd.DataFrame()
    return v


# ---------------------------------------------------------------- sections

def kpis(v):
    b, last, mat = v.bmk, v.last, v.mat
    part = f' · {v.week_days} of 7 days' if v.week_days < 7 else ''
    wk = f'week of {v.week:%Y-%m-%d}{part}'
    mwk = f'week of {mat["week"]:%Y-%m-%d} · {mat["n_labelled"]:,.0f} labelled' if mat is not None else 'no matured week yet'
    n_flag = (v.issues['status'] > 0).sum() if len(v.issues) else 0
    counts = f'{(v.issues["status"] == 2).sum()} shift · {(v.issues["status"] == 1).sum()} watch' if len(v.issues) else ''
    get = lambda k: mat[k] if mat is not None else np.nan
    return '<div class="tiles">' + ''.join([
        tile('Daily volume', fmt(v.vol, INT), delta(v.vol / v.vol_bmk - 1, '{:.1%}'), wk),
        tile('Average score', fmt(last['avg_score'], F1), delta(last['avg_score'] - b['avg_score'], F1), wk),
        tile('Score PSI', fmt(last['score_psi']), chip(psi_status(last['score_psi'])), wk),
        tile('AUC', fmt(get('auc')), delta(get('auc') - b['auc'], F3, good=1) + chip(auc_status(v.auc_drop)), mwk),
        tile('KS', fmt(get('ks')), delta(get('ks') - b['ks'], F3, good=1), mwk),
        tile('Attributes flagged', f'{n_flag} / {len(v.issues)}', f'<span class="dlt">{counts}</span>', wk),
    ]) + '</div>'


def volume_section(v):
    d, b = v.daily, v.bmk
    shade = (b['bmk_start'], b['bmk_end'])
    vol = line_chart(d['date'], [('scored', d['n'], '1')], INT, shade=shade, zero=True,
                     refs=[('benchmark avg', v.vol_bmk, 'ref')])
    score = line_chart(d['date'], [('average score', d['avg_score'], '1')], F1, shade=shade,
                       refs=[('benchmark avg', b['avg_score'], 'ref')])
    return ('<h2>Volume and score</h2><div class="cards">'
            + card('Daily scoring volume', 'applications scored per day', vol,
                   html_table(d, [('date', 'date', None), ('n', 'scored', INT)]))
            + card('Daily average score', 'mean model score per day', score,
                   html_table(d, [('date', 'date', None), ('avg_score', 'avg score', F1), ('avg_prob', 'avg prob', F4)]))
            + '</div>')


def score_section(v):
    p, s, b = v.perf, v.spsi, v.bmk
    refs = [(f'watch {PSI_WATCH:.2f}', PSI_WATCH, 'ref-w')]
    if p['score_psi'].max() >= PSI_WATCH:
        refs.append((f'shift {PSI_SHIFT:.2f}', PSI_SHIFT, 'ref-s'))
    psi = line_chart(p['week'], [('score PSI', p['score_psi'], '1')], F3, F2, zero=True,
                     shade=(b['bmk_start'], b['bmk_end']), refs=refs)
    psi_tbl = html_table(p.assign(status=p['score_psi'].map(psi_status)),
                         [('week', 'week', None), ('n', 'scored', INT), ('score_psi', 'score PSI', F3), ('status', 'status', chip)])
    body = [card('Weekly score PSI', 'score distribution vs the benchmark month', psi, psi_tbl)]
    if not len(s):
        body.append(card('Score distribution this week', 'needs a second week of data', '<p class="note">Nothing to compare yet.</p>'))
    else:
        latest = s[s['week'] == s['week'].max()].sort_values('bin', key=missing_last())
        cats = ['NA' if k == MISSING else str(k + 1) for k in latest['bin']]
        dist = column_chart(cats, [('benchmark', latest['bmk_ratio'], '0'), ('this week', latest['ratio'], '1')], PCT, '{:.0%}',
                            tips=[f'{r}\n{fmt(x, F4)}\tPSI contribution' for r, x in zip(latest['bin_range'], latest['psi'])])
        body.append(card('Score distribution this week', f'share per benchmark prob decile, 1 = lowest risk · week of {latest["week"].iloc[0]:%Y-%m-%d}',
                         legend([('k0', 'benchmark'), ('k1', 'this week')]) + dist,
                         html_table(latest, [('bin_range', 'prob bin', None), ('bmk_ratio', 'benchmark', PCT),
                                             ('ratio', 'this week', PCT), ('psi', 'PSI', F4)])))
        shift = s.assign(d=s['ratio'] - s['bmk_ratio'])
        grid = shift.pivot(index='bin', columns='week', values='d').sort_index(ascending=False, key=missing_last(desc=True))
        cell = shift.set_index(['bin', 'week'])
        labels = latest.set_index('bin')['bin_range'].reindex(grid.index).fillna('').tolist()
        tips = [[f'week of {w:%Y-%m-%d} · {labels[i]}\n{fmt(cell.at[(k, w), "ratio"], PCT)}\tshare this week\n'
                 f'{fmt(cell.at[(k, w), "bmk_ratio"], PCT)}\tbenchmark share\n{fmt(cell.at[(k, w), "psi"], F4)}\tPSI contribution'
                 if (k, w) in cell.index else '' for w in grid.columns] for i, k in enumerate(grid.index)]
        cls = [[div_cls(x, [0.01, 0.02, 0.04]) for x in row] for row in grid.values]
        body.append(card('Weekly score share shift', 'share per prob bin minus its benchmark share; riskiest bin on top',
                         div_legend([1, 2, 4], '{} pp', 'more', 'less') + heatmap(labels, list(grid.columns), cls, tips, label_w=150),
                         pivot_table(grid.set_axis(labels), '{:+.2%}', 'prob bin'), wide=True))
    return '<h2>Score stability</h2><div class="cards">' + ''.join(body) + '</div>'


def perf_section(v):
    m, b, g = v.matured, v.bmk, v.gains
    if not len(m):
        return '<h2>Performance</h2><p class="note">No week has matured labels yet.</p>'
    shade = (b['bmk_start'], b['bmk_end'])
    charts = [('AUC', 'auc', F3, F2), ('KS', 'ks', F3, F2), ('Bad rate', 'bad_rate', PCT, '{:.0%}')]
    body = [card(f'Weekly {title}', f'matured weeks (≥ {MATURE:.0%} labelled) vs the benchmark month',
                 line_chart(m['week'], [(title, m[k], '1')], spec, tspec, shade=shade, refs=[('benchmark', b[k], 'ref')], w=420),
                 html_table(m, [('week', 'week', None), ('n_labelled', 'labelled', INT), (k, title, spec)]))
            for title, k, spec, tspec in charts]
    g = g[g['week'].isin(m['week'])]
    if len(g):
        grid = g.pivot(index='bin', columns='week', values='lift').sort_index(ascending=False, key=missing_last(desc=True))
        cell = g.set_index(['bin', 'week'])
        labels = g[g['week'] == g['week'].max()].set_index('bin')['bin_range'].reindex(grid.index).fillna('').tolist()
        tips = [[f'week of {w:%Y-%m-%d} · {labels[i]}\n{fmt(cell.at[(k, w), "ResponseRate"], PCT)}\tbad rate\n'
                 f'{fmt(cell.at[(k, w), "lift"], F2)}\tlift\n{fmt(cell.at[(k, w), "Total_Int"], INT)}\tloans'
                 if (k, w) in cell.index else '' for w in grid.columns] for i, k in enumerate(grid.index)]
        cls = [[seq_cls(x, [0.6, 0.85, 1.15, 1.5]) for x in row] for row in grid.values]
        body.append(card('Rank ordering by week', 'lift (bin bad rate / week bad rate) per benchmark prob decile; riskiest bin on top',
                         legend([('q1', '< 0.60'), ('q2', '0.60–0.85'), ('q3', '0.85–1.15'), ('q4', '1.15–1.50'), ('q5', '≥ 1.50')])
                         + heatmap(labels, list(grid.columns), cls, tips, label_w=150),
                         pivot_table(grid.set_axis(labels), F2, 'prob bin'), wide=True))
    body.append(gains_by_week(v))
    return '<h2>Performance</h2><div class="cards c3">' + ''.join(body) + '</div>'


def gains_by_week(v):
    '''
    Lift chart + gains table with a week picker: any matured week against the benchmark month or another week.
    Pairs are chosen in the browser, so the chart and its lift table are drawn by liftChart() in JS from the
    JSON embedded here; the gains tables are pre-rendered and only switched.
    '''
    weekly = v.gains[v.gains['week'].isin(v.matured['week'])]
    frames = {'bmk': v.bgains, **{f'{w:%Y-%m-%d}': t for w, t in weekly.groupby('week')}}
    frames = {k: t for k, t in frames.items() if len(t)}
    if not frames:
        return ''
    names = {k: 'benchmark month' if k == 'bmk' else f'week of {k}' for k in frames}
    keys = [k for k in frames if k != 'bmk'][::-1] + ['bmk'] * ('bmk' in frames)     # latest week first
    week, base = keys[0], 'bmk' if 'bmk' in frames else keys[0]

    bins = sorted(set().union(*(t['bin'] for t in frames.values())), key=lambda b: (b == MISSING, b))
    ranges = pd.concat(list(frames.values())).drop_duplicates('bin').set_index('bin')['bin_range']
    num = lambda col: [round(float(x), 4) if np.isfinite(x) else None for x in col]
    weeks = {k: {'name': names[k], 'short': 'bmk' if k == 'bmk' else k[5:],
                 **{key: num(t.set_index('bin').reindex(bins)[col])
                    for key, col in [('lift', 'lift'), ('rate', 'ResponseRate')]}}
             for k, t in frames.items()}
    data = {'bins': [{'label': 'NA' if b == MISSING else str(b + 1), 'range': ranges[b]} for b in bins],
            'max': max([x for w in weeks.values() for x in w['lift'] if x is not None] + [1.2]), 'weeks': weeks}

    options = lambda sel: ''.join(f'<option value="{k}"{" selected" * (k == sel)}>{names[k]}</option>' for k in keys)
    tools = (f'<div class="tools"><label>Week <select>{options(week)}</select></label>'
             f'<label>compare with <select>{options(base)}</select></label></div>')
    tables = ''.join(f'<div class="gt" data-key="{k}"{" hidden" * (k != week)}><h3>Gains table · {names[k]}</h3>'
                     f'{html_table(frames[k], GAINS_TABLE)}</div>' for k in keys)
    payload = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    return card('Lift and gains by week', 'lift per benchmark prob decile, 1 = lowest risk · pick a week and what to compare it with',
                f'{tools}<div class="cards"><div class="liftchart"></div><div class="lifttable tscroll"></div></div>'
                f'{tables}<script type="application/json">{payload}</script>', wide=True)


def drilldown(v, a, dd_id, hidden):
    '''Latest-week bin shares of one attribute against the benchmark.'''
    r = v.issues.set_index('attribute').loc[a]
    b = v.bins[(v.bins['attribute'] == a) & (v.bins['week'] == v.bins['week'].max())]
    b = b.sort_values('bin', key=missing_last())
    cats = ['NA' if k == MISSING else str(k + 1) for k in b['bin']]
    chart = column_chart(cats, [('benchmark', b['bmk_ratio'], '0'), ('this week', b['ratio'], '1')], PCT, '{:.0%}',
                         tips=[f'{x}\n{fmt(p, F4)}\tPSI contribution' for x, p in zip(b['bin_range'], b['psi'])], h=180)
    iv = f'IV {fmt(r["bmk_iv"])} → {fmt(r["iv"])} ({fmt(r["iv_change"], "{:+.0%}")})' if pd.notna(r['bmk_iv']) else 'IV not matured yet'
    return (f'<div class="dd" id="{dd_id}"{" hidden" if hidden else ""}><h3>{esc(a)} {chip(r["status"])}</h3>'
            f'<p class="lead">PSI {fmt(r["psi"])} this week · max {fmt(r["psi_max"])} over the last {RECENT} weeks · {iv} · '
            f'missing {fmt(r["miss"], PCT)} (benchmark {fmt(r["miss_bmk"], PCT)})</p><div class="cards">'
            f'<div>{legend([("k0", "benchmark"), ("k1", "this week")])}{chart}</div>'
            + html_table(b, [('bin_range', 'bin', None), ('bmk_ratio', 'benchmark', PCT), ('ratio', 'this week', PCT), ('psi', 'PSI', F4)])
            + '</div></div>')


def attribute_section(v, mid):
    t = v.issues
    if not len(t):
        return '<h2>Attributes</h2><p class="note">No attribute comparisons yet.</p>'
    dd = {a: f'{mid}-dd{i}' for i, a in enumerate(t['attribute'])}
    rows = [f' data-dd="{dd[a]}" data-flag="{int(s > 0)}" data-name="{esc(a.lower())}"' for a, s in zip(t['attribute'], t['status'])]
    psi = v.attr.pivot(index='attribute', columns='week', values='psi').reindex(t['attribute'])
    trend = {a: sparkline(row.values[-RECENT * 2:]) for a, row in psi.iterrows()}
    cols = [('status', 'status', chip), ('attribute', 'attribute', None), ('psi', 'PSI this week', F3),
            ('psi_max', f'max PSI {RECENT}w', F3), ('weeks_flagged', f'weeks ≥ {PSI_WATCH:.2f}', INT),
            ('attribute', 'PSI trend', lambda a: trend[a]), ('bmk_iv', 'bmk IV', F3), ('iv', f'IV last {IV_WEEKS} matured', F3),
            ('iv_change', 'IV change', '{:+.0%}'), ('miss_bmk', 'missing bmk', PCT), ('miss', 'missing now', PCT)]
    tools = ('<div class="tools filter"><input type="search" placeholder="Find attribute" aria-label="Find attribute">'
             '<label><input type="checkbox"> flagged only</label><span>click a row for its bins</span></div>')
    panels = ''.join(drilldown(v, a, dd[a], i > 0) for i, a in enumerate(t['attribute']))
    issues = card('Attribute issues', f'most severe first · PSI of the latest week · IV averaged over the last {IV_WEEKS} matured weeks',
                  f'<div>{tools}{html_table(t, cols, sortable=True, row_attrs=rows)}</div><div class="ddbox">{panels}</div>', wide=True)

    link = [f' data-dd="{dd[a]}"' for a in t['attribute']]
    top = v.bins.loc[v.bins.groupby(['attribute', 'week'])['psi'].idxmax()].set_index(['attribute', 'week'])
    miss = v.bins[v.bins['bin'] == MISSING].set_index(['attribute', 'week'])
    tips, cls = [], []
    for a, row in psi.iterrows():
        tips.append([f'{a} · week of {w:%Y-%m-%d}\n{fmt(x)}\tPSI · {STATUS[psi_status(x)][0]}\n'
                     f'{fmt(top.at[(a, w), "bmk_ratio"], PCT)} → {fmt(top.at[(a, w), "ratio"], PCT)}\tlargest move {top.at[(a, w), "bin_range"]}\n'
                     f'{fmt(miss.at[(a, w), "ratio"] if (a, w) in miss.index else 0, PCT)}\tmissing (bmk '
                     f'{fmt(miss.at[(a, w), "bmk_ratio"] if (a, w) in miss.index else 0, PCT)})' if pd.notna(x) else ''
                     for w, x in row.items()])
        cls.append([seq_cls(x, PSI_BREAKS) for x in row.values])
    body = [issues, card('Attribute PSI by week', 'each attribute vs its benchmark bins; rows in issue order, click a name for its bins',
                         PSI_LEGEND + heatmap(t['attribute'].tolist(), list(psi.columns), cls, tips, link),
                         pivot_table(psi, F3, 'attribute'), wide=True)]

    ivs = v.attr[v.attr['week'].isin(v.matured['week'])].pivot(index='attribute', columns='week', values='iv_shift').reindex(t['attribute'])
    if ivs.shape[1]:
        cell = v.attr.set_index(['attribute', 'week'])
        tips = [[f'{a} · week of {w:%Y-%m-%d}\n{fmt(x, "{:+.3f}")}\tIV shift\n{fmt(cell.at[(a, w), "iv"])}\tIV this week\n'
                 f'{fmt(cell.at[(a, w), "bmk_iv"])}\tbenchmark IV' if pd.notna(x) else '' for w, x in row.items()]
                for a, row in ivs.iterrows()]
        cls = [[div_cls(x, [0.01, 0.03, 0.05]) for x in row] for row in ivs.values]
        body.append(card('Attribute IV shift by week', 'IV of the matured week minus the benchmark IV, same bins',
                         div_legend([0.01, 0.03, 0.05], '{}', 'gain', 'drop') + heatmap(t['attribute'].tolist(), list(ivs.columns), cls, tips, link),
                         pivot_table(ivs, '{:+.3f}', 'attribute'), wide=True))
    return '<h2>Attributes</h2><div class="cards">' + ''.join(body) + '</div>'


def model_panel(v, mid):
    b = v.bmk
    lead = (f'<div class="lead"><b>{esc(v.name)}</b>{chip(v.status)}'
            f'<span>data {v.daily["date"].min():%Y-%m-%d} → {v.daily["date"].max():%Y-%m-%d}</span>'
            f'<span>benchmark {b["bmk_start"]:%Y-%m-%d} → {b["bmk_end"]:%Y-%m-%d} · {b["n_days"]:.0f} days · '
            f'{b["n"]:,.0f} scored · {b["n_labelled"]:,.0f} labelled</span></div>')
    return (f'<section class="panel" id="{mid}" hidden>{lead}{kpis(v)}{volume_section(v)}{score_section(v)}'
            f'{perf_section(v)}{attribute_section(v, mid)}</section>')


def overview_panel(views):
    t = pd.DataFrame([{
        'model': v.name, 'status': v.status, 'through': v.daily['date'].max(),
        'vol': v.vol, 'vol_chg': v.vol / v.vol_bmk - 1,
        'score': v.last['avg_score'], 'score_chg': v.last['avg_score'] - v.bmk['avg_score'],
        'psi': v.last['score_psi'],
        'auc': v.mat['auc'] if v.mat is not None else np.nan, 'auc_chg': -v.auc_drop,
        'ks': v.mat['ks'] if v.mat is not None else np.nan,
        'ks_chg': v.mat['ks'] - v.bmk['ks'] if v.mat is not None else np.nan,
        'n_shift': int((v.issues['status'] == 2).sum()) if len(v.issues) else 0,
        'n_watch': int((v.issues['status'] == 1).sum()) if len(v.issues) else 0,
        'top': ', '.join(v.issues.loc[v.issues['status'] > 0, 'attribute'][:3]) if len(v.issues) else ''}
        for v in views])
    cols = [('model', 'model', None), ('status', 'status', chip), ('through', 'data through', None),
            ('vol', 'daily volume', INT), ('vol_chg', 'vs bmk', '{:+.1%}'), ('score', 'avg score', F1),
            ('score_chg', 'vs bmk', '{:+.1f}'), ('psi', 'score PSI', F3), ('auc', 'AUC', F3), ('auc_chg', 'vs bmk', '{:+.3f}'),
            ('ks', 'KS', F3), ('ks_chg', 'vs bmk', '{:+.3f}'), ('n_shift', 'attrs shift', INT), ('n_watch', 'attrs watch', INT),
            ('top', 'top issue attributes', None)]
    health = card('Model health', 'latest week for volume, score and PSI · latest matured week for AUC and KS · click a model to open it',
                  html_table(t, cols, sortable=True, row_attrs=[f' data-tab="m{i}"' for i in range(len(views))]), wide=True)

    weeks = sorted(set().union(*(v.perf['week'] for v in views)))
    link = [f' data-tab="m{i}"' for i in range(len(views))]
    psi = pd.DataFrame({v.name: v.perf.set_index('week')['score_psi'] for v in views}).reindex(weeks).T
    auc = pd.DataFrame({v.name: v.matured.set_index('week')['auc'] - v.bmk['auc'] for v in views}).reindex(weeks).T
    psi_tips = [[f'{m} · week of {w:%Y-%m-%d}\n{fmt(x)}\tscore PSI · {STATUS[psi_status(x)][0]}' if pd.notna(x) else ''
                 for w, x in row.items()] for m, row in psi.iterrows()]
    auc_tips = [[f'{m} · week of {w:%Y-%m-%d}\n{fmt(x, "{:+.3f}")}\tAUC vs benchmark' if pd.notna(x) else ''
                 for w, x in row.items()] for m, row in auc.iterrows()]
    strips = [
        card('Score PSI by week', 'every model against its own benchmark month', PSI_LEGEND + heatmap(
            list(psi.index), weeks, [[seq_cls(x, PSI_BREAKS) for x in r] for r in psi.values], psi_tips, link),
            pivot_table(psi, F3, 'model'), wide=True),
        card('AUC change by week', 'matured weeks, AUC minus the benchmark AUC', div_legend(
            [0.01, AUC_WATCH, AUC_SHIFT], '{:.2f}', 'gain', 'drop') + heatmap(
            list(auc.index), weeks, [[div_cls(x, [0.01, AUC_WATCH, AUC_SHIFT]) for x in r] for r in auc.values], auc_tips, link),
            pivot_table(auc, '{:+.3f}', 'model'), wide=True)]
    n_att = sum(v.status > 0 for v in views)
    lead = f'<div class="lead"><b>{len(views)} model{"s" * (len(views) > 1)}</b><span>{n_att} need attention</span></div>'
    return f'<section class="panel" id="overview">{lead}<div class="cards">{health}{"".join(strips)}</div></section>'


# ---------------------------------------------------------------- page

def build_report(tables, path, title='Model tracking report'):
    '''
    Write the HTML report of every model in the tables to path.
    tables: dict returned by model_tracker.run() / track_models(), or the folder they were saved to
    '''
    tables = load_tables(tables)
    views = [model_view(tables, m) for m in tables['benchmark']['model']]
    through = max(v.daily['date'].max() for v in views)
    nav = '<button type="button" data-tab="overview">Overview</button>' + ''.join(
        f'<button type="button" data-tab="m{i}" class="st{v.status}"><i>{STATUS[v.status][1]}</i> {esc(v.name)}</button>'
        for i, v in enumerate(views))
    foot = (f'<p><b>Benchmark</b>: the first month of each model\'s data. A day or week inside that month is compared with the '
            f'rest of the month; later periods with the whole month. Bins are benchmark deciles, NA = missing.</p>'
            f'<p><b>Flags</b>: PSI watch ≥ {PSI_WATCH}, shift ≥ {PSI_SHIFT} · AUC drop watch ≥ {AUC_WATCH}, shift ≥ {AUC_SHIFT} · '
            f'attribute IV drop ≥ {IV_DROP:.0%} over the last {IV_WEEKS} matured weeks (benchmark IV ≥ {IV_MIN}) → watch · '
            f'a week is matured once ≥ {MATURE:.0%} of it is labelled · model status = worse of score PSI and AUC.</p>'
            f'<p>Tables fingerprint <code>{fingerprint(tables)}</code> · rebuild with '
            f'<code>python -m modeltool.tracking_report &lt;table folder&gt; -o report.html</code></p>')
    page = Template(PAGE).substitute(
        title=esc(title), meta=f'data through {through:%Y-%m-%d}', nav=nav, foot=foot,
        body=overview_panel(views) + ''.join(model_panel(v, f'm{i}') for i, v in enumerate(views)))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    return path


DARK = ('color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;'
        '--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;--context:#6b6a65;--band:rgba(57,135,229,.12);'
        '--pos:#0ca30c;--neg:#e66767;--q1:#1b3a61;--q2:#1c5cab;--q3:#2a78d6;--q4:#5598e7;--q5:#9ec5f4;'
        '--dn3:#f3a59f;--dn2:#d0504c;--dn1:#6e2e2b;--d0:#383835;--dp1:#25528c;--dp2:#3d86d9;--dp3:#9ec5f4')

CSS = ''':root{color-scheme:light;--sans:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--accent:#2a78d6;--context:#898781;--band:rgba(42,120,214,.07);
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--pos:#006300;--neg:#d03b3b;
--q1:#cde2fb;--q2:#86b6ef;--q3:#3987e5;--q4:#1c5cab;--q5:#0d366b;
--dn3:#a82a2a;--dn2:#e5625c;--dn1:#f6c3bd;--d0:#f0efec;--dp1:#b7d3f6;--dp2:#5598e7;--dp3:#1c5cab}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){''' + DARK + '''}}
:root[data-theme="dark"]{''' + DARK + '''}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 var(--sans)}
header{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--border)}
.bar{max-width:1240px;margin:0 auto;padding:8px 16px;display:flex;gap:12px;align-items:center}
h1{font-size:17px;margin:0}.meta{flex:1;color:var(--ink-2);font-size:12px}
#theme{border:1px solid var(--border);background:var(--surface);color:var(--ink);border-radius:6px;padding:2px 8px;cursor:pointer}
nav.bar{padding-top:0;gap:4px;overflow-x:auto}
nav button{font:inherit;font-size:13px;border:0;background:none;color:var(--ink-2);padding:6px 10px;border-radius:6px;cursor:pointer;white-space:nowrap}
nav button i{font-style:normal;font-size:10px}
nav button[aria-selected=true]{background:var(--surface);color:var(--ink);box-shadow:inset 0 0 0 1px var(--border);font-weight:600}
main{max-width:1240px;margin:0 auto;padding:16px}
h2{font-size:15px;margin:28px 0 10px}h3{font-size:14px;margin:0 0 4px;display:flex;gap:10px;align-items:center}
.lead{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;color:var(--ink-2);font-size:12.5px;margin:0 0 14px}
.lead b{font-size:18px;color:var(--ink)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));gap:16px}
.cards.c3{grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr))}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:0;min-width:0}
.wide{grid-column:1/-1}
figcaption{margin-bottom:8px}figcaption b{display:block;font-size:13.5px}figcaption span{color:var(--ink-2);font-size:12px}
svg{display:block;width:100%;height:auto;font:11px var(--sans)}
svg text{fill:var(--muted)}
.gl{stroke:var(--grid);stroke-width:1}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.ln1{stroke:var(--accent)}.ln0{stroke:var(--context)}
.dt{stroke:var(--surface);stroke-width:2}.dt1,.b1{fill:var(--accent)}.dt0,.b0{fill:var(--context)}
.ref{stroke:var(--ink-2);stroke-width:1}.ref-w{stroke:var(--warn);stroke-width:1.5}.ref-s{stroke:var(--crit);stroke-width:1.5}
.band{fill:var(--band)}.hit,.hitc{fill:transparent}.hitc:hover{fill:var(--band)}
.xh{stroke:var(--axis);opacity:0}.hb:hover .xh{opacity:1}
.cell:hover{stroke:var(--ink);stroke-width:1.5}
.q1{fill:var(--q1);background:var(--q1)}.q2{fill:var(--q2);background:var(--q2)}.q3{fill:var(--q3);background:var(--q3)}
.q4{fill:var(--q4);background:var(--q4)}.q5{fill:var(--q5);background:var(--q5)}
.dn3{fill:var(--dn3);background:var(--dn3)}.dn2{fill:var(--dn2);background:var(--dn2)}.dn1{fill:var(--dn1);background:var(--dn1)}
.d0{fill:var(--d0);background:var(--d0)}.dp1{fill:var(--dp1);background:var(--dp1)}.dp2{fill:var(--dp2);background:var(--dp2)}
.dp3{fill:var(--dp3);background:var(--dp3)}
.k1{background:var(--accent)}.k0{background:var(--context)}
[data-dd],[data-tab]{cursor:pointer}
g[data-dd]:hover text,g[data-tab]:hover text{fill:var(--ink);text-decoration:underline}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink-2);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:12px;height:10px;border-radius:2px}.lk{display:inline-block;width:14px;height:2px;border-radius:1px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(170px,100%),1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.tile .lbl{font-size:12px;color:var(--ink-2)}.tile .val{font-size:26px;font-weight:600;line-height:1.25}
.note{font-size:11.5px;color:var(--muted)}
.dlt{font-size:12px;color:var(--ink-2)}.dlt.pos{color:var(--pos)}.dlt.neg{color:var(--neg)}
.chip{display:inline-flex;align-items:center;gap:4px;font-size:12px;font-weight:400;color:var(--ink)}.chip i{font-style:normal;font-size:10px}
.st0 i{color:var(--good)}.st1 i{color:var(--warn)}.st2 i{color:var(--crit)}
.tscroll{overflow:auto;max-height:440px}
table{border-collapse:collapse;font-size:12px;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);color:var(--ink-2);font-weight:600}
th.l,td.l{text-align:left}
table.sortable th{cursor:pointer}th[data-dir=asc]::after{content:" ▲"}th[data-dir=desc]::after{content:" ▼"}
tr[data-dd]:hover td,tr[data-tab]:hover td,tr.on td{background:var(--band)}
svg.spark{display:inline-block;vertical-align:middle;width:90px;height:20px}
details summary{cursor:pointer;color:var(--ink-2);font-size:12px;margin-top:8px}
.tools{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px;font-size:12.5px;color:var(--ink-2)}
.tools label{display:inline-flex;gap:6px;align-items:center}
.tools select{font:inherit;padding:3px 6px;border:1px solid var(--border);border-radius:6px;background:var(--page);color:var(--ink)}
.gt h3{margin:16px 0 6px}
.tools input[type=search]{font:inherit;padding:4px 8px;border:1px solid var(--border);border-radius:6px;background:var(--page);color:var(--ink)}
.dd{border-top:1px solid var(--grid);margin-top:14px;padding-top:12px}
footer{max-width:1240px;margin:0 auto;padding:8px 16px 40px;color:var(--ink-2);font-size:12px}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.14);padding:6px 9px;font-size:12px;max-width:380px}
#tip div{display:flex;align-items:center;gap:6px;white-space:nowrap}#tip .th{color:var(--ink-2)}
#tip b{font-variant-numeric:tabular-nums}#tip .key{width:12px;height:2px;border-radius:1px}'''

JS = '''const root = document.documentElement, tip = document.getElementById('tip');
function showTab(id) {
  if (!document.getElementById(id)) id = 'overview';
  document.querySelectorAll('.panel').forEach(p => p.hidden = p.id !== id);
  document.querySelectorAll('nav [data-tab]').forEach(b => b.setAttribute('aria-selected', b.dataset.tab === id));
}
function sortBy(th) {
  const table = th.closest('table'), i = [...th.parentNode.children].indexOf(th);
  const dir = th.dataset.dir === 'desc' ? 'asc' : 'desc', sign = dir === 'asc' ? 1 : -1;
  th.parentNode.querySelectorAll('th').forEach(h => delete h.dataset.dir);
  th.dataset.dir = dir;
  const val = r => 'v' in r.cells[i].dataset ? Number(r.cells[i].dataset.v) : r.cells[i].textContent.toLowerCase();
  table.tBodies[0].append(...[...table.tBodies[0].rows].sort((a, b) => sign * ((val(a) > val(b)) - (val(a) < val(b)))));
}
showTab(location.hash.slice(1));
document.addEventListener('click', e => {
  const tab = e.target.closest('[data-tab]');
  if (tab) {
    e.preventDefault(); showTab(tab.dataset.tab);
    history.replaceState(null, '', '#' + tab.dataset.tab); scrollTo(0, 0); return;
  }
  const th = e.target.closest('table.sortable th');
  if (th) return sortBy(th);
  const link = e.target.closest('[data-dd]');
  if (link) {
    const panel = document.getElementById(link.dataset.dd), card = panel.closest('.card');
    card.querySelectorAll('.dd').forEach(p => p.hidden = p !== panel);
    card.querySelectorAll('tr[data-dd]').forEach(r => r.classList.toggle('on', r.dataset.dd === link.dataset.dd));
    panel.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }
});
document.querySelectorAll('.tools.filter').forEach(tools => {
  const q = tools.querySelector('input[type=search]'), flag = tools.querySelector('input[type=checkbox]');
  const rows = tools.parentElement.querySelectorAll('tbody tr');
  const apply = () => rows.forEach(r => r.hidden =
    (flag.checked && r.dataset.flag === '0') || !r.dataset.name.includes(q.value.trim().toLowerCase()));
  q.addEventListener('input', apply); flag.addEventListener('change', apply);
});
document.addEventListener('pointerover', e => {
  const el = e.target.closest('[data-tip]');
  if (!el || !el.dataset.tip) { tip.hidden = true; return; }
  tip.replaceChildren(...el.dataset.tip.split('\\n').map((line, i) => {
    const row = document.createElement('div'), [value, label, key] = line.split('\\t');
    if (i === 0) { row.className = 'th'; row.textContent = value; return row; }
    if (key) { const k = document.createElement('span'); k.className = 'key k' + key; row.append(k); }
    const b = document.createElement('b'); b.textContent = value; row.append(b);
    if (label) row.append(label);
    return row;
  }));
  tip.hidden = false;
});
document.addEventListener('pointermove', e => {
  if (tip.hidden) return;
  tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (e.clientY + 14 + tip.offsetHeight > innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 14) + 'px';
});
document.addEventListener('pointerout', e => { if (!e.relatedTarget) tip.hidden = true; });
const NS = 'http://www.w3.org/2000/svg';
const f2 = v => v == null ? '–' : v.toFixed(2), pct = v => v == null ? '–' : (100 * v).toFixed(1) + '%';
const signed = v => v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(2);
const change = (a, b, i) => a.lift[i] == null || b.lift[i] == null ? null : a.lift[i] - b.lift[i];
function node(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
function bar(x, y0, w, y1) {
  const r = Math.min(4, w / 2, y0 - y1);
  return 'M' + x + ',' + y0 + 'V' + (y1 + r) + 'Q' + x + ',' + y1 + ' ' + (x + r) + ',' + y1 +
    'H' + (x + w - r) + 'Q' + (x + w) + ',' + y1 + ' ' + (x + w) + ',' + (y1 + r) + 'V' + y0 + 'Z';
}
// Lift per bin: comparison (gray) next to the selected week (accent); same geometry as column_chart() in Python.
// The y axis spans every week's lift, so flipping weeks never rescales it.
function liftChart(d, a, b) {
  const W = 640, H = 220, L = 52, R = 12, T = 14, bot = H - 26, raw = d.max / 4;
  const p = Math.pow(10, Math.floor(Math.log10(raw))), step = p * [1, 2, 5, 10].find(m => raw <= p * m);
  const top = Math.ceil(d.max / step) * step, Y = v => bot - v / top * (bot - T);
  const svg = node('svg', {viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'lift by bin'});
  for (let i = 0; i * step <= top + 1e-9; i++)
    svg.append(node('line', {class: 'gl', x1: L, x2: W - R, y1: Y(i * step), y2: Y(i * step)}),
               node('text', {x: L - 6, y: Y(i * step) + 3.5, 'text-anchor': 'end'}, (i * step).toFixed(step < 1 ? 1 : 0)));
  svg.append(node('line', {class: 'ref', x1: L, x2: W - R, y1: Y(1), y2: Y(1)}),
             node('text', {x: L + 4, y: Y(1) - 4}, 'average'));   // left end: the low-risk bins sit below 1
  const slot = (W - L - R) / d.bins.length, bw = Math.min(24, (slot * 0.7 - 2) / 2);
  d.bins.forEach((bin, i) => {
    const x0 = L + i * slot + (slot - 2 * bw - 2) / 2;
    [b, a].forEach((s, j) => {
      if (s.lift[i] > 0) svg.append(node('path', {class: 'b' + j, d: bar(x0 + j * (bw + 2), bot, bw, Y(s.lift[i]))}));
    });
    const tip = [bin.label + ' · ' + bin.range, f2(a.lift[i]) + '\\t' + a.name + '\\t1', f2(b.lift[i]) + '\\t' + b.name + '\\t0',
                 signed(change(a, b, i)) + '\\tlift change', pct(b.rate[i]) + ' → ' + pct(a.rate[i]) + '\\tbad rate'];
    svg.append(node('text', {x: L + (i + 0.5) * slot, y: H - 8, 'text-anchor': 'middle'}, bin.label),
               node('rect', {class: 'hitc', x: L + i * slot, y: T, width: slot, height: bot - T, 'data-tip': tip.join('\\n')}));
  });
  return svg;
}
function liftTable(d, a, b) {
  const t = document.createElement('table'), head = t.createTHead().insertRow(), body = t.createTBody();
  const cells = (row, values, tag) => values.forEach((v, j) => {
    const c = tag ? row.appendChild(document.createElement(tag)) : row.insertCell();
    c.textContent = v; if (j < 2) c.className = 'l';
  });
  cells(head, ['bin', 'prob bin', 'lift ' + b.short, 'lift ' + a.short, 'change', 'bad rate ' + b.short, 'bad rate ' + a.short], 'th');
  d.bins.forEach((bin, i) => cells(body.insertRow(), [bin.label, bin.range, f2(b.lift[i]), f2(a.lift[i]),
    signed(change(a, b, i)), pct(b.rate[i]), pct(a.rate[i])]));
  return t;
}
document.querySelectorAll('.liftchart').forEach(chart => {
  const card = chart.closest('.card'), d = JSON.parse(card.querySelector('script[type="application/json"]').textContent);
  const [week, base] = card.querySelectorAll('select');
  const draw = () => {
    const a = d.weeks[week.value], b = d.weeks[base.value], legend = document.createElement('div');
    legend.className = 'legend';
    [[b, 'k0'], [a, 'k1']].forEach(([s, k]) => {
      const span = document.createElement('span'), key = document.createElement('i');
      key.className = 'sw ' + k; span.append(key, s.name); legend.append(span);
    });
    chart.replaceChildren(legend, liftChart(d, a, b));
    card.querySelector('.lifttable').replaceChildren(liftTable(d, a, b));
    card.querySelectorAll('.gt').forEach(g => g.hidden = g.dataset.key !== week.value);
  };
  week.addEventListener('change', draw); base.addEventListener('change', draw); draw();
});
document.getElementById('theme').addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
});'''

PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>''' + CSS.replace('$', '$$') + '''</style>
</head>
<body>
<header><div class="bar"><h1>$title</h1><span class="meta">$meta</span>
<button id="theme" type="button" aria-label="Toggle dark mode">◐</button></div>
<nav class="bar">$nav</nav></header>
<main>$body</main>
<footer>$foot</footer>
<div id="tip" hidden></div>
<script>''' + JS.replace('$', '$$') + '''</script>
</body>
</html>
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild the HTML tracking report from a saved table folder.')
    parser.add_argument('tables', help='folder with the model_tracker csv tables')
    parser.add_argument('-o', '--out', help='output html (default: <tables>/tracking_report.html)')
    parser.add_argument('--title', default='Model tracking report')
    args = parser.parse_args()
    print(build_report(args.tables, args.out or os.path.join(args.tables, 'tracking_report.html'), args.title))
