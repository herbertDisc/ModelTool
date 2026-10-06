#########################################################
# Description: Render a model_reporter workbook (model_N.xlsx) as an
#              interactive single-file HTML model card
# Auther: Xinyue He
# Created date: 2026.10
#########################################################

'''
Reads the workbook written by model_report.model_reporter (sheets Summary,
KS_by_first_seg, Bivar, means_table, PSI, plus the embedded charts) and fills
the fixed model-card template, the layout of
model/potf_cl_sms_v1/single_rd_search_potf_sms_v1_iter78/report/peru_model_复贷_sms_v1.html.

Usage:
    from modeltool.model_card_html import render_model_card
    render_model_card('report/model_4.xlsx', 'report/model_4.html',
                      project='potf_cl_sms_v1', run='potf_sms_v1_iter78',
                      windows={'dev': ('2026-05-01', '2026-07-05'),
                               'oot': ('2026-07-06', '2026-07-31'),
                               'val': ('2026-05-01', '2026-07-05')})

    python -m modeltool.model_card_html report/model_4.xlsx -o report/model_4.html \
        --project potf_cl_sms_v1 --run potf_sms_v1_iter78 \
        --window dev=2026-05-01,2026-07-05 --window oot=2026-07-06,2026-07-31
'''

import argparse
import base64
import html
import json
import os
import re

import openpyxl


SEG_NAMES = {'dev': 'Development', 'oot': 'Out-of-time', 'val': 'Validation'}


def _rows(ws):
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _find(rows, pred, start=0):
    '''first (row, col) at/after row `start` whose cell satisfies pred'''
    for i in range(start, len(rows)):
        for j, v in enumerate(rows[i]):
            if pred(v):
                return i, j
    return None


def _png(img):
    return 'data:image/png;base64,' + base64.b64encode(img._data()).decode()


def _block(rows, hr, hc):
    '''read a table whose header row is `hr`; its row labels sit in column `hc`
    and the data columns run right of it until the first empty header cell'''
    ncol = 0
    while hc + 1 + ncol < len(rows[hr]) and rows[hr][hc + 1 + ncol] is not None:
        ncol += 1
    out = []
    for i in range(hr + 1, len(rows)):
        lab = rows[i][hc] if hc < len(rows[i]) else None
        if lab is None:
            break
        out.append([lab] + rows[i][hc + 1:hc + 1 + ncol])
    return out


def _gains_row(r):
    # bin, Total_Int .. KS, lift, cum_lift -> always 14 cells
    r = list(r) + [None] * (14 - len(r))
    return r[:14]


def read_summary(ws):
    rows = _rows(ws)
    hr, hc = _find(rows, lambda v: v == 'seg')
    pop_hdr = rows[hr][hc:hc + 4]
    target = pop_hdr[2]
    pop = [{'seg': r[0], 'total': r[1], 'bad': r[2], 'rate': r[3]} for r in _block(rows, hr, hc)]

    hr, hc = _find(rows, lambda v: v == 'AUC')
    hc -= 1
    perf = [{'seg': r[0], 'auc': r[1], 'ks': r[2]} for r in _block(rows, hr, hc)]

    hr, hc = _find(rows, lambda v: v in ('Attr', 'attr'))
    imp = [{'attr': r[0], 'imp': r[1]} for r in _block(rows, hr, hc)]

    img = _png(ws._images[0]) if ws._images else ''
    return target, pop, perf, imp, img, len(ws._images)


def read_gains(ws):
    rows = _rows(ws)
    gains = {}
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if not isinstance(v, str):
                continue
            m = re.match(r'Benchmark Segment:(.+?) Gains table', v)
            if m:
                kind, seg = 'bmk', m.group(1)
            else:
                m = re.match(r'Segment (.+?) Gains table, bin cut by (the )?benchmark', v)
                kind, seg = ('bench', m.group(1)) if m else (None, None)
                if not m:
                    m = re.match(r'Segment (.+?) Gains table, bin cut by the segment', v)
                    kind, seg = ('seg', m.group(1)) if m else (None, None)
            if kind is None:
                continue
            hr, hc = _find(rows, lambda x: x == 'Total_Int', i + 1)
            # the header cell may sit a few columns right of the title (benchmark block)
            hc = next(c for c in range(j, len(rows[hr])) if rows[hr][c] == 'Total_Int')
            tbl = [_gains_row(r) for r in _block(rows, hr, hc - 1)]
            if kind == 'bmk':
                gains[seg] = {'bench': tbl, 'seg': tbl}
                bmk = seg
            else:
                gains.setdefault(seg, {})[kind] = tbl
    return bmk, gains


def read_bivar(ws, segs):
    rows = _rows(ws)
    ns = len(segs)
    col = lambda k: rows[k][1] if k < len(rows) and len(rows[k]) > 1 else None

    def grp_row(i):
        # an attribute title in column B is followed (after blank/segment rows) by the 'grp' header
        if not isinstance(col(i), str) or col(i) == 'grp':
            return None
        for k in range(i + 1, min(i + 6, len(rows))):
            if col(k) == 'grp':
                return k
            if col(k) is not None:
                return None
        return None

    bivar, starts = [], []
    i = 0
    while i < len(rows):
        g = grp_row(i)
        if g is None:
            i += 1
            continue
        body = []
        k = g + 1
        while k < len(rows) and col(k) is not None and grp_row(k) is None:
            body.append(rows[k][1:2 + 3 * ns])
            k += 1
        bivar.append({'var': col(i), 'rows': body})
        starts.append(i)
        i = k
    # each chart is anchored next to its table: give it to the last header at/above its anchor row
    imgs = {}
    for im in ws._images:
        r0 = im.anchor._from.row
        idx = max((n for n, st in enumerate(starts) if st <= r0 + 1), default=0)
        imgs.setdefault(bivar[idx]['var'], _png(im))
    return bivar, imgs, len(ws._images)


def read_psi(ws):
    rows = _rows(ws)
    hr, hc = _find(rows, lambda v: v == 'variable')
    segs = rows[hr][hc + 1:]
    segs = [s for s in segs if s is not None]
    psi = {r[0]: dict(zip(segs, r[1:])) for r in _block(rows, hr, hc)}
    return psi


def read_means(ws):
    rows = _rows(ws)
    hr, hc = _find(rows, lambda v: v == 'seg')
    return [r[hc:hc + 19] for r in rows[hr + 1:] if r[hc] is not None]


def _win(w):
    if not w:
        return ''
    if isinstance(w, str):
        return w
    return f'{w[0]} → {w[1]}'


def _display(p):
    rel = os.path.relpath(p)
    return p if rel.startswith('..') else rel


def build_data(xlsx_path, windows=None):
    wb = openpyxl.load_workbook(xlsx_path)
    target, pop, perf, imp, summary_img, n_img = read_summary(wb['Summary'])
    segs = [p['seg'] for p in pop]
    ks_sheet = next(n for n in wb.sheetnames if n.startswith('KS'))
    bmk, gains = read_gains(wb[ks_sheet])
    bivar, bivar_img, n_bimg = read_bivar(wb['Bivar'], segs)
    windows = windows or {}
    for p in pop:
        p['win'] = _win(windows.get(p['seg']))
    D = {'pop': pop, 'perf': perf, 'imp': imp, 'gains': gains,
         'psi': read_psi(wb['PSI']), 'bivar': bivar, 'means': read_means(wb['means_table']),
         'summary_img': summary_img, 'bivar_img': bivar_img,
         'segs': segs, 'bmk': bmk, 'target': target}
    return D, wb.sheetnames, n_img + n_bimg


def render_model_card(xlsx_path, out_path=None, *, project='', run='', title=None, eyebrow=None,
                      windows=None, split_note=None, display_path=None):
    '''
    Render model_N.xlsx (from model_reporter) as the HTML model card.
    Input:
        xlsx_path: string
            workbook written by model_reporter
        out_path: string
            html output path; defaults to the workbook path with .html
        project: string
            shown after the title, e.g. 'potf_cl_sms_v1'
        run: string
            run name shown in the stamp, e.g. 'potf_sms_v1_iter78'
        title: string
            defaults to 'Round N' taken from model_N.xlsx
        eyebrow: string
            small caps line above the title
        windows: dict
            {seg: (start, end)} or {seg: 'start → end'} sample windows
        split_note: string
            appended to the population note, e.g. 'dev and val are a random 67/33 split ...'
        display_path: string
            workbook path printed in the footer; defaults to xlsx_path
    Output:
        out_path
    '''
    D, sheets, n_img = build_data(xlsx_path, windows)
    name = os.path.basename(xlsx_path)
    if title is None:
        m = re.search(r'model_(\d+)\.xlsx$', name)
        title = f'Round {m.group(1)}' if m else os.path.splitext(name)[0]
    n_attr = len(D['imp'])
    if eyebrow is None:
        eyebrow = f'Model card ({n_attr} attributes)'
    e = html.escape

    stamp = []
    if run:
        stamp.append(f'run <b>{e(run)}</b>')
    stamp += [f'attributes <b>{n_attr}</b>',
              f'target <b>{e(str(D["target"]))}</b>',
              f'segments <b>{" / ".join(map(e, D["segs"]))}</b>']
    # segments sharing a window are listed together, as in 'dev · val'
    by_win = {}
    for p in D['pop']:
        if p['win']:
            by_win.setdefault(p['win'], []).append(p['seg'])
    stamp += [f'{" · ".join(map(e, s))} <b>{e(w)}</b>' for w, s in by_win.items()]
    stamp = '<br>\n      '.join(['      ' + stamp[0]] + stamp[1:]) if stamp else ''

    data = json.dumps(D, ensure_ascii=False, allow_nan=False, default=str).replace('</', '<\\/')
    out = TEMPLATE
    for k, v in {
        'TITLE': e(title), 'EYEBROW': e(eyebrow), 'PROJECT': e(project),
        'XLSX_NAME': e(name), 'N_ATTR': str(n_attr), 'STAMP': stamp,
        'TARGET': e(str(D['target'])),
        'SPLIT_NOTE': f' · {e(split_note)}' if split_note else '',
        'BMK': e(D['bmk']), 'SEG_LIST': ' / '.join(map(e, D['segs'])),
        'XLSX_PATH': e(display_path or _display(xlsx_path)), 'SHEETS': e(', '.join(sheets)),
        'N_IMG': str(n_img),
    }.items():
        out = out.replace(f'%%{k}%%', v)
    out = out.replace('%%DATA%%', data)

    out_path = out_path or os.path.splitext(xlsx_path)[0] + '.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out_path


def _cli():
    ap = argparse.ArgumentParser(description='Render model_reporter workbooks as HTML model cards')
    ap.add_argument('xlsx', nargs='+')
    ap.add_argument('-o', '--out', help='output html (single workbook only)')
    ap.add_argument('--project', default='')
    ap.add_argument('--run', default='')
    ap.add_argument('--title')
    ap.add_argument('--eyebrow')
    ap.add_argument('--split-note')
    ap.add_argument('--window', action='append', default=[], metavar='SEG=START,END')
    a = ap.parse_args()
    windows = {}
    for w in a.window:
        seg, rng = w.split('=', 1)
        windows[seg] = tuple(rng.split(','))
    for x in a.xlsx:
        p = render_model_card(x, a.out if len(a.xlsx) == 1 else None, project=a.project, run=a.run,
                              title=a.title, eyebrow=a.eyebrow, windows=windows,
                              split_note=a.split_note)
        print(p)


# Template: the layout of peru_model_复贷_sms_v1.html; %%NAME%% tokens are filled by render_model_card
TEMPLATE = r'''<title>%%TITLE%% Model Card</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#eef1ee; --surface:#ffffff; --surface-2:#f6f8f6;
  --ink:#141a18; --ink-2:#3d4a45; --muted:#67766f; --line:#dbe2dd; --line-strong:#c3cec8;
  --accent:#0c6355; --accent-ink:#0c6355; --accent-soft:#dcece8;
  --good:#2c7048; --warn:#95701a; --crit:#a3382f; --clay:#a9552b;
  --good-soft:#dfeee4; --warn-soft:#f4ecd6; --crit-soft:#f6e0dd;
  --shadow:0 1px 2px rgba(20,26,24,.05), 0 8px 24px -16px rgba(20,26,24,.35);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0d1211; --surface:#141b19; --surface-2:#19211f;
    --ink:#e7edea; --ink-2:#bcc9c4; --muted:#8b9a94; --line:#25302d; --line-strong:#33403c;
    --accent:#4dbba8; --accent-ink:#63cbb8; --accent-soft:#12312c;
    --good:#6cc08d; --warn:#d7b25a; --crit:#e08b81; --clay:#d99164;
    --good-soft:#16301f; --warn-soft:#2e2716; --crit-soft:#331d1a;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#0d1211; --surface:#141b19; --surface-2:#19211f;
  --ink:#e7edea; --ink-2:#bcc9c4; --muted:#8b9a94; --line:#25302d; --line-strong:#33403c;
  --accent:#4dbba8; --accent-ink:#63cbb8; --accent-soft:#12312c;
  --good:#6cc08d; --warn:#d7b25a; --crit:#e08b81; --clay:#d99164;
  --good-soft:#16301f; --warn-soft:#2e2716; --crit-soft:#331d1a;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
h1,h2,h3{font-family:"Fraunces",Georgia,serif;font-weight:700;text-wrap:balance;margin:0}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
a{color:var(--accent-ink)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}

/* ---------- masthead ---------- */
.mast{border-bottom:1px solid var(--line);background:var(--surface)}
.mast-in{max-width:1180px;margin:0 auto;padding:38px 28px 30px;
  display:grid;grid-template-columns:minmax(0,1fr) auto;gap:28px;align-items:end}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted)}
h1{font-size:clamp(30px,4.4vw,46px);line-height:1.04;letter-spacing:-.015em;margin:10px 0 8px}
h1 .dim{color:var(--muted);font-weight:500}
.sub{color:var(--ink-2);max-width:62ch;font-size:15px}
.stamp{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);
  text-align:right;line-height:1.9;white-space:nowrap}
.stamp b{color:var(--ink);font-weight:600}

/* ---------- nav ---------- */
nav{position:sticky;top:0;z-index:30;background:color-mix(in srgb,var(--surface) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.nav-in{max-width:1180px;margin:0 auto;padding:0 28px;display:flex;gap:2px;overflow-x:auto}
nav a{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);text-decoration:none;padding:13px 12px;
  border-bottom:2px solid transparent;white-space:nowrap}
nav a:hover{color:var(--ink)}
nav a.on{color:var(--accent-ink);border-bottom-color:var(--accent)}

main{max-width:1180px;margin:0 auto;padding:0 28px 100px}
section{padding-top:56px;scroll-margin-top:56px}
.sec-head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  border-bottom:1px solid var(--line-strong);padding-bottom:12px;margin-bottom:24px}
.sec-head h2{font-size:23px;letter-spacing:-.01em}
.sec-head .note{color:var(--muted);font-size:13px;margin-left:auto}

.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}
.pad{padding:20px 22px}

/* ---------- kpi ---------- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:14px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
  box-shadow:var(--shadow);position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--accent);opacity:.85}
.kpi.g::before{background:var(--good)} .kpi.w::before{background:var(--warn)}
.kpi .k{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted)}
.kpi .v{font-family:"Fraunces",serif;font-size:31px;line-height:1.15;margin-top:4px;
  font-variant-numeric:tabular-nums}
.kpi .m{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.kpi .m.win{font-family:"IBM Plex Mono",monospace;font-size:10.5px;margin-top:2px;opacity:.85}

/* ---------- tables ---------- */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface);
  box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);font-weight:500;text-align:right;padding:11px 12px;white-space:nowrap;
  border-bottom:1px solid var(--line-strong);background:var(--surface-2);position:sticky;top:0}
th.l,td.l{text-align:left}
td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--surface-2)}
tr.tot td{border-top:1px solid var(--line-strong);font-weight:600;background:var(--surface-2)}
tr.tot:hover td{background:var(--surface-2)}
td.name{font-family:"IBM Plex Mono",monospace;font-size:12px}
.mark{color:var(--accent-ink);font-weight:600}

/* ---------- tabs / controls ---------- */
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.tab{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;
  padding:7px 14px;border-radius:999px;border:1px solid var(--line-strong);background:var(--surface);
  color:var(--muted);cursor:pointer}
.tab:hover{color:var(--ink)}
.tab.on{background:var(--accent);border-color:var(--accent);color:var(--surface)}
input[type=search]{font-family:"IBM Plex Mono",monospace;font-size:12.5px;padding:8px 12px;
  border:1px solid var(--line-strong);border-radius:999px;background:var(--surface);color:var(--ink);
  min-width:230px}
input[type=search]::placeholder{color:var(--muted)}

/* ---------- chips ---------- */
.chip{display:inline-flex;align-items:center;gap:5px;font-family:"IBM Plex Mono",monospace;
  font-size:10.5px;letter-spacing:.06em;padding:2px 8px;border-radius:999px;
  background:var(--surface-2);color:var(--muted);border:1px solid var(--line)}
.chip.good{background:var(--good-soft);color:var(--good);border-color:transparent}
.chip.warn{background:var(--warn-soft);color:var(--warn);border-color:transparent}
.chip.crit{background:var(--crit-soft);color:var(--crit);border-color:transparent}

/* ---------- importance ---------- */
.imp-row{display:grid;grid-template-columns:26px minmax(150px,1.4fr) minmax(90px,3fr) 58px 74px;
  gap:12px;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)}
.imp-row:last-child{border-bottom:none}
.imp-rank{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);text-align:right}
.imp-name{font-family:"IBM Plex Mono",monospace;font-size:12px;overflow:hidden;text-overflow:ellipsis}
.imp-name a{text-decoration:none;color:inherit;border-bottom:1px dotted var(--line-strong)}
.imp-name a:hover{color:var(--accent-ink)}
.bar{height:8px;border-radius:2px;background:var(--accent);opacity:.9;min-width:2px}
.bar-wrap{background:var(--surface-2);border-radius:2px}
.imp-val{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-2);text-align:right;
  font-variant-numeric:tabular-nums}

/* ---------- bivariate ---------- */
.biv-grid{display:flex;flex-direction:column;gap:18px}
.biv{background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow);
  overflow:hidden}
.biv-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:14px 18px;
  border-bottom:1px solid var(--line);background:var(--surface-2)}
.biv-head .rk{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted)}
.biv-head h3{font-family:"IBM Plex Mono",monospace;font-size:14px;font-weight:600;letter-spacing:-.01em}
.biv-body{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:0}
.biv-body .tblwrap{overflow-x:auto;border-right:1px solid var(--line)}
.biv-body table{font-size:12px}
.biv-body th{padding:9px 10px}
.biv-body td{padding:6px 10px}
.biv-img{padding:12px;display:flex;align-items:center;justify-content:center;background:#fff}
.biv-img img{max-width:100%;height:auto;display:block;border-radius:4px}
@media (max-width:880px){
  .biv-body{grid-template-columns:1fr}
  .biv-body .tblwrap{border-right:none;border-bottom:1px solid var(--line)}
  .mast-in{grid-template-columns:1fr}
  .stamp{text-align:left}
}
.hidden{display:none !important}
.figure{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px;
  box-shadow:var(--shadow);display:flex;justify-content:center}
.figure img{max-width:100%;height:auto}
.legend{display:flex;gap:8px 20px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin-top:12px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:26px;height:11px;border-radius:2px;border:1px solid var(--line);flex:none}
.foot{margin-top:70px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
.cap{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin:0 0 8px}
.count{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
</style>

<header class="mast">
  <div class="mast-in">
    <div>
      <div class="eyebrow">%%EYEBROW%%</div>
      <h1>%%TITLE%% <span class="dim">— %%PROJECT%%</span></h1>
      <p class="sub">Full contents of <span class="mono">%%XLSX_NAME%%</span>: population, gains
      and KS by segment, all %%N_ATTR%% attributes with importance, PSI, bivariate profiles and
      distribution statistics.</p>
    </div>
    <div class="stamp">
%%STAMP%%
    </div>
  </div>
</header>

<nav><div class="nav-in" id="nav">
  <a href="#population">Population</a>
  <a href="#performance">Performance</a>
  <a href="#gains">Gains tables</a>
  <a href="#importance">Importance</a>
  <a href="#psi">PSI</a>
  <a href="#bivariate">Bivariate</a>
  <a href="#stats">Distributions</a>
</div></nav>

<main>
  <section id="population">
    <div class="sec-head"><h2>Population overview</h2>
      <span class="note">Dependence rate = share of <span class="mono">%%TARGET%%</span> = 1%%SPLIT_NOTE%%</span></div>
    <div class="kpis" id="popKpis"></div>
    <div class="scroll" style="margin-top:18px"><table id="popTable"></table></div>
  </section>

  <section id="performance">
    <div class="sec-head"><h2>Model performance</h2>
      <span class="note">Measured on the first selected segment</span></div>
    <div class="kpis" id="perfKpis"></div>
    <div style="margin-top:20px">
      <p class="cap">Performance chart carried in the workbook</p>
      <div class="figure"><img id="summaryImg" alt="Model performance chart by segment"></div>
    </div>
  </section>

  <section id="gains">
    <div class="sec-head"><h2>Gains tables</h2>
      <span class="note">Deciles ordered high score → low score</span></div>
    <div class="tabs" id="gainTabs"></div>
    <div id="gainPanels"></div>
  </section>

  <section id="importance">
    <div class="sec-head"><h2>Attribute importance</h2>
      <span class="note">%%N_ATTR%% attributes · gain-normalised, sums to 1</span></div>
    <div class="card pad" id="impList"></div>
  </section>

  <section id="psi">
    <div class="sec-head"><h2>Population stability index</h2>
      <span class="note">%%BMK%% is the reference · &lt;0.10 stable · 0.10–0.25 shifted · &gt;0.25 unstable</span></div>
    <div class="tabs" style="align-items:center">
      <input type="search" id="psiSearch" placeholder="filter attribute…" aria-label="Filter PSI table">
      <span class="count" id="psiCount"></span>
    </div>
    <div class="scroll"><table id="psiTable"></table></div>
  </section>

  <section id="bivariate">
    <div class="sec-head"><h2>Bivariate profiles</h2>
      <span class="note">Bin volume, bin mean and bad rate across %%SEG_LIST%%</span></div>
    <div class="tabs" style="align-items:center">
      <input type="search" id="bivSearch" placeholder="filter attribute…" aria-label="Filter bivariate cards">
      <button class="tab on" data-sort="imp">By importance</button>
      <button class="tab" data-sort="abc">A–Z</button>
      <span class="count" id="bivCount"></span>
    </div>
    <div class="biv-grid" id="bivGrid"></div>
  </section>

  <section id="stats">
    <div class="sec-head"><h2>Distribution statistics</h2>
      <span class="note">Counts, missingness and deciles per attribute</span></div>
    <div class="tabs" id="statTabs" style="align-items:center">
      <input type="search" id="statSearch" placeholder="filter attribute…" aria-label="Filter statistics table">
      <span class="count" id="statCount"></span>
    </div>
    <div class="scroll"><table id="statTable"></table></div>
  </section>

  <p class="foot">Rendered from <span class="mono">%%XLSX_PATH%%</span>
  — every sheet (%%SHEETS%%) and all %%N_IMG%% embedded charts are reproduced above.</p>
</main>

<script type="application/json" id="D">%%DATA%%</script>
<script>
const D = JSON.parse(document.getElementById("D").textContent);
const $ = (s,r=document)=>r.querySelector(s);
const el = (t,c,h)=>{const e=document.createElement(t); if(c)e.className=c; if(h!=null)e.innerHTML=h; return e;};
const SEGN = new Proxy({dev:"Development", oot:"Out-of-time", val:"Validation"},{get:(o,k)=>o[k]||k});
const SEGS = D.segs, BMK = D.bmk, NS = SEGS.length;
const WINS = D.pop.map(p=>p.win).filter(Boolean);
const WIN_ALL = !WINS.length ? "" : WINS.reduce((a,b)=>{
  const [as_,ae]=a.split(" → "), [bs,be]=b.split(" → ");
  return (as_<bs?as_:bs)+" → "+(ae>be?ae:be);
});

const nf = (v,d=4)=> v==null||v==="" ? "" : (typeof v==="number" ? v.toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d}) : v);
const ni = v => v==null ? "" : (typeof v==="number" ? Math.round(v).toLocaleString("en-US") : v);
const pc = (v,d=2)=> v==null||v==="" ? "" : (typeof v==="number" ? (v*100).toFixed(d)+"%" : v);
const sig = v => { if(v==null||v==="")return ""; if(typeof v!=="number")return v;
  const a=Math.abs(v); if(a===0)return "0"; if(a>=1000)return v.toLocaleString("en-US",{maximumFractionDigits:1});
  if(a>=1)return v.toFixed(3); if(a>=0.001)return v.toFixed(4); return v.toExponential(2); };
const psiClass = v => v==null?"":(v<0.10?"good":(v<=0.25?"warn":"crit"));
const psiWord  = v => v==null?"":(v<0.10?"stable":(v<=0.25?"shift":"unstable"));

/* ---------------- population ---------------- */
const popK = $("#popKpis");
D.pop.forEach(p=>{
  const k = el("div","kpi"+(p.seg==="oot"?" w":""));
  k.innerHTML = `<div class="k">${SEGN[p.seg]} · ${p.seg}</div>
    <div class="v">${pc(p.rate,2)}</div>
    <div class="m">${ni(p.bad)} bad of ${ni(p.total)}</div>
    <div class="m win">${p.win}</div>`;
  popK.append(k);
});
const totN = D.pop.reduce((a,b)=>a+b.total,0), totB = D.pop.reduce((a,b)=>a+b.bad,0);
popK.append(el("div","kpi g",`<div class="k">All segments</div><div class="v">${pc(totB/totN,2)}</div>
  <div class="m">${ni(totB)} bad of ${ni(totN)}</div>`));

buildTable($("#popTable"),
  ["Segment","Description","Window","Total count",D.target+" = 1","Dependence rate","Share of rows"],
  D.pop.map(p=>[
    {v:p.seg,cls:"l name"},{v:SEGN[p.seg],cls:"l"},{v:p.win,cls:"name"},{v:ni(p.total)},{v:ni(p.bad)},
    {v:pc(p.rate,2)},{v:pc(p.total/totN,1)}]),
  [{v:"total",cls:"l name"},{v:"",cls:"l"},{v:WIN_ALL,cls:"name"},{v:ni(totN)},{v:ni(totB)},{v:pc(totB/totN,2)},{v:"100.0%"}]);

/* ---------------- performance ---------------- */
const perfK = $("#perfKpis");
D.perf.forEach(p=>{
  perfK.append(el("div","kpi"+(p.seg==="oot"?" w":""),
   `<div class="k">${p.seg} · AUC</div><div class="v">${p.auc.toFixed(4)}</div>
    <div class="m">Gini ${(2*p.auc-1).toFixed(4)}</div>`));
});
D.perf.forEach(p=>{
  perfK.append(el("div","kpi"+(p.seg==="oot"?" w":""),
   `<div class="k">${p.seg} · KS</div><div class="v">${p.ks.toFixed(4)}</div>
    <div class="m">${SEGN[p.seg]}</div>`));
});
$("#summaryImg").src = D.summary_img;

/* ---------------- gains ---------------- */
const GH = ["Bin","Count","Min score","Mean score","Max score","Min odds","Responder","% resp",
            "Cum % resp","Cum % non-resp","Response rate","KS","Lift","Cum lift"];
const GAINS = [
  {id:BMK,    label:BMK+" · benchmark",     rows:D.gains[BMK].bench,
   note:`${BMK==="dev"?"Development":SEGN[BMK]} deciles — these cuts define the benchmark bins reused below.`}];
SEGS.filter(s=>s!==BMK && D.gains[s]).forEach(s=>{
  const nm = SEGN[s], bn = BMK==="dev"?"development":SEGN[BMK].toLowerCase();
  GAINS.push(
  {id:s+"-b", label:s+" · "+BMK+" bins",   rows:D.gains[s].bench,
   note: s==="oot" ? `${nm} scored into the ${bn} bin cuts, so bin volumes drift.`
                   : `${nm} scored into the ${bn} bin cuts.`},
  {id:s+"-s", label:s+" · own bins",       rows:D.gains[s].seg,
   note:`${nm} re-cut into its own equal-size deciles.`});
});
const gt=$("#gainTabs"), gp=$("#gainPanels");

/* conditional formatting: signed deviation from a neutral value -> tinted cell */
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function dev(v,neutral,span){            // -1 .. +1
  if(v==null||typeof v!=="number"||!neutral) return 0;
  return clamp((v/neutral-1)/span,-1,1);
}
function idxTint(d){                     // over-indexed = accent, under-indexed = clay
  if(!d) return "";
  const tone = d>0 ? "var(--accent)" : "var(--clay)";
  return `background:color-mix(in srgb, ${tone} ${(Math.abs(d)*28).toFixed(1)}%, transparent)`;
}
function bar(f,tone,pct){                // in-cell bar behind right-aligned figures
  const w=(clamp(f,0,1)*100).toFixed(1);
  return `background:linear-gradient(to left, color-mix(in srgb, ${tone} ${pct}%, transparent) ${w}%, transparent ${w}%)`;
}

GAINS.forEach((g,i)=>{
  const b=el("button","tab"+(i===0?" on":""),g.label); b.dataset.g=g.id; gt.append(b);
  const panel=el("div","gain-panel"+(i===0?"":" hidden")); panel.dataset.g=g.id;
  panel.append(el("p","cap",g.note));
  const wrap=el("div","scroll"), t=el("table");
  const body=[]; let totRow=null;
  const nb=g.rows.length-1, bins=g.rows.slice(0,nb);
  const ksMax=Math.max(...bins.map(r=>r[11]||0));
  const base=g.rows[nb][10];                       // segment-level response rate
  const nMax=Math.max(...bins.map(r=>r[1]||0));
  g.rows.forEach((r,idx)=>{
    const isTot = idx===nb;
    const dRate = isTot?0:dev(r[10],base,0.9);
    const dLift = isTot?0:dev(r[12],1,0.9);
    const dCum  = isTot?0:dev(r[13],1,0.9);
    const cells=[
      {v:isTot?"total":r[0], cls:"l name"},
      {v:ni(r[1]), style:isTot?"":bar(r[1]/nMax,"var(--ink)",11)},
      {v:nf(r[2])},{v:nf(r[3])},{v:nf(r[4])},
      {v:ni(r[5])},{v:ni(r[6])},{v:pc(r[7])},{v:pc(r[8])},{v:pc(r[9])},
      {v:pc(r[10]), style:idxTint(dRate)},
      {v:nf(r[11]), cls:(!isTot&&r[11]===ksMax)?"mark":"", style:isTot?"":bar((r[11]||0)/ksMax,"var(--accent)",22)},
      {v:r[12]==null?"":r[12].toFixed(3), style:idxTint(dLift)},
      {v:r[13]==null?"":r[13].toFixed(3), style:idxTint(dCum)}];
    if(isTot) totRow=cells; else body.push(cells);
  });
  buildTable(t,GH,body,totRow);
  wrap.append(t); panel.append(wrap);
  panel.append(el("div","legend",`
    <span><span class="sw" style="background:color-mix(in srgb, var(--accent) 28%, transparent)"></span>
      bad rate / lift above the segment baseline (${pc(base,2)}, lift 1.000)</span>
    <span><span class="sw" style="background:color-mix(in srgb, var(--clay) 28%, transparent)"></span>
      below baseline</span>
    <span><span class="sw" style="background:linear-gradient(to left, color-mix(in srgb, var(--accent) 22%, transparent) 65%, transparent 65%)"></span>
      KS depth · peak <span class="mark">${nf(ksMax)}</span></span>
    <span><span class="sw" style="background:linear-gradient(to left, color-mix(in srgb, var(--ink) 11%, transparent) 65%, transparent 65%)"></span>
      bin volume</span>`));
  gp.append(panel);
});
gt.addEventListener("click",e=>{
  const b=e.target.closest("button"); if(!b)return;
  gt.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===b));
  gp.querySelectorAll(".gain-panel").forEach(p=>p.classList.toggle("hidden",p.dataset.g!==b.dataset.g));
});

/* ---------------- importance ---------------- */
const impMax = D.imp[0].imp, impList=$("#impList");
D.imp.forEach((r,i)=>{
  const p = D.psi[r.attr]||{};
  const row=el("div","imp-row");
  row.innerHTML = `<div class="imp-rank">${i+1}</div>
    <div class="imp-name"><a href="#biv-${cssId(r.attr)}">${r.attr}</a></div>
    <div class="bar-wrap"><div class="bar" style="width:${(r.imp/impMax*100).toFixed(2)}%"></div></div>
    <div class="imp-val">${(r.imp*100).toFixed(2)}%</div>
    <div style="text-align:right"><span class="chip ${psiClass(p.oot)}">${nf(p.oot,4)}</span></div>`;
  impList.append(row);
});
impList.prepend(el("p","cap","Rank · attribute · importance share · out-of-time PSI"));

/* ---------------- psi ---------------- */
const psiRows = D.imp.map(r=>({attr:r.attr, ...D.psi[r.attr]}));
function renderPsi(q=""){
  const rows = psiRows.filter(r=>r.attr.toLowerCase().includes(q));
  buildTable($("#psiTable"),["Attribute",...SEGS,"Verdict (oot)"],
    rows.map(r=>[{v:r.attr,cls:"l name"},...SEGS.map(s=>({v:nf(r[s])})),
      {v:`<span class="chip ${psiClass(r.oot)}">${psiWord(r.oot)}</span>`,cls:"l",raw:true}]));
  const bad = rows.filter(r=>r.oot>0.10).length;
  $("#psiCount").textContent = `${rows.length} shown · ${bad} above 0.10 on oot`;
}
$("#psiSearch").addEventListener("input",e=>renderPsi(e.target.value.trim().toLowerCase()));
renderPsi();

/* ---------------- bivariate ---------------- */
const impIdx = new Map(D.imp.map((r,i)=>[r.attr,i]));
const bivCards = D.bivar.map(b=>{
  const rank = impIdx.has(b.var) ? impIdx.get(b.var)+1 : 999;
  const p = D.psi[b.var]||{};
  const card = el("div","biv"); card.id = "biv-"+cssId(b.var);
  card.dataset.name = b.var.toLowerCase(); card.dataset.rank = rank;
  const impv = D.imp[rank-1] ? (D.imp[rank-1].imp*100).toFixed(2)+"%" : "";
  card.innerHTML = `<div class="biv-head">
      <span class="rk">#${rank}</span><h3>${b.var}</h3>
      <span class="chip">imp ${impv}</span>
      <span class="chip ${psiClass(p.oot)}">PSI oot ${nf(p.oot,4)}</span>
      <span class="chip">${b.rows.length} bins${b.rows.some(r=>r[0]===-1)?" · incl. null":""}</span>
    </div>`;
  const body = el("div","biv-body");
  const tw = el("div","tblwrap"), t = el("table");
  const rates = b.rows.map(r=>r[1+2*NS]).filter(v=>typeof v==="number");
  const rmax = Math.max(...rates,0.0001);
  buildTable(t,["Bin",...["n","mean","bad"].flatMap(h=>SEGS.map(s=>h+" "+s))],
    b.rows.map(r=>[
      {v:r[0]===-1?"null":r[0],cls:"l name"},
      ...r.slice(1,1+NS).map(x=>({v:ni(x)})),
      ...r.slice(1+NS,1+2*NS).map(x=>({v:sig(x)})),
      {v:`<span style="display:inline-flex;align-items:center;gap:6px;justify-content:flex-end">
          <span style="width:${(r[1+2*NS]/rmax*38).toFixed(1)}px;height:7px;border-radius:2px;background:var(--accent);opacity:.75"></span>
          ${pc(r[1+2*NS],2)}</span>`,raw:true},
      ...r.slice(2+2*NS,1+3*NS).map(x=>({v:pc(x,2)}))]));
  tw.append(t);
  const im = el("div","biv-img"); const img = el("img");
  img.src = D.bivar_img[b.var]; img.alt = "Bivariate chart for "+b.var; img.loading="lazy";
  im.append(img);
  body.append(tw,im); card.append(body);
  return {name:b.var, rank, node:card};
});
const grid=$("#bivGrid");
function renderBiv(){
  const q=$("#bivSearch").value.trim().toLowerCase();
  const sort=grid.dataset.sort||"imp";
  const list=[...bivCards].sort((a,b)=> sort==="abc"? a.name.localeCompare(b.name) : a.rank-b.rank);
  grid.innerHTML="";
  let n=0;
  list.forEach(c=>{ const hit=c.name.toLowerCase().includes(q); if(hit){n++; grid.append(c.node);} });
  $("#bivCount").textContent = `${n} of ${bivCards.length} attributes`;
}
$("#bivSearch").addEventListener("input",renderBiv);
document.querySelectorAll("[data-sort]").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("[data-sort]").forEach(x=>x.classList.toggle("on",x===b));
  grid.dataset.sort=b.dataset.sort; renderBiv();
}));
renderBiv();

/* ---------------- distribution statistics ---------------- */
const STATH = ["Segment","Attribute","Total","n","% missing","Mean","Std","Min","10%","20%","30%","40%",
               "50%","60%","70%","80%","90%","Max"];
let statSeg="all";
const segTabs=$("#statTabs");
["all",...SEGS].forEach((s,i)=>{
  const b=el("button","tab"+(i===0?" on":""),s==="all"?"All segments":s); b.dataset.seg=s;
  segTabs.prepend(b);
});
segTabs.addEventListener("click",e=>{const b=e.target.closest("[data-seg]"); if(!b)return;
  segTabs.querySelectorAll("[data-seg]").forEach(x=>x.classList.toggle("on",x===b));
  statSeg=b.dataset.seg; renderStats();});
function renderStats(){
  const q=$("#statSearch").value.trim().toLowerCase();
  const rows=D.means.filter(r=>(statSeg==="all"||r[0]===statSeg) && String(r[1]).toLowerCase().includes(q));
  buildTable($("#statTable"),STATH, rows.map(r=>{
    const c=[{v:r[0],cls:"l name"},{v:r[1],cls:"l name"},{v:ni(r[2])},{v:ni(r[3])},
             {v:pc(r[4],2), cls:r[4]>0.5?"mark":""}];
    for(let i=5;i<19;i++) c.push({v:sig(r[i])});
    return c;
  }));
  $("#statCount").textContent = `${rows.length} rows`;
}
$("#statSearch").addEventListener("input",renderStats);
renderStats();

/* ---------------- helpers ---------------- */
function cssId(s){ return s.replace(/[^A-Za-z0-9_-]/g,"_"); }
function buildTable(t,head,body,total){
  t.innerHTML="";
  const th=el("thead"), tr=el("tr");
  head.forEach((h,i)=>tr.append(el("th",i<(head[0]==="Segment"?2:1)?"l":"",h)));
  th.append(tr); t.append(th);
  const tb=el("tbody");
  const add=(cells,cls)=>{const r=el("tr",cls);
    cells.forEach(c=>{const td=el("td",c.cls||""); if(c.style)td.style.cssText=c.style;
      if(c.raw)td.innerHTML=c.v; else td.textContent=c.v; r.append(td);});
    tb.append(r);};
  body.forEach(c=>add(c));
  if(total) add(total,"tot");
  t.append(tb);
}

/* ---------------- nav highlight ---------------- */
const links=[...document.querySelectorAll("#nav a")];
const obs=new IntersectionObserver(es=>{
  es.forEach(e=>{ if(e.isIntersecting){
    links.forEach(l=>l.classList.toggle("on", l.getAttribute("href")==="#"+e.target.id)); }});
},{rootMargin:"-20% 0px -70% 0px"});
document.querySelectorAll("main section").forEach(s=>obs.observe(s));

</script>
'''

if __name__ == '__main__':
    _cli()
