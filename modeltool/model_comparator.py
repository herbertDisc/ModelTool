###############################################################
# Description: Compare new model vs benchmark models
#              Outputs the 模型表现 Excel format
# Author: adapted from model_report.py style
###############################################################

import pandas as pd
import numpy as np
import xlsxwriter
from sklearn.metrics import roc_curve, auc as sk_auc


class model_comparator():
    '''
    Compare benchmark model scores/probs against new model scores/probs and
    produce an Excel report with three sheets:
      • 模型表现   — per-target performance table (AUC, KS, coverage, top10 …)
                    for all models × segments, plus a score correlation matrix.
      • Gains_Table — decile KS/gains table for each new model × target × segment.
      • Cross_Eval  — wide matrix: one row per (segment, model), columns cover
                    every target's metrics side-by-side for easy cross-comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset.  Must contain a 'loan_id' column plus every column
        named in `targets` and `new_models`.
    targets : list of str  OR  dict {display_label: col_name}
        Target column(s).  List uses the column name as the display label;
        dict lets you set a custom label (shown as the section subtitle).
    new_models : list of str  OR  dict {display_name: col_name}
        Columns in `df` that hold new-model scores / probs.
        List uses the column name as the display name.
    outputpath : str
        Path for the output Excel file.
    df_benchmark : pd.DataFrame or str, optional
        Benchmark data — a DataFrame or a path to a CSV file.
        Joined to `df` on 'loan_id'.  Expected column pattern:
            <model_prefix>_probs   (probability / raw score)
            <model_prefix>_score   (integer score, optional)
    benchmark_models : list of str, optional
        Model name prefixes to pull from `df_benchmark`
        (e.g. 'cn_acq_cl_sms_app_v6').  If None all models found are used.
    segs : dict {seg_name: boolean pd.Series}, optional
        Extra sub-population cuts applied on top of 'overall'.
        Each Series must be aligned to `df`'s index.
    period : str
        Label written into the 'period' column (default 'oot').
    use_score : bool
        If True rank by *_score; otherwise rank by *_probs (default).
    new_scores : list of str  OR  dict {display_name: col_name}, optional
        Score columns (integer credit scores) for the new model(s) used
        exclusively for quintile cutting in the Cross_Eval sheet.
        If None, falls back to the `new_models` columns.

    Sample usage
    ------------
    from modeltool.model_comparator import model_comparator

    mc = model_comparator(
        df=df_oot,
        targets={'T7max': 't7_max'},
        new_models={'IZI_v1_st0': 'new_probs'},
        outputpath='comparison_st0.xlsx',
        df_benchmark='data/benchmark_models.csv',
        benchmark_models=[
            'cn_potf_le1_cl_sms_app_v1',
            'cn_potf_le2_risk_sms_app_v1',
            'cn_potf_risk_cb_dyna_v1',
        ],
        segs={'st=0': df_oot['settled_times'] == 0},
        period='oot',
    )
    mc.run()
    '''

    def __init__(self,
                 df,
                 targets,
                 new_models,
                 outputpath,
                 df_benchmark=None,
                 benchmark_models=None,
                 segs=None,
                 period='oot',
                 use_score=False,
                 new_scores=None):

        self.df = df.copy().reset_index(drop=True)
        self.outputpath = outputpath
        self.period = period
        self.use_score = use_score

        # Normalise targets → {label: col}
        if isinstance(targets, dict):
            self.targets = targets
        else:
            self.targets = {t: t for t in targets}

        # Normalise new_models → {display_name: col_in_df}
        if isinstance(new_models, dict):
            self.new_models = new_models
        else:
            self.new_models = {m: m for m in new_models}

        self._new_model_names = set(self.new_models.keys())

        # Score cols for cross_eval quintile cutting (new models)
        # new_scores accepts same format as new_models; falls back to new_models cols
        if new_scores is None:
            self._cross_new_scores = dict(self.new_models)
        elif isinstance(new_scores, dict):
            self._cross_new_scores = new_scores
        else:
            self._cross_new_scores = {s: s for s in new_scores}

        # ------------------------------------------------------------------
        # Load and merge benchmark data
        # ------------------------------------------------------------------
        self.benchmark_models = {}   # {display_name: ranking_col_in_merged_df}
        self._benchmark_probs = {}   # {display_name: probs_col}  (for correlation)
        self._cross_bmk_scores = {}  # {display_name: score_col}  (for cross_eval cutting)

        if df_benchmark is not None:
            if isinstance(df_benchmark, str):
                df_benchmark = pd.read_csv(df_benchmark)
            df_benchmark = df_benchmark.copy()

            probs_cols = [c for c in df_benchmark.columns if c.endswith('_probs')]
            all_prefixes = [c[:-6] for c in probs_cols]

            if benchmark_models is None:
                selected_prefixes = all_prefixes
            else:
                selected_prefixes = [p for p in benchmark_models
                                     if p in all_prefixes]

            keep = {'loan_id'}
            for prefix in selected_prefixes:
                rank_col  = prefix + ('_score' if use_score else '_probs')
                prob_col  = prefix + '_probs'
                score_col = prefix + '_score'
                if rank_col in df_benchmark.columns:
                    keep.add(rank_col)
                    self.benchmark_models[prefix] = rank_col
                if prob_col in df_benchmark.columns:
                    keep.add(prob_col)
                    self._benchmark_probs[prefix] = prob_col
                # always load score col for cross_eval if available
                if score_col in df_benchmark.columns:
                    keep.add(score_col)
                    self._cross_bmk_scores[prefix] = score_col
                else:
                    self._cross_bmk_scores[prefix] = rank_col   # fallback

            df_bm = (df_benchmark[list(keep)]
                     .drop_duplicates('loan_id')
                     .reset_index(drop=True))
            self.df = self.df.merge(df_bm, on='loan_id', how='left')

            # Invalidate benchmark prob values outside [0, 1) — treat as uncovered
            for prob_col in self._benchmark_probs.values():
                if prob_col in self.df.columns:
                    invalid = (self.df[prob_col] < 0) | (self.df[prob_col] >= 1)
                    self.df.loc[invalid, prob_col] = np.nan

        # Probs cols for new models (for correlation matrix)
        self._new_probs = dict(self.new_models)  # same cols used for ranking

        # ------------------------------------------------------------------
        # Segment dict – 'overall' is always first
        # ------------------------------------------------------------------
        n = len(self.df)
        overall_mask = pd.Series([True] * n, index=self.df.index)
        self.segs = {'overall': overall_mask}
        if segs:
            for k, v in segs.items():
                self.segs[k] = v.reset_index(drop=True) if hasattr(v, 'reset_index') else v

        # ------------------------------------------------------------------
        # Excel workbook
        # ------------------------------------------------------------------
        self.workbook = xlsxwriter.Workbook(
            self.outputpath, options={'nan_inf_to_errors': True})
        self._load_format()

    # -----------------------------------------------------------------------
    # Format definitions
    # -----------------------------------------------------------------------
    def _load_format(self):
        wb = self.workbook

        self.fmt_seg_title = wb.add_format({
            'bold': True, 'font_name': 'Microsoft YaHei Light',
            'font_size': 15, 'valign': 'top', 'fg_color': '#FABF8F',
        })
        self.fmt_sec_title = wb.add_format({
            'bold': True, 'font_name': 'Microsoft YaHei Light',
            'font_size': 13, 'valign': 'top',
            'fg_color': '#FABF8F',
        })
        self.fmt_subtitle = wb.add_format({
            'bold': True, 'font_name': 'Microsoft YaHei Light',
            'font_size': 11, 'valign': 'top',
        })
        self.fmt_header = wb.add_format({
            'bold': True, 'font_name': 'Microsoft YaHei Light',
            'text_wrap': True, 'valign': 'top',
            'fg_color': '#C5D9F1', 'border': 1,
        })
        self.fmt_idx = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top', 'border': 1,
        })
        self.fmt_body = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top', 'border': 1,
        })
        self.fmt_num3 = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top',
            'border': 1, 'num_format': '0.000',
        })
        self.fmt_pct = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top',
            'border': 1, 'num_format': '0.00%',
        })
        # New-model rows: light-green highlight
        self.fmt_body_new = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top',
            'border': 1, 'fg_color': '#EBF1DE',
        })
        self.fmt_num3_new = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top',
            'border': 1, 'num_format': '0.000', 'fg_color': '#EBF1DE',
        })
        self.fmt_pct_new = wb.add_format({
            'font_name': 'Microsoft YaHei Light', 'valign': 'top',
            'border': 1, 'num_format': '0.00%', 'fg_color': '#EBF1DE',
        })

    # -----------------------------------------------------------------------
    # Metric computation
    # -----------------------------------------------------------------------
    def _metrics(self, y_true, y_pred, total_in_seg):
        '''
        Returns (cnt, coverage, bad_rate, auc, ks, top10_badrate, top10_lift).
        Uses only rows where both y_true and y_pred are non-null.
        '''
        mask = ~(pd.isna(y_pred) | pd.isna(y_true))
        yt = np.array(y_true[mask], dtype=float)
        yp = np.array(y_pred[mask], dtype=float)

        cnt = int(len(yt))
        coverage = cnt / total_in_seg if total_in_seg > 0 else np.nan
        bad_rate = float(yt.mean()) if cnt > 0 else np.nan

        if cnt < 10 or yt.sum() == 0 or yt.sum() == cnt:
            return cnt, coverage, bad_rate, np.nan, np.nan, np.nan, np.nan

        fpr, tpr, _ = roc_curve(yt, yp)
        mdl_auc = round(float(sk_auc(fpr, tpr)), 3)
        mdl_ks  = round(float(np.max(tpr - fpr)), 3)

        top_n = max(1, int(cnt * 0.10))
        order = np.argsort(yp)[::-1]
        top10_br   = round(float(yt[order[:top_n]].mean()), 4)
        top10_lift = round(top10_br / bad_rate, 3) if bad_rate > 0 else np.nan

        return cnt, coverage, bad_rate, mdl_auc, mdl_ks, top10_br, top10_lift

    # -----------------------------------------------------------------------
    # Row builder
    # -----------------------------------------------------------------------
    def _build_perf_rows(self, target_col):
        rows = []
        all_models = {**self.new_models, **self.benchmark_models}

        for seg_name, mask in self.segs.items():
            seg_df = self.df[mask].copy()
            total = len(seg_df)
            if total == 0:
                continue

            for model_name, score_col in all_models.items():
                if score_col not in seg_df.columns:
                    continue
                cnt, cov, br, mdl_auc, ks, top10_br, top10_lift = self._metrics(
                    seg_df[target_col].reset_index(drop=True),
                    seg_df[score_col].reset_index(drop=True),
                    total,
                )
                rows.append({
                    'period':        self.period,
                    'segment':       seg_name,
                    'model':         model_name,
                    'cnt':           cnt,
                    'coverage':      cov,
                    'bad_rate':      br,
                    'auc':           mdl_auc,
                    'ks':            ks,
                    'top10_badrate': top10_br,
                    'top10_lift':    top10_lift,
                    '_is_new':       model_name in self._new_model_names,
                })
        return rows

    # -----------------------------------------------------------------------
    # Writers
    # -----------------------------------------------------------------------
    HEADERS = ['period', 'segment', 'model', 'cnt', 'coverage',
               'bad_rate', 'auc', 'ks', 'top10_badrate', 'top10_lift']

    def _write_perf_table(self, ws, start_row, rows):
        for c, h in enumerate(self.HEADERS):
            ws.write(start_row, c, h, self.fmt_header)
        row = start_row + 1

        prev_seg = None
        for r in rows:
            is_new = r['_is_new']
            fb   = self.fmt_body_new if is_new else self.fmt_body
            fn3  = self.fmt_num3_new if is_new else self.fmt_num3
            fpct = self.fmt_pct_new  if is_new else self.fmt_pct

            if prev_seg is not None and r['segment'] != prev_seg:
                row += 1         # blank separator between segments
            prev_seg = r['segment']

            ws.write(row, 0, r['period'],        fb)
            ws.write(row, 1, r['segment'],        fb)
            ws.write(row, 2, r['model'],          fb)
            ws.write(row, 3, r['cnt'],            fb)
            ws.write(row, 4, r['coverage'],       fpct)
            ws.write(row, 5, r['bad_rate'],       fpct)
            ws.write(row, 6, r['auc'],            fn3)
            ws.write(row, 7, r['ks'],             fn3)
            ws.write(row, 8, r['top10_badrate'],  fpct)
            ws.write(row, 9, r['top10_lift'],     fn3)
            row += 1

        return row + 1   # leave blank row after section

    def _write_corr_matrix(self, ws, start_row):
        ws.write(start_row, 0, '新模型与主流模型的相关系数', self.fmt_sec_title)
        row = start_row + 2

        # Collect probs columns for all models
        probs_map = {}
        for name, col in self._new_probs.items():
            probs_map[name] = col
        for name, col in self._benchmark_probs.items():
            if name not in probs_map:
                probs_map[name] = col

        model_names = list(probs_map.keys())
        if not model_names:
            return row

        # Header row
        ws.write(row, 0, 'Score', self.fmt_header)
        for c, name in enumerate(model_names):
            ws.write(row, c + 1, name, self.fmt_header)
        row += 1

        # Correlation matrix
        valid_names = [n for n in model_names if probs_map[n] in self.df.columns]
        df_sc = self.df[[probs_map[n] for n in valid_names]].dropna()
        corr = np.corrcoef(df_sc.values.T) if len(df_sc) > 1 else np.eye(len(valid_names))

        for i, name in enumerate(valid_names):
            ws.write(row + i, 0, name, self.fmt_idx)
            for j in range(len(valid_names)):
                ws.write(row + i, j + 1, round(float(corr[i, j]), 4), self.fmt_num3)
        row += len(valid_names) + 2

        # Coverage row
        ws.write(row, 0, 'Coverage (OOT non-null):', self.fmt_idx)
        total = len(self.df)
        for c, name in enumerate(valid_names):
            col = probs_map[name]
            cnt = int(self.df[col].notna().sum())
            pct = cnt / total * 100 if total > 0 else 0
            ws.write(row, c + 1, f'{cnt:,} ({pct:.1f}%)', self.fmt_body)
        row += 2
        return row

    # -----------------------------------------------------------------------
    # Gains table helpers
    # -----------------------------------------------------------------------
    def _ks_table(self, y_true, y_pred, nbins=10):
        '''
        Compute a decile-level gains/KS table.
        Returns a DataFrame with columns:
            Total_Int, Min_Score, Mean_Score, Max_Score,
            Responder, PercResponder, CumPercResponder,
            CumPercNonResponder, ResponseRate, KS, lift, cum_lift
        plus a totals row appended at the bottom.
        '''
        mask = ~(pd.isna(y_pred) | pd.isna(y_true))
        yt = y_true[mask].reset_index(drop=True)
        yp = y_pred[mask].reset_index(drop=True)

        # quantile binning
        try:
            grp, bins = pd.qcut(yp, nbins, labels=False, retbins=True, duplicates='drop')
        except ValueError:
            grp, bins = pd.qcut(yp, max(2, yp.nunique()), labels=False,
                                retbins=True, duplicates='drop')
        grp = grp.fillna(-1)
        if len(bins) > 2:
            bins[0], bins[-1] = -np.inf, np.inf
        else:
            bins = np.insert(bins, [0, len(bins)], [-np.inf, np.inf])

        df_tmp = pd.DataFrame({'grp': grp, 'dep': yt, 'score': yp})
        total       = len(df_tmp)
        total_bad   = df_tmp['dep'].sum()
        total_good  = total - total_bad

        g = df_tmp.groupby('grp', sort=True, observed=False)
        total_int   = g['grp'].count().sort_index(ascending=False)
        responder   = g['dep'].sum().sort_index(ascending=False)
        non_resp    = total_int - responder
        min_s       = g['score'].min().sort_index(ascending=False)
        mean_s      = g['score'].mean().sort_index(ascending=False)
        max_s       = g['score'].max().sort_index(ascending=False)

        perc_resp       = responder / total_bad
        cum_perc_resp   = np.cumsum(responder) / total_bad
        cum_perc_non    = np.cumsum(non_resp) / total_good
        resp_rate       = responder / total_int
        ks              = cum_perc_resp - cum_perc_non
        lift            = resp_rate / (total_bad / total)
        cum_lift        = (np.cumsum(responder) / np.cumsum(total_int)) / (total_bad / total)

        cols = ['Total_Int', 'Min_Score', 'Mean_Score', 'Max_Score',
                'Responder', 'PercResponder', 'CumPercResponder',
                'CumPercNonResponder', 'ResponseRate', 'KS', 'lift', 'cum_lift']
        tbl = pd.concat([total_int, min_s, mean_s, max_s, responder,
                         perc_resp, cum_perc_resp, cum_perc_non,
                         resp_rate, ks, lift, cum_lift],
                        keys=cols, axis=1)

        totals = pd.Series([total, yp.min(), yp.mean(), yp.max(),
                            total_bad, 1, '', '', total_bad / total,
                            abs(ks).max(), '', ''], index=cols)
        tbl = pd.concat([tbl, totals.to_frame().T], ignore_index=True)
        return tbl

    def _write_one_gains_block(self, ws, row, tbl, title, ratio_cols, is_new=True):
        '''Write a single gains-table block (title + header + data rows + conditional fmt).'''
        ws.write(row, 1, title, self.fmt_seg_title)
        ws.set_row(row, 18, self.fmt_seg_title)
        row += 2

        headers = list(tbl.columns)
        for c, h in enumerate(headers):
            ws.write(row, c + 1, h, self.fmt_header)
        row += 1

        for r_idx, data_row in tbl.iterrows():
            is_total = r_idx == len(tbl) - 1
            highlight = is_new and is_total
            for c, h in enumerate(headers):
                val = data_row[h]
                if h in ratio_cols and val != '':
                    fmt = self.fmt_pct_new if highlight else self.fmt_pct
                else:
                    fmt = self.fmt_body_new if highlight else self.fmt_body
                ws.write(row, c + 1, val, fmt)
            row += 1

        resp_col_idx = headers.index('ResponseRate') + 1
        ws.conditional_format(
            row - len(tbl), resp_col_idx,
            row - 1,        resp_col_idx,
            {'type': 'data_bar'},
        )
        return row + 3

    def write_gains_table(self):
        '''
        Write Gains_Table worksheet: decile KS table for every model
        (new + benchmark) × target × segment.
        New-model totals rows are highlighted green; benchmark rows use default formatting.
        '''
        ws = self.workbook.add_worksheet('Gains_Table')
        ws.set_column('A:A', 2)
        ws.set_column('B:B', 14)

        ratio_cols = ['PercResponder', 'CumPercResponder', 'CumPercNonResponder', 'ResponseRate']
        col_widths = {'Total_Int': 10, 'Min_Score': 11, 'Mean_Score': 11, 'Max_Score': 11,
                      'Responder': 10, 'KS': 9, 'lift': 9, 'cum_lift': 10}
        for c, col in enumerate(['grp'] + list(col_widths.keys()) + ratio_cols):
            ws.set_column(c + 1, c + 1, col_widths.get(col, 14))

        all_models = {**{n: (col, True)  for n, col in self.new_models.items()},
                      **{n: (col, False) for n, col in self.benchmark_models.items()}}

        row = 1
        for tgt_label, tgt_col in self.targets.items():
            if tgt_col not in self.df.columns:
                continue
            for mdl_name, (mdl_col, is_new) in all_models.items():
                if mdl_col not in self.df.columns:
                    continue
                for seg_name, mask in self.segs.items():
                    seg_df = self.df[mask].copy()
                    valid = seg_df[[tgt_col, mdl_col]].dropna()
                    if len(valid) < 10:
                        continue
                    tbl = self._ks_table(
                        valid[tgt_col].reset_index(drop=True),
                        valid[mdl_col].reset_index(drop=True),
                    )
                    title = f'{tgt_label}  |  {mdl_name}  |  {seg_name}'
                    row = self._write_one_gains_block(
                        ws, row, tbl, title, ratio_cols, is_new=is_new)

    # -----------------------------------------------------------------------
    # Cross-target evaluation (quintile cross-tab matrix)
    # -----------------------------------------------------------------------
    def _cross_tab(self, y_true, score_new, score_bmk, nbins=5):
        '''
        Build a quintile cross-tabulation of bad rate and count.
        Returns (bad_mat, cnt_mat, col_avg, col_cnt) where:
          bad_mat  — DataFrame rows=new quintile, cols=bmk quintile + "Row avg"
          cnt_mat  — same shape, integer counts + "Row total"
          col_avg  — Series of column-wise bad rates + overall ("Row avg")
          col_cnt  — Series of column-wise counts + total ("Row total")
        '''
        labels = [f'Q{i+1}' for i in range(nbins)]

        def safe_qcut(s, n, lbls):
            try:
                return pd.qcut(s, n, labels=lbls, duplicates='drop')
            except ValueError:
                return pd.qcut(s, max(2, s.nunique()), labels=lbls[:max(2, s.nunique())],
                               duplicates='drop')

        new_q = safe_qcut(score_new, nbins, labels)
        bmk_q = safe_qcut(score_bmk, nbins, labels)

        df_tmp = pd.DataFrame({'nq': new_q, 'bq': bmk_q, 'dep': y_true}).dropna()

        bad_mat = (df_tmp.groupby(['nq', 'bq'], observed=True)['dep']
                   .mean().unstack())
        cnt_mat = (df_tmp.groupby(['nq', 'bq'], observed=True)['dep']
                   .count().unstack().fillna(0).astype(int))

        bad_mat['Row avg']   = df_tmp.groupby('nq', observed=True)['dep'].mean()
        cnt_mat['Row total'] = df_tmp.groupby('nq', observed=True)['dep'].count().astype(int)

        # Reverse row order: Q5 (highest prob = highest bad rate) at top,
        # Q1 (lowest prob = lowest bad rate) at bottom → lowest Y target sits at bottom-right
        bad_mat = bad_mat.sort_index(ascending=False)
        cnt_mat = cnt_mat.sort_index(ascending=False)

        col_avg = df_tmp.groupby('bq', observed=True)['dep'].mean()
        col_avg['Row avg'] = df_tmp['dep'].mean()

        col_cnt = df_tmp.groupby('bq', observed=True)['dep'].count()
        col_cnt['Row total'] = len(df_tmp)

        return bad_mat, cnt_mat, col_avg, col_cnt

    def _write_cross_seg_block(self, ws, start_row, start_col, seg_name,
                               bad_mat, cnt_mat, col_avg, col_cnt):
        '''
        Write one segment's bad-rate + count tables at (start_row, start_col).
        Returns number of rows consumed.
        '''
        def q_label(q, first, last):
            if str(q) == str(first):
                return f'{q}(low)'
            if str(q) == str(last):
                return f'{q}(high)'
            return str(q)

        bad_cols = list(bad_mat.columns)   # Q1..Q5 + Row avg
        cnt_cols = list(cnt_mat.columns)   # Q1..Q5 + Row total
        n_q      = len(bad_cols)
        c0       = start_col
        r        = start_row

        # Segment name sub-header
        n_total = int(col_cnt.get('Row total', 0))
        ws.write(r, c0, f'[{seg_name}]  n={n_total:,}', self.fmt_subtitle)
        r += 1

        # ---- Bad Rate table ----
        ws.write(r, c0, 'Bad Rate', self.fmt_header)
        first_c, last_c = bad_cols[0], bad_cols[-2]
        for dc, ch in enumerate(bad_cols):
            ws.write(r, c0 + 1 + dc, q_label(ch, first_c, last_c), self.fmt_header)
        r += 1

        bad_data_start = r
        first_r, last_r = bad_mat.index[0], bad_mat.index[-1]
        for q_row, data in bad_mat.iterrows():
            ws.write(r, c0, q_label(q_row, first_r, last_r), self.fmt_idx)
            for dc, ch in enumerate(bad_cols):
                val = data[ch]
                ws.write(r, c0 + 1 + dc, val if not pd.isna(val) else '', self.fmt_pct)
            r += 1

        ws.write(r, c0, 'Col avg', self.fmt_idx)
        for dc, ch in enumerate(bad_cols):
            val = col_avg.get(ch, np.nan)
            ws.write(r, c0 + 1 + dc, val if not pd.isna(val) else '', self.fmt_pct)
        col_avg_row = r
        r += 2   # gap between bad-rate and count tables

        # 3-color scale: inner Q×Q matrix only (exclude Row avg col and Col avg row)
        ws.conditional_format(
            bad_data_start, c0 + 1,
            col_avg_row - 1, c0 + n_q - 1,
            {'type': '3_color_scale',
             'min_color': '#63BE7B',
             'mid_color': '#FFEB84',
             'max_color': '#F8696B'},
        )

        # ---- Count table ----
        ws.write(r, c0, 'Count', self.fmt_header)
        first_c2, last_c2 = cnt_cols[0], cnt_cols[-2]
        for dc, ch in enumerate(cnt_cols):
            ws.write(r, c0 + 1 + dc, q_label(ch, first_c2, last_c2), self.fmt_header)
        r += 1

        first_r2, last_r2 = cnt_mat.index[0], cnt_mat.index[-1]
        for q_row, data in cnt_mat.iterrows():
            ws.write(r, c0, q_label(q_row, first_r2, last_r2), self.fmt_idx)
            for dc, ch in enumerate(cnt_cols):
                ws.write(r, c0 + 1 + dc, int(data[ch]), self.fmt_body)
            r += 1

        ws.write(r, c0, 'Col total', self.fmt_idx)
        for dc, ch in enumerate(cnt_cols):
            ws.write(r, c0 + 1 + dc, int(col_cnt.get(ch, 0)), self.fmt_body)
        r += 1

        return r - start_row   # rows consumed

    def write_cross_eval(self):
        '''
        Write Cross_Eval worksheet.
        Loop: target → new_model → benchmark_model.
        For each group, all segments are placed SIDE BY SIDE horizontally so
        overall / seg1 / seg2 / … can be compared at a glance.
        Each seg block contains: bad-rate 5×5 + col avg, then count 5×5 + col total.
        3-color scale applied to inner Q×Q cells only (no Row/Col avg coloring).
        '''
        if not self.benchmark_models:
            return

        ws = self.workbook.add_worksheet('Cross_Eval')
        ws.set_column('A:A', 2)

        C0  = 1   # left margin column
        row = 1

        for tgt_label, tgt_col in self.targets.items():
            if tgt_col not in self.df.columns:
                continue
            for new_name, new_scr_col in self._cross_new_scores.items():
                if new_scr_col not in self.df.columns:
                    continue
                for bmk_name, bmk_scr_col in self._cross_bmk_scores.items():
                    if bmk_scr_col not in self.df.columns:
                        continue

                    # Group header (one per target × model pair)
                    ws.write(row, C0, f'{tgt_label}', self.fmt_seg_title)
                    ws.set_row(row, 18, self.fmt_seg_title)
                    row += 2
                    ws.write(row, C0,
                             f'OOT Cross Evaluation: {new_name} × {bmk_name}',
                             self.fmt_sec_title)
                    row += 1
                    ws.write(row, C0,
                             f'Rows = {new_name} quintile / '
                             f'Cols = {bmk_name} quintile / '
                             f'Q1=lowest score  Q5=highest score',
                             self.fmt_subtitle)
                    row += 2

                    tables_row   = row
                    col_cursor   = C0
                    max_rows_used = 0

                    for seg_name, mask in self.segs.items():
                        valid = self.df.loc[mask, [tgt_col, new_scr_col, bmk_scr_col]].dropna()
                        if len(valid) < 10:
                            continue

                        bad_mat, cnt_mat, col_avg, col_cnt = self._cross_tab(
                            valid[tgt_col].reset_index(drop=True),
                            valid[new_scr_col].reset_index(drop=True),
                            valid[bmk_scr_col].reset_index(drop=True),
                        )

                        n_q = len(bad_mat.columns)   # Q1..Q5 + Row avg
                        # set column widths for this seg's block
                        ws.set_column(col_cursor, col_cursor, 11)          # row-label col
                        for dc in range(n_q):
                            ws.set_column(col_cursor + 1 + dc, col_cursor + 1 + dc, 10)

                        rows_used = self._write_cross_seg_block(
                            ws, tables_row, col_cursor,
                            seg_name, bad_mat, cnt_mat, col_avg, col_cnt,
                        )
                        max_rows_used = max(max_rows_used, rows_used)
                        col_cursor += n_q + 2   # advance: row-label(1) + data(n_q) + gap(1)

                    row = tables_row + max_rows_used + 4

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------
    def run(self):
        '''Generate the full Excel report with three sheets: 模型表现, Gains_Table, Cross_Eval.'''
        ws = self.workbook.add_worksheet('模型表现')
        ws.set_column('A:A', 10)
        ws.set_column('B:B', 18)
        ws.set_column('C:C', 32)
        ws.set_column('D:D', 8)
        ws.set_column('E:J', 13)

        ws.write(0, 0, '模型表现：oot效果&比较模型对比', self.fmt_sec_title)
        row = 2

        for label, target_col in self.targets.items():
            if target_col not in self.df.columns:
                print(f'[WARN] target column "{target_col}" not found — skipped.')
                continue
            ws.write(row, 0, label, self.fmt_subtitle)
            row += 1
            perf_rows = self._build_perf_rows(target_col)
            row = self._write_perf_table(ws, row, perf_rows)

        self._write_corr_matrix(ws, row)

        self.write_gains_table()
        self.write_cross_eval()

        self.workbook.close()
        print(f'Saved → {self.outputpath}')
