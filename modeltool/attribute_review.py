#########################################################
# Description: Review attributes via bivariate analysis
# Auther: Xinyue He
# Created date: 2024.08
#########################################################

import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import numpy as np
import os


class attribute_reviewer():
    '''
    Class to review attributes via bivariate analysis.

    Input:
        df: pd.DataFrame
            Dataset containing attributes and dependent variable.
        segs: list of string or None
            List of column names that define the segment. If None (together
            with bmk_seg=None), the whole DataFrame is treated as a single
            segment named 'all'.
        bmk_seg: list of strings/numbers or None
            Segment selected as the benchmark. Must be None when segs is None.
        dep: string
            Column name of the dependent variable.
        varlist: list of string
            List of attribute column names to review.
        outputpath: string
            File path for saving the final output Excel.
        cat_vars: list of string or None
            List of categorical attribute column names. If None, non-numeric
            columns in varlist are automatically detected and treated as
            categorical.

    Sample code:
        # With segmentation
        engine = attribute_reviewer(
            df=df,
            segs=['seg'],
            dep='bad_flag',
            bmk_seg=['train'],
            varlist=['attr1', 'attr2', 'attr3'],
            outputpath='attribute_review.xlsx',
            cat_vars=['attr3'],   # or None for auto-detection
        )
        engine.run()

        # Without segmentation (whole df as one segment)
        engine = attribute_reviewer(
            df=df,
            segs=None,
            dep='bad_flag',
            bmk_seg=None,
            varlist=['attr1', 'attr2', 'attr3'],
            outputpath='attribute_review.xlsx',
        )
        engine.run()
    '''

    _DUMMY_SEG_COL = '_seg_'
    _DUMMY_SEG_VAL = 'all'

    def __init__(self, df, dep, varlist, outputpath, segs=None, bmk_seg=None, cat_vars=None) -> None:
        self.df = df
        self.dep = dep
        self._no_seg = segs is None and bmk_seg is None
        if self._no_seg:
            self.segs = [self._DUMMY_SEG_COL]
            self.bmk_seg = (self._DUMMY_SEG_VAL,)
        else:
            self.segs = segs
            self.bmk_seg = tuple([str(x) for x in bmk_seg])
        self.varlist = varlist
        self.varlist_bivar = varlist
        self.outputpath = outputpath

        auto_cat = []
        for v in varlist:
            try:
                df[v].astype(float)
            except (ValueError, TypeError):
                auto_cat.append(v)
        if cat_vars is not None:
            auto_cat = list(dict.fromkeys(auto_cat + [v for v in cat_vars if v in varlist]))
        self.cat_vars = auto_cat
        self.num_vars = [v for v in varlist if v not in self.cat_vars]

        self.seg_title_format = None
        self.title_format = None
        self.header_format = None
        self.idx_format = None
        self.body_format = None
        self.ratio_format = None
        self.auc_format = None
        self._create_excel()
        os.makedirs('tmp', exist_ok=True)

    def _create_excel(self):
        '''
        Initialization of creating excel.
        '''
        self.workbook = xlsxwriter.Workbook(self.outputpath, options={'nan_inf_to_errors': True})
        self._load_format()

    def _load_format(self):
        '''
        Store all formats of the excel writer.
        '''
        self.seg_title_format = self.workbook.add_format({
            "bold": True,
            "font": "Calibri",
            "font_size": 13,
            "font_color": "#FFFFFF",
            "text_wrap": False,
            "valign": "vcenter",
            "fg_color": "#1F3864",
        })
        self.title_format = self.workbook.add_format({
            "bold": True,
            "font": "Calibri",
            "font_size": 11,
            "font_color": "#1F3864",
            "text_wrap": False,
            "valign": "top",
        })
        self.header_format = self.workbook.add_format({
            "bold": True,
            "font": "Calibri",
            "font_color": "#FFFFFF",
            "text_wrap": True,
            "valign": "vcenter",
            "fg_color": "#2E75B6",
            "border": 1,
            "border_color": "#1F3864",
        })
        self.idx_format = self.workbook.add_format({
            "bold": False,
            "font": "Calibri",
            "valign": "top",
            'text_wrap': False,
            "border": 1,
            "border_color": "#B8CCE4",
            "fg_color": "#EBF3FB",
        })
        self.body_format = self.workbook.add_format({
            "bold": False,
            "font": "Calibri",
            "valign": "top",
            'text_wrap': False,
            "border": 1,
            "border_color": "#B8CCE4",
        })
        self.ratio_format = self.workbook.add_format({
            "bold": False,
            "text_wrap": True,
            "font": "Calibri",
            "valign": "top",
            "border": 1,
            "border_color": "#B8CCE4",
            'num_format': '0.00%',
        })
        self.auc_format = self.workbook.add_format({
            "bold": False,
            "text_wrap": True,
            "font": "Calibri",
            "valign": "top",
            "border": 1,
            "border_color": "#B8CCE4",
            'num_format': '0.0000',
        })

    def write_seg_line(self, worksheet, row, col, title):
        '''
        Write a single segment title line.
        '''
        worksheet.write(row, col, title)
        worksheet.set_row(row, 15, self.seg_title_format)
        return 2

    def writedf(self, df, worksheet, row, col, indexing=False, ratio_col=None, auc_col=None, cdt_fmt=None, title=None):
        '''
        Write a whole dataframe into worksheet.
        '''
        if ratio_col is None:
            ratio_col = []
        if auc_col is None:
            auc_col = []
        if cdt_fmt is None:
            cdt_fmt = []

        def writerow(ws, row, col, arr, fmt):
            for idx, val in enumerate(arr):
                ws.write(row, col + idx, val, fmt)

        def writecol(ws, row, col, arr, fmt):
            for idx, val in enumerate(arr):
                ws.write(row + idx, col, val, fmt)

        if title is not None:
            worksheet.write(row, col, title, self.title_format)
            row += 1

        idx_len = 0
        col_len = df.columns.nlevels

        if indexing:
            idx_len = df.index.nlevels
            writerow(worksheet, row + col_len - 1, col, df.index.names, self.header_format)
            for idx, val in enumerate(df.index.values):
                if idx_len == 1:
                    val = [val]
                writerow(worksheet, row + idx + col_len, col, val, self.idx_format)

        for col_num, value in enumerate(df.columns.values):
            if col_len > 1:
                writecol(worksheet, row, col + idx_len + col_num, value, self.header_format)
            else:
                worksheet.write(row, col + idx_len + col_num, value, self.header_format)
            if value in ratio_col:
                writecol(worksheet, row + col_len, col + idx_len + col_num, df[value], self.ratio_format)
            elif value in auc_col:
                writecol(worksheet, row + col_len, col + idx_len + col_num, df[value], self.auc_format)
            else:
                writecol(worksheet, row + col_len, col + idx_len + col_num, df[value], self.body_format)

            if value in cdt_fmt:
                worksheet.conditional_format(
                    row + 1, col + idx_len + col_num,
                    row + len(df[value]), col + idx_len + col_num,
                    {'type': 'data_bar'}
                )

        df_shape = [df.shape[0], df.shape[1]]
        df_shape[1] += idx_len
        return df_shape

    def pct_rank_qcut(self, series, nbins=None, bins=None):
        '''
        Function for binning the data.
        '''
        if bins is None:
            grp, bins = pd.qcut(series, nbins, labels=False, retbins=True, duplicates='drop')
            grp = grp.fillna(-1)
            if len(bins) > 2:
                bins[0] = -np.inf
                bins[-1] = np.inf
            else:
                bins = np.insert(bins, [0, len(bins)], [-np.inf, np.inf])
            return grp, list(bins)
        else:
            grp = pd.cut(series, bins=bins, labels=False)
            grp = grp.fillna(-1)
            return grp, list(bins)

    def calc_iv(self, df, nbins=10):
        '''
        Calculate IV for each variable in varlist on the given dataframe.
        Categorical variables are grouped by their actual values; numeric
        variables are binned via quantile cut.

        Input:
            df: dataframe (typically the benchmark segment)
            nbins: number of bins
        Output:
            dict {variable: iv_value}
        '''
        total_events = df[self.dep].sum()
        total_non_events = len(df) - total_events
        iv_dict = {}

        for col in self.varlist:
            try:
                if col in self.cat_vars:
                    df_tmp = pd.DataFrame({
                        'grp': df[col].astype(str).fillna('Null').reset_index(drop=True),
                        'dep': df[self.dep].reset_index(drop=True),
                    })
                else:
                    grp, _ = self.pct_rank_qcut(df[col].reset_index(drop=True), nbins=nbins)
                    df_tmp = pd.DataFrame({'grp': grp, 'dep': df[self.dep].reset_index(drop=True)})
                grouped = df_tmp.groupby('grp')['dep'].agg(['sum', 'count'])
                grouped.columns = ['events', 'total']
                grouped['non_events'] = grouped['total'] - grouped['events']
                grouped['pct_e'] = (grouped['events'] / total_events).clip(lower=1e-6)
                grouped['pct_ne'] = (grouped['non_events'] / total_non_events).clip(lower=1e-6)
                grouped['woe'] = np.log(grouped['pct_e'] / grouped['pct_ne'])
                grouped['iv'] = (grouped['pct_e'] - grouped['pct_ne']) * grouped['woe']
                iv_dict[col] = round(grouped['iv'].sum(), 6)
            except Exception:
                iv_dict[col] = 0.0

        return iv_dict

    def calc_auc(self, df):
        '''
        Calculate ROC AUC for each variable in varlist on the given dataframe
        using the Mann-Whitney U statistic (no external dependencies).
        Returns AUC in [0, 1]; values below 0.5 indicate negative correlation.

        Input:
            df: dataframe (typically the benchmark segment)
        Output:
            dict {variable: auc_value}
        '''
        auc_dict = {}
        y = df[self.dep].reset_index(drop=True).values.astype(float)

        for col in self.varlist:
            try:
                x = pd.to_numeric(df[col], errors='coerce').reset_index(drop=True).values
                mask = ~(np.isnan(x) | np.isnan(y))
                y_m = y[mask].astype(int)
                x_m = x[mask]
                n1 = int(y_m.sum())
                n0 = len(y_m) - n1
                if n1 == 0 or n0 == 0:
                    auc_dict[col] = 0.5
                    continue
                order = np.argsort(x_m, kind='stable')
                y_s = y_m[order]
                neg_cs = np.cumsum(y_s == 0)
                u = int(neg_cs[y_s == 1].sum())
                auc_dict[col] = round(u / (n1 * n0), 4)
            except Exception:
                auc_dict[col] = 0.5

        return auc_dict

    def write_summary(self, iv_dict, sheetname, iv_by_seg=None, auc_dict=None, auc_by_seg=None):
        '''
        Write a summary tab with attributes ordered by benchmark IV descending.
        If iv_by_seg is provided (dict of {seg_label: iv_dict}), an IV column
        per segment is appended beside the benchmark IV column.
        '''
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)
        worksheet.set_tab_color("black")

        iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Attribute', 'IV_benchmark'])

        if iv_by_seg:
            seg_labels = sorted(iv_by_seg.keys())
            for seg in seg_labels:
                iv_df[f'IV_{seg}'] = iv_df['Attribute'].map(iv_by_seg[seg])

        if auc_dict:
            iv_df['AUC_benchmark'] = iv_df['Attribute'].map(auc_dict)
        if auc_by_seg:
            for seg in sorted(auc_by_seg.keys()):
                iv_df[f'AUC_{seg}'] = iv_df['Attribute'].map(auc_by_seg[seg])

        iv_df = iv_df.sort_values('IV_benchmark', ascending=False).reset_index(drop=True)
        iv_df.index = iv_df.index + 1
        iv_df.index.name = 'Rank'

        iv_cols = [c for c in iv_df.columns if c.startswith('IV')]
        auc_cols = [c for c in iv_df.columns if c.startswith('AUC')]

        row += self.write_seg_line(worksheet, row, col, "Attribute IV & AUC Summary (Benchmark + All Segments)")
        max_attr_len = iv_df['Attribute'].str.len().max()
        self.writedf(iv_df, worksheet, row, col, indexing=True, cdt_fmt=iv_cols + auc_cols,
                     auc_col=auc_cols,
                     title="Attributes ordered by benchmark IV (descending)")
        worksheet.set_column(col, col, 6)
        worksheet.set_column(col + 1, col + 1, max_attr_len + 2)

    def bivar(self, df, segs, nbins=10, bins=None, draw=True):
        '''
        Calculate bivariate analysis and optionally save charts.

        Input:
            df: input dataframe
            segs: column name(s) for segmentation
            nbins: number of bins
            bins: pre-defined bins dict (optional)
            draw: whether to save charts
        '''
        res = {}
        bin_dict = {}
        for col in self.varlist_bivar:
            df_tmp = df[segs + [self.dep, col]].copy()
            if bins is None:
                df_tmp["grp"], bin_dict[col] = self.pct_rank_qcut(df_tmp[col], nbins=nbins)
            else:
                df_tmp["grp"], _ = self.pct_rank_qcut(df_tmp[col], nbins=nbins, bins=bins[col])

            bivar = df_tmp.groupby(segs + ["grp"]).agg({col: ['size', 'mean'], self.dep: ['mean']})
            bivar.columns = ['n', 'nmean', 'dep_rate']
            res[col] = bivar.unstack(level=0).reorder_levels([1, 0], axis=1)

            if draw:
                seglist = list(bivar.index.levels[0])
                plt.figure(figsize=(8, 4))
                for s in seglist:
                    plt.plot(bivar.loc[s].index, bivar.loc[s]['dep_rate'], label=s)
                lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncols=int(np.ceil(len(res) / 15)))
                plt.xlabel(col)
                plt.ylabel("bad_rate")
                plt.title(col)
                plt.savefig(f'tmp/{col}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()

        return res, bin_dict

    def chimerge_bins(self, series, dep_series, max_bins=10, init_bins=20):
        '''
        ChiMerge algorithm: iteratively merge adjacent bins with the lowest
        chi-square statistic until max_bins is reached.

        Input:
            series: feature column (reset index before passing)
            dep_series: binary dependent column (reset index before passing)
            max_bins: target maximum number of bins
            init_bins: number of initial fine quantile bins
        Output:
            list of bin edges (includes -inf and inf)
        '''
        _, edges = self.pct_rank_qcut(series, nbins=init_bins)

        # Build initial frequency table per bin
        labels = pd.cut(series, bins=edges, labels=False).fillna(-1).astype(int)
        df_tmp = pd.DataFrame({'grp': labels, 'dep': dep_series})
        valid_bins = sorted(b for b in df_tmp['grp'].unique() if b >= 0)

        freq = []
        for b in valid_bins:
            mask = df_tmp['grp'] == b
            ev = int(dep_series[mask].sum())
            tot = int(mask.sum())
            freq.append({'l': b, 'r': b + 1, 'events': ev, 'non_events': tot - ev, 'total': tot})

        # Merge until max_bins
        while len(freq) > max_bins:
            chi2_vals = []
            for i in range(len(freq) - 1):
                b1, b2 = freq[i], freq[i + 1]
                tot = b1['total'] + b2['total']
                tot_e = b1['events'] + b2['events']
                tot_ne = b1['non_events'] + b2['non_events']
                if tot_e == 0 or tot_ne == 0 or tot == 0:
                    chi2_vals.append(0.0)
                    continue
                e11 = b1['total'] * tot_e / tot
                e12 = b1['total'] * tot_ne / tot
                e21 = b2['total'] * tot_e / tot
                e22 = b2['total'] * tot_ne / tot
                chi2 = (
                    (b1['events'] - e11) ** 2 / max(e11, 1e-6) +
                    (b1['non_events'] - e12) ** 2 / max(e12, 1e-6) +
                    (b2['events'] - e21) ** 2 / max(e21, 1e-6) +
                    (b2['non_events'] - e22) ** 2 / max(e22, 1e-6)
                )
                chi2_vals.append(chi2)

            m = int(np.argmin(chi2_vals))
            b1, b2 = freq[m], freq[m + 1]
            merged = {
                'l': b1['l'], 'r': b2['r'],
                'events': b1['events'] + b2['events'],
                'non_events': b1['non_events'] + b2['non_events'],
                'total': b1['total'] + b2['total'],
            }
            freq = freq[:m] + [merged] + freq[m + 2:]

        # Reconstruct edge list from surviving bin right-edge indices
        final_edges = [-np.inf]
        for b in freq[:-1]:
            final_edges.append(edges[b['r']])
        final_edges.append(np.inf)
        return final_edges

    def chimerge(self, df, segs, max_bins=10, bins=None, draw=True):
        '''
        Calculate chi-square binning analysis and optionally save charts.

        Input:
            df: input dataframe
            segs: column name(s) for segmentation
            max_bins: maximum number of bins for chi-square merging
            bins: pre-defined bin edges dict (optional, skips chi-square computation)
            draw: whether to save charts
        '''
        res = {}
        bin_dict = {}

        for col in self.varlist_bivar:
            df_tmp = df[segs + [self.dep, col]].copy()
            try:
                if bins is None:
                    s = df_tmp[col].reset_index(drop=True)
                    d = df_tmp[self.dep].reset_index(drop=True)
                    bin_dict[col] = self.chimerge_bins(s, d, max_bins=max_bins)
                    edges = bin_dict[col]
                else:
                    edges = bins[col]

                df_tmp['grp'] = pd.cut(df_tmp[col], bins=edges, labels=False).fillna(-1)
                bv = df_tmp.groupby(segs + ['grp']).agg({col: ['size', 'mean'], self.dep: ['mean']})
                bv.columns = ['n', 'nmean', 'dep_rate']
                res[col] = bv.unstack(level=0).reorder_levels([1, 0], axis=1)

                if draw:
                    seglist = list(bv.index.levels[0])
                    plt.figure(figsize=(8, 4))
                    for s in seglist:
                        plt.plot(bv.loc[s].index, bv.loc[s]['dep_rate'], label=s)
                    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                     ncols=int(np.ceil(len(res) / 15)))
                    plt.xlabel(col)
                    plt.ylabel('bad_rate')
                    plt.title(f'{col} (chi-sq bins)')
                    plt.savefig(f'tmp/chi_{col}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                    plt.close()
            except Exception:
                if bins is None:
                    bin_dict[col] = [-np.inf, np.inf]

        return res, bin_dict

    def write_chimerge(self, df, segs, sheetname, bmk_seg, max_bins=10):
        '''
        Write chi-square binning worksheet to Excel, mirroring the bivar tab layout.
        '''
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)

        if len(segs) == 1:
            bin_dict = self.chimerge(df[df[segs[0]] == bmk_seg], segs=segs,
                                     max_bins=max_bins, draw=False)[1]
        else:
            bin_dict = self.chimerge(df[df[segs] == bmk_seg], segs=segs,
                                     max_bins=max_bins, draw=False)[1]

        chi_l = self.chimerge(df, segs=segs, bins=bin_dict)[0]

        for var in self.varlist_bivar:
            if var not in chi_l:
                continue
            row += self.write_seg_line(worksheet, row, col, var)
            worksheet.insert_image(row, col + len(chi_l[var].columns) + 5,
                                   f'tmp/chi_{var}.png', {'x_scale': 0.8, 'y_scale': 0.8})
            row += max(self.writedf(chi_l[var].fillna('Null'), worksheet, row, col,
                                    indexing=True, ratio_col=['dep_rate'], cdt_fmt=['dep_rate'])[0], 12) + 3

    def write_bivar(self, df, segs, sheetname, bmk_seg):
        '''
        Write bivar worksheet to Excel.
        '''
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)

        if len(segs) == 1:
            bin_dict = self.bivar(df[df[segs[0]] == bmk_seg], segs=segs, draw=False)[1]
        else:
            bin_dict = self.bivar(df[df[segs] == bmk_seg], segs=segs, draw=False)[1]

        bivar_l = self.bivar(df, segs=segs, bins=bin_dict)[0]

        for var in self.varlist_bivar:
            row += self.write_seg_line(worksheet, row, col, var)
            worksheet.insert_image(row, col + len(bivar_l[var].columns) + 5, f"tmp/{var}.png",
                                   {'x_scale': 0.8, 'y_scale': 0.8})
            row += max(self.writedf(bivar_l[var].fillna('Null'), worksheet, row, col,
                                    indexing=True, ratio_col=['dep_rate'], cdt_fmt=['dep_rate'])[0], 12) + 3

    def bivar_cat(self, df, segs, draw=True):
        '''
        Categorical bivariate analysis: groups by actual category values
        instead of numeric bins.

        Input:
            df: input dataframe
            segs: column name(s) for segmentation
            draw: whether to save bar charts
        Output:
            dict {col: unstacked_dataframe}
        '''
        res = {}
        cat_vars_bivar = [v for v in self.varlist_bivar if v in self.cat_vars]
        for col in cat_vars_bivar:
            df_tmp = df[segs + [self.dep, col]].copy()
            df_tmp[col] = df_tmp[col].astype(str).fillna('Null')
            bv = df_tmp.groupby(segs + [col]).agg({col: 'size', self.dep: 'mean'})
            bv.columns = ['n', 'dep_rate']
            res[col] = bv.unstack(level=0).reorder_levels([1, 0], axis=1)

            if draw:
                seglist = list(bv.index.get_level_values(0).unique())
                cats = list(bv.index.get_level_values(-1).unique())
                x = np.arange(len(cats))
                width = 0.8 / max(len(seglist), 1)
                _, ax = plt.subplots(figsize=(max(8, len(cats) * 0.6 + 2), 4))
                for i, s in enumerate(seglist):
                    vals = [bv.loc[(s, c), 'dep_rate'] if (s, c) in bv.index else np.nan for c in cats]
                    ax.bar(x + i * width, vals, width=width, label=s)
                ax.set_xticks(x + width * (len(seglist) - 1) / 2)
                ax.set_xticklabels(cats, rotation=45, ha='right')
                ax.set_xlabel(col)
                ax.set_ylabel('bad_rate')
                ax.set_title(col)
                lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig(f'tmp/cat_{col}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()

        return res

    def write_bivar_cat(self, df, segs, sheetname):
        '''
        Write categorical bivar worksheet to Excel.
        '''
        cat_vars_bivar = [v for v in self.varlist_bivar if v in self.cat_vars]
        if not cat_vars_bivar:
            return

        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)

        bivar_cat_l = self.bivar_cat(df, segs=segs)

        for var in cat_vars_bivar:
            if var not in bivar_cat_l:
                continue
            row += self.write_seg_line(worksheet, row, col, var)
            img_path = f'tmp/cat_{var}.png'
            if os.path.exists(img_path):
                worksheet.insert_image(row, col + len(bivar_cat_l[var].columns) + 5,
                                       img_path, {'x_scale': 0.8, 'y_scale': 0.8})
            row += max(self.writedf(bivar_cat_l[var].fillna('Null'), worksheet, row, col,
                                    indexing=True, ratio_col=['dep_rate'], cdt_fmt=['dep_rate'])[0], 12) + 3

    def run(self):
        '''
        Run the full attribute review process.
        '''
        df = self.df.copy()
        if self._no_seg:
            df[self._DUMMY_SEG_COL] = self._DUMMY_SEG_VAL
        for seg in self.segs:
            df[seg] = df[seg].astype(str)

        # Compute IV and AUC on benchmark segment; sort varlist by IV descending
        bmk_df = df[df[self.segs[0]] == self.bmk_seg[0]]
        iv_dict = self.calc_iv(bmk_df)
        auc_dict = self.calc_auc(bmk_df)
        self.varlist = sorted(self.varlist, key=lambda x: iv_dict.get(x, 0), reverse=True)

        # Compute IV and AUC for every non-benchmark segment
        iv_by_seg = {}
        auc_by_seg = {}
        if not self._no_seg:
            for seg_val in sorted(df[self.segs[0]].unique()):
                if seg_val == self.bmk_seg[0]:
                    continue
                seg_df = df[df[self.segs[0]] == seg_val]
                iv_by_seg[seg_val] = self.calc_iv(seg_df)
                auc_by_seg[seg_val] = self.calc_auc(seg_df)

        self.write_summary(iv_dict, "Summary", iv_by_seg=iv_by_seg or None,
                           auc_dict=auc_dict, auc_by_seg=auc_by_seg or None)
        nonzero_vars = [v for v in self.varlist[:300] if iv_dict.get(v, 0) > 0]
        self.varlist_bivar = nonzero_vars

        # Numeric variables → quantile-bin bivar and chimerge sheets
        num_vars_bivar = [v for v in self.varlist_bivar if v in self.num_vars]
        if num_vars_bivar:
            self.varlist_bivar = num_vars_bivar
            self.write_bivar(df, [self.segs[0]], "Bivar", self.bmk_seg[0])
            self.write_chimerge(df, [self.segs[0]], "Chimerge", self.bmk_seg[0])

        # Categorical variables → categorical bivar sheet
        self.varlist_bivar = nonzero_vars  # restore non-zero IV list for cat filter
        self.write_bivar_cat(df, [self.segs[0]], "Bivar_Cat")

        self.workbook.close()
