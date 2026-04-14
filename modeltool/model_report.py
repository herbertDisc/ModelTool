#########################################################
# Description: Generate the model report
# Auther: Xinyue He
# Created date: 2024.08
#########################################################

'''
Update Log:
2024.09.03
    1. Supports pickle file as model object input.
    2. Supports numeric input for benchmark segment input.
    3. All plot legends now locate on the left of the plot with maximum 15 for each colums.
    4. Change all fonts to Microsoft Yahei light.
    5. ResponseRate in KS tables will have conditional formatting.

2024.09.20
    1. Fix the bin cut issue when the len of the bin is smaller than 3.
'''


import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import xlsxwriter
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class model_reporter():
    '''
    class to generate the model report
    Input:
        csvfile: string
            file path of the whole dataset that contains all the model attributes, dependence and seg
        segs: list of string
            list of column names that define the segment
        bmk_seg: list of strings or numbers
            segment that selected as the bmk
        dep: string
            column name of the dependence variable
        model_path: string
            file path of the model object
        model_config: dict
            modeling training hyperparameters
        outputpath: string
            file path for saving the final output
        model_type: string xgb or lgb
            model type
        scring: boolean
            if scoring process is needed
        probs: string default probs
            if scring is false, column name of the model probability prediction
        scr: string default scr
            if scring is false, column name of the model odds prediction
        scr_logbase: string default None
            if scring is True, odds transfering natural log base

    Output:
        tmp: folder
            folder that contains all the png generated
        excel:
            final report        

    sample code:
    hyper_params ={'colsample_bytree': 0.9518705522909501,
    'gamma': 2.680171627119694,
    'learning_rate': 0.1753514970430317,
    'max_depth': 2,
    'min_child_weight': 6,
    'n_estimators': 140,
    'reg_alpha': 6.332504289145607,
    'reg_lambda': 83.94000795191253,
    'subsample': 0.9637043603586488,
    'seed':1024, 
    'scale_pos_weight':4,
    'early_stopping_rounds':20,
    'eval_metric':'auc'}

    # object declaration
    engine = model_reporter(csvfile="../reorder_model_01/final_data.csv",
                            segs=['seg','M'],
                            dep='t7_cnt',
                            bmk_seg=['train',1],
                            model_path='../reorder_model_01/lcless2_final_model_v1.json',
                            model_config = hyper_params,
                            outputpath='model_report.xlsx',
                            scring=True,
                            scr_logbase=2.3)

    # run
    engine.run()
    '''
    def __init__(self,
                 csvfile,
                 segs,
                 bmk_seg,
                 dep,
                 model_path,
                 model_config,
                 outputpath,
                 model_type='xgb',
                 scring=False,
                 probs='probs',
                 scr='scr',
                 scr_logbase = None) -> None:
        self.csvfile = csvfile
        self.segs = segs
        self.dep = dep
        self.bmk_seg = tuple([str(x) for x in bmk_seg])
        self.outputpath = outputpath
        self.probs = probs
        self.scr = scr
        self.scr_logbase = scr_logbase

        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.model_config = model_config
        self.scring = scring
        self._load_model()

        self.seg_title_format = None
        self.title_format = None
        self.header_format = None
        self.idx_format = None
        self.body_format = None
        self.ratio_format = None
        self._create_excel()
        os.makedirs('tmp', exist_ok=True)

    def _load_model(self):
        '''
        Function to load model.
        '''
        if self.model_type == 'xgb':
            self.model = xgb.Booster(model_file=self.model_path)
            self.varlist = self.model.feature_names
            imp =self.model.get_score(importance_type="gain")
            self.varlist_sort = sorted(imp, key=imp.get, reverse=True)
            self.cat_varlist = [self.varlist_sort[i] for i in range(len(self.varlist_sort)) if self.model.feature_types[i]=='category']
        else:
            self.model = lgb.Booster(model_file=self.model_path)
            self.varlist = self.model.feature_name()
            self.varlist_sort = [f for f, _ in sorted(zip(self.model.feature_name(),
                                                    self.model.feature_importance(importance_type='gain')),
                                                key=lambda x: x[1], reverse=True)]
            self.cat_varlist = []
        self.num_varlist = [f for f in self.varlist_sort if f not in self.cat_varlist]


    def _create_excel(self):
        '''
        Initialization of creating excel
        '''
        self.workbook = xlsxwriter.Workbook(self.outputpath,options={'nan_inf_to_errors': True})
        self._load_format()


    def _load_format(self):
        '''
        store all formats of the excel writter
        '''
        self.seg_title_format = self.workbook.add_format({
                "bold": True,
                "font": "Microsoft YaHei Light",
                "font_size":15,
                "text_wrap": False,
                "valign": "top",
                "fg_color": "#FABF8F",
                # "border": 1,
            }
        )
        self.title_format = self.workbook.add_format({
                "bold": True,
                "font": "Microsoft YaHei Light",
                "font_size":13,
                "text_wrap": False,
                "valign": "top",
                # "fg_color": "#C5D9F1",
                # "border": 1,
            }
        )
        self.header_format = self.workbook.add_format({
                "bold": True,
                "font": "Microsoft YaHei Light",
                "text_wrap": True,
                "valign": "top",
                "fg_color": "#C5D9F1",
                "border": 1,
            }
        )
        self.idx_format = self.workbook.add_format({
                "bold": False,
                "font": "Microsoft YaHei Light",
                "valign": "top",
                'text_wrap':False,
                # "fg_color": "#D7E4BC",
                "border": 1,})
        self.body_format = self.workbook.add_format({
                "bold": False,
                "font": "Microsoft YaHei Light",
                "valign": "top",
                'text_wrap':False,
                # "fg_color": "#D7E4BC",
                "border": 1,})
        self.ratio_format = self.workbook.add_format({"bold": False,
                "text_wrap": True,
                "font": "Microsoft YaHei Light",
                "valign": "top",
                # "fg_color": "#D7E4BC",
                "border": 1,
                'num_format':'0.00%',
                })


    def write_seg_line(self, worksheet, row, col, title):
        '''
        write a single line
        '''
        worksheet.write(row, col, title)
        worksheet.set_row(row, 15, self.seg_title_format)
        return 2

    def writedf(self, df, worksheet, row, col, indexing=False, ratio_col=None, cdt_fmt=None, title=None):
        '''
        write a whole dataframe into wroksheet
        '''
        if ratio_col is None:
            ratio_col = []
        if cdt_fmt is None:
            cdt_fmt = []
        def writerow(ws,row,col,arr,fmt):
            for idx,val in enumerate(arr):
                ws.write(row,col+idx,val,fmt)
        def writecol(ws,row,col,arr,fmt):
            for idx,val in enumerate(arr):
                ws.write(row+idx,col,val,fmt)
        # Get the xlsxwriter workbook and worksheet objects.
        # writer.book.add_format(header_format)

        if title is not None:
            worksheet.write(row,col,title,self.title_format)
            row += 1
        idx_len = 0
        
        col_len = df.columns.nlevels

        # if index is needed
        if indexing:
            idx_len = df.index.nlevels
            writerow(worksheet,row+col_len-1,col,df.index.names,self.header_format)
            for idx,val in enumerate(df.index.values):
                if idx_len == 1:
                    val = [val]
                writerow(worksheet,row+idx+col_len,col,val,self.idx_format)

        # Write body
        for col_num, value in enumerate(df.columns.values):
            # header for the column
            if col_len > 1:
                writecol(worksheet, row, col+idx_len+col_num, value, self.header_format)
            else:
                worksheet.write(row, col+idx_len+col_num, value, self.header_format)
            if value in ratio_col:
                writecol(worksheet, row+col_len, col+idx_len+col_num, df[value], self.ratio_format)
            else:
                writecol(worksheet, row+col_len, col+idx_len+col_num, df[value], self.body_format)

            if value in cdt_fmt:
                worksheet.conditional_format(row+1,col+idx_len+col_num,
                                             row+len(df[value]),col+idx_len+col_num,
                                             {'type': 'data_bar'})
    
        df_shape = [df.shape[0],df.shape[1]]
        df_shape[1] += idx_len
        return df_shape

    def writedf_para(self, df_list, worksheet, row, col, indexing=False, ratio_col=None, cdt_fmt=None, title=None):
        '''
        write a list of dataframe in parallel
        Input:
            df_list: list
                List of dataframe to be written
            worksheet: worksheet
                the worksheet(tab)
            row: int
                which row in the worksheet to start with
            col: int
                which col in the worksheet to start with
            indexing:
                where to write the index of df
            ratio_col:
                the column to be shown as ratio
            cdt_fmt:
                the column to be shown with conditional formatting
            title:
                title of the df if needed
        Output:
            max_row:
                the row where the df ends
            cur_col:
                the current column
        '''
        if ratio_col is None:
            ratio_col = []
        if cdt_fmt is None:
            cdt_fmt = []
        cur_col = col
        max_row = 0
        for idx, df in enumerate(df_list):
            t = None if title is None else title[idx]
            row_tmp = row+1 if t is None else row
            res = self.writedf(df, worksheet, row_tmp, cur_col, indexing=indexing, ratio_col=ratio_col, cdt_fmt=cdt_fmt, title=t)
            cur_col += res[1] + 1
            max_row = max(max_row,res[0])
        return (max_row,cur_col)

    def get_perf(self, df, seg_col):
        '''
        Function to calculate the performance
        '''
        def auc_ks(df):
            fpr, tpr, _ = metrics.roc_curve(df.iloc[:,0], df.iloc[:,1])
            auc = metrics.auc(fpr, tpr)
            ks = max(tpr - fpr)
            plt.plot(fpr, tpr)
            return pd.Series([auc, ks])

        res = df.groupby(seg_col)[[self.dep,self.probs]].apply(auc_ks)
        lgd = plt.legend(['%s (AUC = %0.3f)'% (res.index[i],res.values[i][0]) for i in range(len(res))], loc='center left', bbox_to_anchor=(1, 0.5), ncols = int(np.ceil(len(res)/15)))
        res = res.sort_index()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig('tmp/perf_%s.png' % (seg_col), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        return pd.DataFrame(res.values,index=res.index,columns=['AUC','KS'])

    def pct_rank_qcut(self, series, nbins=None, bins = None):
        '''
        Function for binning the data
        '''
        if bins is None:
            _, bins = pd.qcut(series, nbins, labels = False, retbins = True,duplicates='drop')
            if len(bins) > 2:
                bins[0] = -np.inf
                bins[-1] = np.inf
            else:
                bins=np.insert(bins,[0,len(bins)],[-np.inf,np.inf])
            

        grp = pd.cut(series, bins=bins, labels = False)
        grp = grp.fillna(-1)

        return grp, list(bins)

    def KS(self, df, bins=10, custom_bins = None, sort_in=False):
        """
        Function to generate KS-table
        Input:
            df: 数据集
            dep: target
            score: score
            bins: 分箱个数
        """
        if custom_bins is not None:
            df['score_rank'] = pd.cut(df[self.probs], bins=custom_bins, include_lowest=True)
        else:
            df['score_rank'], custom_bins = self.pct_rank_qcut(df[self.probs], nbins=bins,bins=custom_bins)
        #KS Analysis on dev sample
        Total = df[self.dep].count()
        TotalResponder = df[self.dep].sum()
        TotalNonResponder = Total-TotalResponder
        Total_Int = df.groupby(['score_rank'],sort=False,observed=False)['score_rank'].count()
        Responder = df.groupby(['score_rank'],sort=False,observed=False)[self.dep].sum()
        NonResponder = Total_Int-Responder

        Min_Score = df.groupby(['score_rank'],sort=False,observed=False)[self.probs].min()
        Mean_Score = df.groupby(['score_rank'],sort=False,observed=False)[self.probs].mean()
        Max_Score = df.groupby(['score_rank'],sort=False,observed=False)[self.probs].max()
        Min_Odds = df.groupby(['score_rank'],sort=False,observed=False)[self.scr].min()

        Total_Int = Total_Int.sort_index(ascending=sort_in)
        Responder = Responder.sort_index(ascending=sort_in)
        NonResponder = NonResponder.sort_index(ascending=sort_in)

        Min_Score = Min_Score.sort_index(ascending=sort_in)
        Mean_Score = Mean_Score.sort_index(ascending=sort_in)
        Max_Score = Max_Score.sort_index(ascending=sort_in)
        Min_Odds = Min_Odds.sort_index(ascending=sort_in)


        PercResponder = (Responder/TotalResponder)
        CumPercResponder = (np.cumsum(Responder)/TotalResponder)    
        CumPercNonResponder = (np.cumsum(NonResponder)/TotalNonResponder)
        ResponseRate = (Responder/Total_Int)

        KS = CumPercResponder-CumPercNonResponder
        lift = (ResponseRate)/(TotalResponder/Total)
        cum_lift = (np.cumsum(Responder)/np.cumsum(Total_Int))/(TotalResponder/Total) 
        dev_name = [Total_Int,Min_Score,Mean_Score,Max_Score,Min_Odds,Responder,PercResponder,CumPercResponder,CumPercNonResponder,ResponseRate,KS,lift,cum_lift]
        columns = ["Total_Int","Min_Score","Mean_Score","Max_Score","Min_Odds","Responder","PercResponder","CumPercResponder","CumPercNonResponder","ResponseRate","KS","lift","cum_lift"]
        KS_dev = pd.concat(dev_name, keys=columns, axis=1)
        S=pd.Series([Total,df[self.probs].min(),df[self.probs].mean(),df[self.probs].max(),df[self.scr].min(),TotalResponder,1,"","",(TotalResponder/Total),abs(KS).max(),"",""]
                    ,index=columns)
        KS_dev = KS_dev._append(S,ignore_index=True)

        fpr_dev = dict()
        tpr_dev = dict()
        roc_auc_dev = dict() 
        fpr_dev, tpr_dev, _ = roc_curve(df[self.dep], df[self.probs])
        roc_auc_dev = auc(fpr_dev, tpr_dev)

        return KS_dev,roc_auc_dev, custom_bins

    def _psi(self, df, col, bmk_pct, bmk_bins, grp=None, numerical=True):
        '''
        function to calculate single column psi
        '''
        # in case new data is missing any bins
        res = []
        for i in df[grp].unique():
            df_tmp = bmk_pct.copy()
            df_tmp[grp] = i
            res.append(df_tmp)
        df_bmk = pd.concat(res)
        # get bins
        if numerical:
            df['bins'],_ = self.pct_rank_qcut(df[col], bins=bmk_bins)
            df_pct = (df.groupby([grp, 'bins'])[col].size()/df.groupby(grp)[col].size()).to_frame().reset_index()
            df_pct = df_pct.merge(df_bmk, on=[grp, 'bins'],how='right').fillna(0.000001)
        else:
            df_pct = (df.groupby([grp,col]).size()/df.groupby(grp)[col].size()).to_frame().reset_index()
            df_pct = df_pct.merge(df_bmk, on=[col],how='right').fillna(0.000001)
        df_pct['woe'] = (df_pct[col] - df_pct['bmk']) * np.log(df_pct[col]/df_pct['bmk'])
        df_pct = df_pct.set_index([grp,'bins'])
        dfout_pct = df_pct[col].unstack().round(4)
        dfout_pct.columns = [str(x)+'_pct' for x in dfout_pct.columns]
        df_bin_psi = df_pct['woe'].unstack().round(4)
        out = pd.concat([df_bin_psi.sum(axis=1).round(4).rename('PSI'),df_bin_psi,dfout_pct],axis=1)
        return out

    def variable_psi(self, df, grp, cols, cate_cols, bmk_bin_dict, bmk_pct_dict):

        dfout = []
        for col in cols:
            numeric = False if col in cate_cols else True
            df_tmp = df[[grp,col]].copy()
            bmk_bin = bmk_bin_dict[col]
            bmk_pct = bmk_pct_dict[col].rename(columns={'pct':'bmk'})
            dfout.append(self._psi(df=df_tmp, col=col, bmk_pct=bmk_pct, bmk_bins=bmk_bin, grp = grp, numerical=numeric)['PSI'].rename(col))
        out = pd.concat(dfout, axis=1).reset_index()
        return out

    def get_bmk_bins(self, df, cols, nbins):
        '''
        function to get the benchmark bins and corresponding percent
        '''
        out_bins = {}
        out_pct = {}

        # for each attribute and score get the bin and pct
        for col in cols:
            grp, out_bins[col] = self.pct_rank_qcut(df[col], nbins=nbins)
            pct = (grp.groupby(grp).size()/len(grp))
            out_pct[col] = pct
        
        return out_bins, out_pct

    def benchmark(self, df, cols, nbins):
        out_bins = {}
        out_pct = {}
        # for each attribute and score get the bin and pct
        for col in cols:
            grp, out_bins[col] = self.pct_rank_qcut(df[col], nbins=nbins)
            pct = (grp.groupby(grp).size()/len(grp))
            out_pct[col] = pct

        df_out_bins = pd.Series(out_bins).to_frame().reset_index()
        df_out_bins.columns = ['attribute','bins']
        df_out_bins['bins'] = df_out_bins['bins'].apply(json.dumps)
        df_out_pct = pd.concat(out_pct, names=['attribute','bins']).rename('pct').reset_index()

        out_pct = {key: value.drop(columns='attribute') for key, value in df_out_pct.groupby('attribute')}
        return out_bins, out_pct

    def bivar(self, df, segs, nbins=10, bins=None, draw=True):
        '''
        Function to calculate bivar as well as saving the bivar chart
        Input:
            df:
                input dataframe
            segs:
                column name for segmentation
            nbins:
                number of bins to calculate bivar
            draw:
                whether to draw the chart
        '''
        res = {}
        bin_dict = {}
        for col in self.num_varlist:
            df_tmp = df[segs + [self.dep, col]].copy()
            if bins is None:
                df_tmp["grp"], bin_dict[col] =  self.pct_rank_qcut(df_tmp[col], nbins=nbins)
            else:
                df_tmp["grp"],_ = self.pct_rank_qcut(df_tmp[col], nbins=nbins, bins=bins[col])
    
            bivar = df_tmp.groupby(segs+["grp"]).agg({col:['size','mean'],self.dep:['mean']})
            bivar.columns = ['n', 'nmean', 'dep_rate']
            res[col] = bivar.unstack(level=0).reorder_levels([1,0],axis=1)

            if draw:
                seglist = list(bivar.index.levels[0])
                plt.figure(figsize=(8,4))
                for s in seglist:
                    # plt.plot(bivar.loc[s]['nmean'],bivar.loc[s]['dep_rate'],label=s)
                    # plt.plot(list(range(len(bivar.loc[s]['nmean']))),bivar.loc[s]['dep_rate'],label=s)
                    plt.plot(bivar.loc[s].index,bivar.loc[s]['dep_rate'],label=s)
                lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncols = int(np.ceil(len(res)/15)))
                plt.xlabel(col)
                plt.ylabel("bad_rate")
                plt.title(col)
                plt.savefig(f'tmp/{col}.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()
        return res, bin_dict

    def write_summary(self, df, segs, dep):
        '''
        Function to write all the summary worksheets
        Input:
            df:
                Input dataframe
            segs:
                list of segments
            dep:
                dependent column name
        '''
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet('Summary')
        worksheet.set_tab_color("black")

        ############# Overview of the population by segments
        summary_df = df.groupby(segs)[dep].agg(['size','sum'])
        summary_df.columns = ['Total_count',dep]
        summary_df['indep_rate'] = summary_df[dep]/summary_df['Total_count']
    
        row += self.write_seg_line(worksheet, row, col, 
                                    "Population overview")
        row += self.writedf(summary_df, worksheet, row, col,indexing=True,ratio_col=['dep_rate']
                            ,title="1.Model dependence overview by segments")[0] + 3

        ############# model information
        row += self.write_seg_line(worksheet, row, col, 
                                    "Model Performance (AUC, KS) by defined segments")
        perf_df_first = self.get_perf(df, self.segs[0])

        worksheet.insert_image(row, col, f'tmp/perf_{self.segs[0]}.png', {'x_scale': 0.8, 'y_scale': 0.8})
        row += 20
        row += self.writedf(perf_df_first, worksheet, row, col,indexing=True,ratio_col=['AUC']
                    ,title="2.Model Performance by the first selected segments")[0] +3

        if len(segs) > 1:
            perf_df = self.get_perf(df, self.segs)

            worksheet.insert_image(row, col, f'tmp/perf_{self.segs}.png', {'x_scale': .8, 'y_scale': .8})
            row += 20
            row += self.writedf(perf_df, worksheet, row, col, indexing=True, ratio_col=['AUC']
                                    ,title="2.Model Performance by all the input segments")[0] +3


        ############# model attributes with importance
        row += self.write_seg_line(worksheet, row, col,
                                    "Model attributes with the attribute importance")

        if self.model_type == 'xgb':
            imp = self.model.get_score(importance_type="gain")
            imp_df = (
                pd.DataFrame(list(imp.items()), columns=["Attr", "Imp"])
                .sort_values("Imp", ascending=False)
                .reset_index(drop=True)
            )
        else:
            imp_df = pd.DataFrame({"Attr": self.model.feature_name(),
                                   "Imp": self.model.feature_importance(importance_type='gain')
                                }).sort_values("Imp", ascending=False)
        imp_df["Imp"] = imp_df["Imp"] / imp_df["Imp"].sum()
        max_len = imp_df['Attr'].str.len().max()
        imp_df = imp_df.set_index('Attr')
        self.writedf(imp_df, worksheet, row, col, indexing=True, ratio_col=['Imp']
                     ,title="1.Model attributes and importance")


        worksheet.set_column(0,0,1)
        worksheet.set_column(col,col,max_len + 1)

    def write_ks(self, df, segs, sheetname, bmk_seg):
        '''
        Function to write KS-table worksheet
        '''
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)

        res = df.groupby(segs).apply(self.KS, include_groups=False)
        bmk_bin = res.loc[bmk_seg][2]
        res_by_bmk = df.groupby(segs).apply(self.KS, custom_bins = bmk_bin, include_groups=False)

        # write bmk
        row += self.write_seg_line(worksheet, row, col, f"Benchmark {bmk_seg} Gains Table")
        row += self.writedf(res.loc[bmk_seg][0],
                            worksheet,
                            row, col+7,
                            indexing=True,
                            ratio_col=['PercResponder', 'CumPercResponder', 'CumPercNonResponder', 'ResponseRate'],
                            cdt_fmt=[ 'ResponseRate', 'lift'],
                            title= f"Benchmark Segment:{bmk_seg} Gains table"
                            )[0] + 3

        # write each seg by bmk's bin + its own bin
        for i,value in enumerate(res.values):
            if res.index[i] != bmk_seg:
                row += self.write_seg_line(worksheet, row, col, f"Segment {res.index[i]} Gains Table with different bincuts")
                row += self.writedf_para([res_by_bmk.values[i][0],value[0]],
                                         worksheet,
                                         row, col,
                                         indexing=True,
                                         ratio_col=['PercResponder', 'CumPercResponder', 'CumPercNonResponder', 'ResponseRate'],
                                         cdt_fmt=[ 'ResponseRate', 'lift'],
                                         title=[f"Segment {res.index[i]} Gains table, bin cut by benchmark {bmk_seg} defined bins",
                                                f"Segment {res.index[i]} Gains table, bin cut by the segment distribution."]
                                        )[0] + 3

    def write_bivar(self,df,segs, sheetname, bmk_seg):
        '''
        Function to write bivar worksheet
        '''
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)

        if len(segs)==1:
            bin_dict = self.bivar(df[df[segs[0]]==bmk_seg], segs = segs, draw=False)[1]
        else:
            bin_dict = self.bivar(df[df[segs]==bmk_seg], segs = segs, draw=False)[1]

        # dup seg columns in order to get the group name within the group by function
        bivar_l= self.bivar(df, segs=segs, bins=bin_dict)[0]

        for var in self.varlist_sort:
            row += self.write_seg_line(worksheet, row, col, var)
            worksheet.insert_image(row, col+len(bivar_l[var].columns)+5, f"tmp/{var}.png" ,{'x_scale': 0.8, 'y_scale': 0.8})
            row += max(self.writedf(bivar_l[var].fillna('Null'), worksheet, row, col, indexing=True, ratio_col=['Dep_rate'], cdt_fmt=['Dep_rate'])[0],12) + 3

    def write_means(self,df,seg, sheetname):
        '''
        function to write means-table worksheet
        '''
        def means_report(df, varlist, percentiles=None):
            if percentiles is None:
                percentiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            res = df[varlist].describe(percentiles=percentiles).T
            res = res.rename(columns={'count':'n'})
            res.insert(loc=0,column = 'total_count',value=len(df))
            res.insert(loc=2,column='%nmiss',value=df[varlist].isna().mean(axis=0).T.values)
            return res
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)

        res = df.groupby(seg).apply(means_report,varlist = self.varlist_sort, include_groups=False).reset_index()

        max_row,max_col = self.writedf(res, worksheet, row, col,ratio_col = ['%nmiss'], indexing=False)
        worksheet.autofilter(1, 1, max_row, max_col)

    def write_psi(self, df, seg, bmk_seg, sheetname):
        benchmark_bin, benchmark_pct = self.benchmark(df[df[seg[0]] == bmk_seg[0]], self.varlist_sort, nbins=10)

        res = self.variable_psi(df=df,
                                grp=seg[0],
                                cols=self.varlist_sort,
                                cate_cols=[],
                                bmk_bin_dict=benchmark_bin,
                                bmk_pct_dict=benchmark_pct)
        row = 1
        col = 1
        worksheet = self.workbook.add_worksheet(sheetname)
        max_row,max_col = self.writedf(res, worksheet, row, col,ratio_col = self.varlist, indexing=False)
        worksheet.autofilter(1, 1, max_row, max_col)

    def run(self):
        '''
        object entrance to runn all the processes
        '''
        df = pd.read_csv(self.csvfile)
        for seg in self.segs:
            df[seg] = df[seg].astype(str)

        if self.scring:
            if self.model_type == 'xgb':
                data_m = xgb.DMatrix(df[self.varlist])
            else:
                data_m = df[self.varlist]
            df['probs'] = self.model.predict(data_m)
            df['scr'] = 650 + round(30*(np.log((1-df['probs'])/df['probs'])-np.log(self.scr_logbase))/np.log(2))

        self.write_summary(df,self.segs,self.dep)
        self.write_ks(df,self.segs[0],"KS_by_first_seg",self.bmk_seg[0])
        self.write_bivar(df,[self.segs[0]],"Bivar",self.bmk_seg[0])
        self.write_means(df,self.segs[0],"means_table")
        self.write_psi(df,self.segs,self.bmk_seg,"PSI")
        
        if len(self.bmk_seg) >1:
            self.write_ks(df,self.segs,"Appdx.All_segs_perf",self.bmk_seg)
            self.write_means(df,self.segs,"Appdx.means_tabl_seg_combo")
        self.workbook.close()
