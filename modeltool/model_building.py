from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import hyperopt
import optuna
import pandas as pd
import os, random
import gc
from .model_report import model_reporter
from datetime import datetime

def auc_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    mdl_auc = auc(fpr, tpr)
    mdl_ks = max(tpr - fpr)
    return [mdl_auc, mdl_ks]

def train_single_model(df_dev,
                       df_oot,
                       dep,
                       param,
                       varlist=None,
                       weight_col = None,
                       test_size= 0.33,
                       seed=123,
                       verbose=False,
                       generate_report=False,
                       additional_target=[]):
    '''
    Sample param:
    {'colsample_bytree': 0.9214981176986129,
     'gamma': 1.8876015000476731,
     'learning_rate': 0.19778523299652384,
     'max_depth': 3,
     'min_child_weight': 4,
     'n_estimators': 30,
     'reg_alpha': 8.804469172260143,
     'reg_lambda': 37.69628182077377,
     'subsample': 0.8749124885305246,
     'seed': 12345,
     'early_stopping_rounds': 15,
    }
    '''
    df_dev = df_dev.copy()
    if varlist is None:
        varlist = [x for x in df_dev.columns if x != dep]

    if weight_col is None:
        df_dev['_tmp_weight'] = 1
        weight_col = '_tmp_weight'

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(df_dev[varlist], df_dev[dep],df_dev[weight_col], test_size=test_size, random_state=seed)

    # For some defualt value if not set in param
    default_param = {'importance_type': "total_gain",
                     'eval_metric': 'auc'}
    for key, value in default_param.items():
        param.setdefault(key, value)

    xgb_model = XGBClassifier(**param)
    xgb_model.set_params(early_stopping_rounds=8)
    xgb_model = xgb_model.fit(X_train, y_train,sample_weight=w_train,
                                eval_set=[(X_train[varlist], y_train),
                                          (X_test[varlist], y_test),
                                            ],
                                verbose=verbose)

    train_pred = xgb_model.predict_proba(X_train[varlist])[:,1]
    test_pred = xgb_model.predict_proba(X_test[varlist])[:,1]
    perf = {}
    for key,datain in df_oot.items():
        datain = datain.copy()
        datain.loc[:,'pred'] = xgb_model.predict_proba(datain[varlist])[:,1]
        perf[key] = auc_ks(datain[dep],datain['pred'])
    perf['dev'] = auc_ks(y_train,train_pred)
    perf['test'] = auc_ks(y_test,test_pred)

    if generate_report:
        os.makedirs('tmp_single_model', exist_ok=True)
        nowtime = datetime.now().strftime("%y%m%d%H%M%S")
        X_train, X_test, y_train, y_test = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=seed)
        df_report = df_dev.copy()
        df_report.loc[X_train.index,'seg'] = 'dev'
        df_report.loc[X_test.index,'seg'] = 'val'
        df_oot_all = pd.concat([df.assign(seg=key) for key, df in df_oot.items()], ignore_index=True)
        df_report = pd.concat([df_report.reset_index(drop=True), df_oot_all.reset_index(drop=True)],axis=0)
        df_report['seg'] = df_report['seg'].fillna('oot')
        df_report[['seg', dep]+additional_target+varlist].to_csv("./tmp_single_model/_tmp_output_variable.csv")
        xgb_model.save_model(f"./tmp_single_model/_tmp_model_{nowtime}.json")
        log_score = 1/(df_dev[dep].sum()/len(df_dev)) -1
        mdlr =model_reporter("./tmp_single_model/_tmp_output_variable.csv",['seg'],['dev'],dep,f"./tmp_single_model/_tmp_model_{nowtime}.json",{},f'./tmp_single_model/model_{nowtime}.xlsx', scring=True, scr_logbase=log_score)
        mdlr.run()
    return xgb_model, perf

def hyperopt_search(df_dev,
                    df_oot,
                    dep,
                    param_space,
                    ntrials,
                    weight_col=None,
                    varlist=None,
                    test_size= 0.33,
                    seed=123):

    def hyperopt_objective(params):

        model_param = {
            'max_depth':int(float(params['max_depth'])),
            'learning_rate':params['learning_rate'],
            'n_estimators':int(float(params['n_estimators'])),
            'min_child_weight':int(float(params['min_child_weight'])),
            'subsample':params["subsample"],
            'colsample_bytree':params["colsample_bytree"],
            'gamma':params["gamma"],
            'reg_alpha':params["reg_alpha"],
            'reg_lambda':params["reg_lambda"],
            'scale_pos_weight':params["scale_pos_weight"],
        }

        _,perf = train_single_model(df_dev=df_dev,
                                    df_oot=df_oot,
                                    dep=dep,
                                    weight_col=weight_col,
                                    param=model_param,
                                    varlist=varlist,
                                    test_size= test_size,
                                    seed=seed)

        return 1 - perf['test'][0]

    best = hyperopt.fmin(hyperopt_objective,
                         space=param_space,
                         algo=hyperopt.tpe.suggest,
                         max_evals=ntrials,
                         trials=hyperopt.Trials())

    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])
    best['n_estimators'] = int(best['n_estimators'])
    # best['scale_pos_weight'] = param_space['scale_pos_weight']

    model,perf = train_single_model(df_dev=df_dev,
                                    df_oot=df_oot,
                                    dep=dep,
                                    param=best,
                                    weight_col=weight_col,
                                    varlist=varlist,
                                    test_size= test_size,
                                    seed=seed)
    
    return model, best, perf

def iter_random_search(df_dev,
                        df_oot,
                        dep,
                        param_space,
                        ntrials,
                        varlist,
                        weight_col=None,
                        stop=30,
                        keep_importance = 1,
                        test_size= 0.33,
                        seed=123,
                        generate_report=False):

    random_integer = random.randint(10000, 20000)
    os.makedirs('iter_randsearch/var', exist_ok=True)
    os.makedirs('iter_randsearch/model', exist_ok=True)
    os.makedirs('iter_randsearch/report', exist_ok=True)
    log_score = 1/(df_dev[dep].sum()/len(df_dev)) -1
    var_round_len = len(varlist) - 1
    while len(varlist) > stop and var_round_len != len(varlist):
        var_round_len = len(varlist)

        model, param, perf = hyperopt_search(df_dev=df_dev,
                                             df_oot=df_oot,
                                             dep=dep,
                                             param_space=param_space,
                                             ntrials=ntrials,
                                             varlist=varlist,
                                             weight_col=weight_col,
                                             test_size= test_size,
                                             seed=seed)

        imp_df = pd.DataFrame({'col':df_dev[varlist].columns,
                  'imp':model.feature_importances_})
        imp_df_sort = imp_df.sort_values('imp',ascending=False)
        imp_df_sort['cum_imp'] = imp_df_sort['imp'].cumsum()
        imp_df.to_csv(f"iter_randsearch/var/round_{random_integer}_{len(imp_df)}.csv", index=False)
        print(f'round_{len(imp_df)}:', perf)
        print(f'round_{len(imp_df)}:', param)
        var_next = imp_df_sort[(imp_df_sort['cum_imp']<=keep_importance)&(imp_df_sort['imp']>0)]['col'].to_list()

        model.save_model(f"iter_randsearch/model/model_{random_integer}_{len(imp_df)}.json")
        gc.collect()
        if generate_report:
            X_train, X_test, y_train, y_test = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=seed)
            df_report = df_dev.copy()
            df_report.loc[X_train.index,'seg'] = 'dev'
            df_report.loc[X_test.index,'seg'] = 'val'
            df_oot_all = pd.concat([df.assign(seg=key) for key, df in df_oot.items()], ignore_index=True)
            df_report = pd.concat([df_report.reset_index(drop=True), df_oot_all.reset_index(drop=True)],axis=0)
            df_report['seg'] = df_report['seg'].fillna('oot')
            df_report[['seg', 't0_cnt', 't3_cnt', 't7_cnt']+varlist].to_csv("./iter_randsearch/_tmp_output_variable.csv")
            mdlr =model_reporter("./iter_randsearch/_tmp_output_variable.csv",['seg'],['dev'],dep,f"iter_randsearch/model/model_{random_integer}_{len(imp_df)}.json",{},f'iter_randsearch/report/model_{random_integer}_{len(imp_df)}.xlsx', scring=True, scr_logbase=log_score)
            mdlr.run()

        varlist = var_next
    return

def single_random_search(df_dev,
                        df_oot,
                        dep,
                        param_space,
                        rounds,
                        ntrials,
                        varlist,
                        weight_col=None,
                        test_size= 0.33,
                        seed=None,
                        generate_report=False,
                        unique_id=['loan_id']):

    os.makedirs('single_rd_search', exist_ok=True)
    os.makedirs('single_rd_search/model', exist_ok=True)
    os.makedirs('single_rd_search/report', exist_ok=True)
    log_score = 1/(df_dev[dep].sum()/len(df_dev)) -1

    res = []
    for i in range(rounds):
        all_params = {'round':i}
        random_integer = random.randint(10000, 20000)
        if not seed:
            all_params['seed'] = random_integer
        else:
            all_params['seed'] = seed
        model, param, perf = hyperopt_search(df_dev=df_dev,
                                             df_oot=df_oot,
                                             dep=dep,
                                             param_space=param_space,
                                             weight_col=weight_col,
                                             ntrials=ntrials,
                                             varlist=varlist,
                                             test_size= test_size,
                                             seed=all_params['seed'])

        perf_flat = {f"{k}_auc": v[0] for k, v in perf.items()}
        perf_flat.update({f"{k}_ks": v[1] for k, v in perf.items()})

        imp_df = pd.DataFrame({'col':df_dev[varlist].columns,
                  'imp':model.feature_importances_})
        imp_df_sort = imp_df.sort_values('imp',ascending=False)
        all_params.update(perf_flat)
        all_params.update(param)
        all_params['param_conc'] = param
        all_params['var_sort'] = imp_df_sort['col'].to_list()
        print(f'round_{i}:', perf)
        print(f'round_{i}:', param)

        model.save_model(f"single_rd_search/model/model_{i}.json")
        gc.collect()
        if generate_report:
            X_train, X_test, _, _ = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=all_params['seed'])
            df_report = df_dev.copy()
            df_report.loc[X_train.index,'seg'] = 'dev'
            df_report.loc[X_test.index,'seg'] = 'val'
            df_oot_all = pd.concat([df.assign(seg=key) for key, df in df_oot.items()], ignore_index=True)
            df_report = pd.concat([df_report.reset_index(drop=True), df_oot_all.reset_index(drop=True)],axis=0)
            df_report['seg'] = df_report['seg'].fillna('oot')
            df_report[unique_id+['seg', 't0_cnt', 't3_cnt', 't7_cnt']+varlist].to_csv("./single_rd_search/report/_tmp_output_variable.csv",index=False)
            mdlr =model_reporter("./single_rd_search/report/_tmp_output_variable.csv",['seg'],['dev'],dep,f"single_rd_search/model/model_{i}.json",{},f'single_rd_search/report/model_{i}.xlsx', scring=True, scr_logbase=log_score)
            mdlr.run()
        res.append(all_params)

    out = pd.DataFrame(res)
    out.to_csv('./single_rd_search/rd_search_report.csv',index=False)
    return out