from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import hyperopt
import optuna
import pandas as pd
import os, random
import gc
from .model_report import model_reporter
from datetime import datetime
import pickle

def auc_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    mdl_auc = auc(fpr, tpr)
    mdl_ks = max(tpr - fpr)
    return [round(mdl_auc,3), round(mdl_ks,3)]

def train_single_model(df_dev,
                       df_oot,
                       dep,
                       param,
                       varlist=None,
                       weight_col = None,
                       test_size= 0.33,
                       seed=123,
                       generate_report=False,
                       model_type = 'xgb',
                       additional_target=['loan_id']):
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
    if varlist is None:
        varlist = [x for x in df_dev.columns if x != dep]

    if weight_col is None:
        df_dev['_tmp_weight'] = 1
        weight_col = '_tmp_weight'

    X_train, X_valid, y_train, y_valid, w_train, _ = train_test_split(
        df_dev[varlist],
        df_dev[dep],
        df_dev[weight_col],
        test_size=test_size,
        random_state=seed
        )

    if model_type == 'xgb':
        default_param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
        D_train = xgb.DMatrix(X_train, y_train, weight=w_train)
        D_valid = xgb.DMatrix(X_valid, y_valid)
    else:
        default_param = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1}
        D_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
        D_valid = lgb.Dataset(X_valid, label=y_valid)
    for key, value in default_param.items():
        param.setdefault(key, value)


    early_stopping_rounds = param.pop('early_stopping_rounds', 8)
    callbacks = []
    if model_type == 'xgb':
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds)
    else:
        early_stop = lgb.early_stopping(early_stopping_rounds, first_metric_only=True)
    callbacks.append(early_stop)

    n_estimators = param.pop('n_estimators', 100)
    ### train the model
    if model_type == 'xgb':
        model = xgb.train(
            params=param,
            dtrain=D_train,
            evals=[(D_train, 'Train'),
                        (D_valid, 'Valid'),
                        ],
            num_boost_round=n_estimators,
            verbose_eval=False,
            callbacks=callbacks)
        ### evaluation
        train_pred = model.predict(D_train)
        test_pred = model.predict(D_valid)
    else:
        model = lgb.train(
            params=param,
            train_set=D_train,
            valid_sets=[D_train, D_valid
                  ],
            num_boost_round=n_estimators,
            callbacks=callbacks)
        ### evaluation
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_valid)


    perf = {}
    for key,datain in df_oot.items():
        datain = datain.copy()
        if model_type == 'xgb':
            D_oot = xgb.DMatrix(datain[varlist])
        else:
            D_oot = datain[varlist]
        datain['pred'] = model.predict(D_oot)

        perf[key] = auc_ks(datain[dep], datain['pred'])
    perf['dev'] = auc_ks(y_train, train_pred)
    perf['test'] = auc_ks(y_valid, test_pred)

    if generate_report and len(varlist)< 200:
        os.makedirs('tmp_single_model', exist_ok=True)
        nowtime = datetime.now().strftime("%y%m%d%H%M%S")
        df_report = df_dev.copy()
        df_report.loc[X_train.index,'seg'] = 'dev'
        df_report.loc[X_valid.index,'seg'] = 'val'
        df_oot_all = pd.concat([df.assign(seg=key) for key, df in df_oot.items()], ignore_index=True)
        df_report = pd.concat([df_report.reset_index(drop=True), df_oot_all.reset_index(drop=True)],axis=0)
        df_report['seg'] = df_report['seg'].fillna('oot')
        df_report[['seg', dep]+additional_target+varlist].to_csv("./tmp_single_model/_tmp_output_variable.csv")
        model.save_model(f"./tmp_single_model/_tmp_model_{nowtime}.json")
        log_score = 1/(df_dev[dep].sum()/len(df_dev)) -1
        mdlr =model_reporter(csvfile="./tmp_single_model/_tmp_output_variable.csv",
                             segs=['seg'],
                             bmk_seg=['dev'],
                             dep=dep,
                             model_path=f"./tmp_single_model/_tmp_model_{nowtime}.json",
                             model_config={},
                             outputpath=f'./tmp_single_model/model_{nowtime}.xlsx',
                             scring=True,
                             scr_logbase=log_score)
        mdlr.run()
    return model, perf

def hyperopt_search(df_dev,
                    df_oot,
                    dep,
                    param_space,
                    ntrials,
                    model_type='xgb',
                    weight_col=None,
                    varlist=None,
                    test_size= 0.33,
                    seed=123):

    def hyperopt_objective(params):

        model_param = {}
        if 'max_depth' in params:
            model_param['max_depth'] = int(float(params['max_depth']))
        if 'learning_rate' in params:
            model_param['learning_rate'] = params['learning_rate']
        if 'n_estimators' in params:
            model_param['n_estimators'] = int(float(params['n_estimators']))
        if 'min_child_weight' in params:
            model_param['min_child_weight'] = int(float(params['min_child_weight']))
        if 'subsample' in params:
            model_param['subsample'] = params['subsample']
        if 'colsample_bytree' in params:
            model_param['colsample_bytree'] = params['colsample_bytree']
        if 'gamma' in params:
            model_param['gamma'] = params['gamma']
        if 'min_split_gain' in params:
            model_param['min_split_gain'] = params['min_split_gain']
        if 'reg_alpha' in params:
            model_param['reg_alpha'] = params['reg_alpha']
        if 'reg_lambda' in params:
            model_param['reg_lambda'] = params['reg_lambda']
        if 'scale_pos_weight' in params:
            model_param['scale_pos_weight'] = params['scale_pos_weight']
        if 'num_leaves' in params:
            model_param['num_leaves'] = int(float(params['num_leaves']))
        if 'min_data_in_leaf' in params:
            model_param['min_data_in_leaf'] = int(float(params['min_data_in_leaf']))
        if 'min_child_samples' in params:
            model_param['min_child_samples'] = int(float(params['min_child_samples']))


        model,perf = train_single_model(df_dev=df_dev,
                                    df_oot=df_oot,
                                    dep=dep,
                                    model_type=model_type,
                                    weight_col=weight_col,
                                    param=model_param,
                                    varlist=varlist,
                                    test_size= test_size,
                                    seed=seed)

        return {
            'loss': 1 - perf['test'][0],
            'status': hyperopt.STATUS_OK,
            'model': model,
            'perf': perf,
        }

    trials = hyperopt.Trials()
    best = hyperopt.fmin(hyperopt_objective,
                         space=param_space,
                         algo=hyperopt.tpe.suggest,
                         max_evals=ntrials,
                         trials=trials)

    for key in [
        'max_depth',
        'min_child_weight',
        'n_estimators',
        'num_leaves',
        'min_data_in_leaf',
        'min_child_samples',
    ]:
        if key in best:
            best[key] = int(best[key])

    best_model = trials.best_trial['result']['model']
    best_perf = trials.best_trial['result']['perf']
    
    return best_model, best, best_perf

def iter_random_search(df_dev,
                       df_oot,
                       dep,
                       param_space,
                       ntrials,
                       varlist,
                       weight_col=None,
                       model_type='xgb',
                       stop=30,
                       keep_importance = 1,
                       test_size= 0.33,
                       seed=123,
                       generate_report=False,
                       suffix='',
                       additional_target=['loan_id']):
    """
    run a iterative random search for variable reduction
    
    """
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    base_dir = f"iter_randsearch{('_' + suffix) if suffix else ''}"
    os.makedirs(os.path.join(base_dir, 'var'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'report'), exist_ok=True)
    log_score = 1/(df_dev[dep].sum()/len(df_dev)) -1
    var_round_len = len(varlist) - 1
    while len(varlist) > stop and var_round_len != len(varlist):
        var_round_len = len(varlist)

        model, param, perf = hyperopt_search(df_dev=df_dev,
                                             df_oot=df_oot,
                                             dep=dep,
                                             model_type=model_type,
                                             param_space=param_space,
                                             ntrials=ntrials,
                                             varlist=varlist,
                                             weight_col=weight_col,
                                             test_size= test_size,
                                             seed=seed)

        if model_type == 'xgb':
            imp = model.get_score(importance_type="gain")
            imp_df = (
                    pd.DataFrame(list(imp.items()), columns=["col", "imp"])
                    .sort_values("imp", ascending=False)
                    .reset_index(drop=True)
                )
        else:
            imp_df = pd.DataFrame({"col": model.feature_name(),
                                   "imp": model.feature_importance(importance_type='gain')
                                }).sort_values("imp", ascending=False)
        imp_df["imp"] = imp_df["imp"] / imp_df["imp"].sum()
        imp_df["cum_imp"] = imp_df["imp"].cumsum()
        imp_df.to_csv(os.path.join(base_dir, 'var', f"round_{nowtime}_{len(imp_df)}.csv"), index=False)

        print(f'round_{len(imp_df)}:', perf)
        print(f'round_{len(imp_df)}:', param)
        var_next = imp_df[(imp_df['cum_imp']<=keep_importance)&(imp_df['imp']>0)]['col'].to_list()

        if model_type == 'xgb':
            model_path = os.path.join(base_dir, 'model', f"model_{nowtime}_{len(imp_df)}.json")
        else:
            model_path = os.path.join(base_dir, 'model', f"model_{nowtime}_{len(imp_df)}.txt")
        model.save_model(model_path)
        gc.collect()

        if generate_report and len(varlist)< 200:
            X_train, X_test, y_train, y_test = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=seed)
            df_report = df_dev.copy()
            df_report.loc[X_train.index,'seg'] = 'dev'
            df_report.loc[X_test.index,'seg'] = 'val'
            df_oot_all = pd.concat([df.assign(seg=key) for key, df in df_oot.items()], ignore_index=True)
            df_report = pd.concat([df_report.reset_index(drop=True), df_oot_all.reset_index(drop=True)],axis=0)
            df_report['seg'] = df_report['seg'].fillna('oot')
            tmp_output = os.path.join(base_dir, '_tmp_output_variable.csv')
            df_report[['seg', dep]+additional_target+varlist].to_csv(tmp_output)
            report_path = os.path.join(base_dir, 'report', f"model_{nowtime}_{len(imp_df)}.xlsx")
            mdlr =model_reporter(csvfile=tmp_output,
                                 segs=['seg'],
                                 bmk_seg=['dev'],
                                 dep=dep,
                                 model_type=model_type,
                                 model_path=model_path,
                                 model_config={},
                                 outputpath=report_path,
                                 scring=True,
                                 scr_logbase=log_score)
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
                         model_type='xgb',
                         weight_col=None,
                         test_size= 0.33,
                         seed=None,
                         generate_report=False,
                         suffix='',
                         additional_target=['loan_id']):
    """
    run a random search on the hyperparameter space

    """
    base_dir = f"single_rd_search{('_' + suffix) if suffix else ''}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'report'), exist_ok=True)
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
                                             model_type=model_type,
                                             param_space=param_space,
                                             weight_col=weight_col,
                                             ntrials=ntrials,
                                             varlist=varlist,
                                             test_size= test_size,
                                             seed=all_params['seed'])

        perf_flat = {f"{k}_auc": v[0] for k, v in perf.items()}
        perf_flat.update({f"{k}_ks": v[1] for k, v in perf.items()})

        # imp_df = pd.DataFrame({'col':df_dev[varlist].columns,
        #           'imp':model.get_score(importance_type="total_gain")})
        # imp_df_sort = imp_df.sort_values('imp',ascending=False)
        if model_type == 'xgb':
            imp = model.get_score(importance_type="gain")
            imp_df = (
                    pd.DataFrame(list(imp.items()), columns=["col", "imp"])
                    .sort_values("imp", ascending=False)
                    .reset_index(drop=True)
                )
        else:
            imp_df = pd.DataFrame({"col": model.feature_name(),
                                   "imp": model.feature_importance(importance_type='gain')
                                }).sort_values("imp", ascending=False)
        imp_df["imp"] = imp_df["imp"] / imp_df["imp"].sum()
        imp_df["cum_imp"] = imp_df["imp"].cumsum()
        all_params.update(perf_flat)
        all_params.update(param)
        all_params['param_conc'] = param
        all_params['var_sort'] = imp_df['col'].to_list()
        print(f'round_{i}:', perf)
        print(f'round_{i}:', param)

        model.save_model(f"single_rd_search/model/model_{i}.json")
        gc.collect()
        if generate_report and len(varlist)< 200:
            X_train, X_test, _, _ = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=all_params['seed'])
            df_report = df_dev.copy()
            df_report.loc[X_train.index,'seg'] = 'dev'
            df_report.loc[X_test.index,'seg'] = 'val'
            df_oot_all = pd.concat([df.assign(seg=key) for key, df in df_oot.items()], ignore_index=True)
            df_report = pd.concat([df_report.reset_index(drop=True), df_oot_all.reset_index(drop=True)],axis=0)
            df_report['seg'] = df_report['seg'].fillna('oot')
            tmp_output = os.path.join(base_dir, 'report', '_tmp_output_variable.csv')
            df_report[['seg', dep]+additional_target+varlist].to_csv(tmp_output,index=False)
            report_path = os.path.join(base_dir, 'report', f"model_{i}.xlsx")
            mdlr =model_reporter(csvfile=tmp_output,
                                 segs=['seg'],
                                 bmk_seg=['dev'],
                                 dep=dep,
                                 model_type=model_type,
                                 model_path=f"single_rd_search/model/model_{i}.json",
                                 model_config={},
                                 outputpath=report_path,
                                 scring=True,
                                 scr_logbase=log_score)
            mdlr.run()
        res.append(all_params)

    out = pd.DataFrame(res)
    out.to_csv(os.path.join(base_dir, 'rd_search_report.csv'),index=False)
    return out