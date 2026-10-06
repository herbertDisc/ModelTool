from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import hyperopt
import pandas as pd
import os, random
import gc
from .model_report import model_reporter
from .model_card_html import render_model_card
from datetime import datetime
import numpy as np

def write_html_report(report_path, run):
    '''
    render the model_reporter workbook as the HTML model card next to it and drop the workbook
    '''
    html_path = render_model_card(report_path,
                                  project=os.path.basename(os.getcwd()),
                                  run=run)
    os.remove(report_path)
    return html_path

def auc_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    mdl_auc = auc(fpr, tpr)
    mdl_ks = max(tpr - fpr)

    return {'auc': round(mdl_auc,3), 'ks': round(mdl_ks,3)}

def reg_metrics(y_true, y_pred):
    '''
    Default eval_func for model_type='xgb_linear' (continuous dep, e.g. net cash amount).
    auc_ks assumes a 0/1 target, which doesn't apply here. Spearman is computed via
    rank + corrcoef (not pandas' method='spearman') to avoid a scipy dependency, matching
    the approach already used elsewhere in this codebase.
    '''
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    mask = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    spearman = float(np.corrcoef(y_true.rank(), y_pred.rank())[0, 1]) if len(y_true) >= 2 else float('nan')

    return {'rmse': round(rmse, 3), 'mae': round(mae, 3), 'spearman': round(spearman, 3)}

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
                       additional_target=['loan_id'],
                       eval_func=None):
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
        df_dev.loc[:, '_tmp_weight'] = 1
        weight_col = '_tmp_weight'

    X_train, X_valid, y_train, y_valid, w_train, _ = train_test_split(
        df_dev[varlist],
        df_dev[dep],
        df_dev[weight_col],
        test_size=test_size,
        random_state=seed
        )

    is_xgb = model_type in ('xgb', 'xgb_linear')
    if model_type == 'xgb':
        default_param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        }
    elif model_type == 'xgb_linear':
        # gblinear: a linear (not tree-based) booster, fit via boosted coordinate descent —
        # regularized linear/logistic regression trained the xgboost way. Default objective
        # here is regression (reg:squarederror) since this is meant for continuous targets
        # like a net cash amount; pass objective='binary:logistic' in param to classify instead.
        default_param = {
            'objective': 'reg:squarederror',
            'booster': 'gblinear',
            'eval_metric': 'rmse',
        }
    else:
        default_param = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1}

    if is_xgb:
        D_train = xgb.DMatrix(X_train, y_train, weight=w_train)
        D_valid = xgb.DMatrix(X_valid, y_valid)
    else:
        D_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
        D_valid = lgb.Dataset(X_valid, label=y_valid)
    for key, value in default_param.items():
        param.setdefault(key, value)


    early_stopping_rounds = param.pop('early_stopping_rounds', 8)
    callbacks = []
    if is_xgb:
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds)
    else:
        early_stop = lgb.early_stopping(early_stopping_rounds, first_metric_only=True)
    callbacks.append(early_stop)

    # XGBoost requires callable objectives to be passed via obj=, not inside params.
    # Keep 'binary:logistic' in params so eval_metric='auc' and early stopping still work.
    custom_obj = None
    if is_xgb and callable(param.get('objective')):
        custom_obj = param.pop('objective')
        param['objective'] = 'binary:logistic'

    n_estimators = param.pop('n_estimators', 100)
    ### train the model
    if is_xgb:
        model = xgb.train(
            params=param,
            dtrain=D_train,
            evals=[(D_train, 'Train'),
                        (D_valid, 'Valid'),
                        ],
            num_boost_round=n_estimators,
            obj=custom_obj,
            verbose_eval=False,
            callbacks=callbacks)
        ### evaluation — custom obj outputs raw margin; output_margin keeps consistent scores
        train_pred = model.predict(D_train, output_margin=custom_obj is not None)
        test_pred = model.predict(D_valid, output_margin=custom_obj is not None)
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
    if eval_func is None:
        eval_func = reg_metrics if model_type == 'xgb_linear' else auc_ks
    for key,datain in df_oot.items():
        datain = datain.copy()
        if is_xgb:
            D_oot = xgb.DMatrix(datain[varlist])
        else:
            D_oot = datain[varlist]
        datain['pred'] = model.predict(D_oot, output_margin=custom_obj is not None)

        perf[key] = eval_func(datain[dep].to_numpy(), datain['pred'].to_numpy())
    perf['dev'] = eval_func(y_train.to_numpy(), train_pred)
    perf['test'] = eval_func(y_valid.to_numpy(), test_pred)

    if generate_report:
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
        write_html_report(f'./tmp_single_model/model_{nowtime}.xlsx', 'tmp_single_model')
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
                    seed=123,
                    eval_func=None,
                    extra_param=None):

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


        if extra_param:
            model_param.update(extra_param)

        model,perf = train_single_model(df_dev=df_dev,
                                    df_oot=df_oot,
                                    dep=dep,
                                    model_type=model_type,
                                    weight_col=weight_col,
                                    param=model_param,
                                    varlist=varlist,
                                    test_size= test_size,
                                    seed=seed,
                                    eval_func=eval_func)

        if 'top10lift' in perf['test']:
            metric_key, metric_val = 'top10lift', perf['test']['top10lift']
        else:
            metric_key, metric_val = next(iter(perf['test'].items()))
        # rmse/mae (used by reg_metrics, model_type='xgb_linear') are lower-is-better,
        # unlike auc/ks/top10lift — hyperopt always minimizes 'loss', so don't flip those.
        loss = metric_val if metric_key in ('rmse', 'mae') else 1 - metric_val

        return {
            'loss': loss,
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
                       additional_target=['loan_id'],
                       eval_func=None,
                       extra_param=None):
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
                                             seed=seed,
                                             eval_func=eval_func,
                                             extra_param=extra_param)

        if model_type in ('xgb', 'xgb_linear'):
            # gblinear only supports 'weight' (its coefficients) — 'gain'/'cover' are
            # tree-specific and raise on a linear booster.
            importance_type = 'weight' if model_type == 'xgb_linear' else 'gain'
            imp = model.get_score(importance_type=importance_type)
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

        if model_type in ('xgb', 'xgb_linear'):
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
            write_html_report(report_path, base_dir)

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
                         additional_target=['loan_id'],
                         eval_func=None,
                         extra_param=None,
                         report_max_vars=200):
    """
    run a random search on the hyperparameter space

    report_max_vars caps how many top-importance features the generated report's
    bivar/means/PSI sections render (model_reporter's max_report_vars) — it does not
    limit varlist itself, so a report is generated regardless of how large varlist is.
    Pass None to render every feature in varlist (slow / huge for wide varlists).
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
                                             seed=all_params['seed'],
                                             eval_func=eval_func,
                                             extra_param=extra_param)

        perf_flat = {}
        for k, v in perf.items():
            if isinstance(v, dict):
                for metric, val in v.items():
                    perf_flat[f"{k}_{metric}"] = val

        # imp_df = pd.DataFrame({'col':df_dev[varlist].columns,
        #           'imp':model.get_score(importance_type="total_gain")})
        # imp_df_sort = imp_df.sort_values('imp',ascending=False)
        if model_type in ('xgb', 'xgb_linear'):
            importance_type = 'weight' if model_type == 'xgb_linear' else 'gain'
            imp = model.get_score(importance_type=importance_type)
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

        if model_type in ('xgb', 'xgb_linear'):
            model_path = os.path.join(base_dir, 'model', f"model_{i}.json")
        else:
            model_path = os.path.join(base_dir, 'model', f"model_{i}.txt")
        model.save_model(model_path)
        gc.collect()
        if generate_report:
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
                                 model_path=model_path,
                                 model_config={},
                                 outputpath=report_path,
                                 scring=True,
                                 scr_logbase=log_score,
                                 max_report_vars=report_max_vars)
            mdlr.run()
            write_html_report(report_path, base_dir)
        res.append(all_params)

    out = pd.DataFrame(res)
    out.to_csv(os.path.join(base_dir, 'rd_search_report.csv'),index=False)
    return out