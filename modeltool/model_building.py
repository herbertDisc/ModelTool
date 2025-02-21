from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import hyperopt
import optuna
import pandas as pd
import os, random
import gc
from .model_report import model_reporter
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
                       oot_seg=None,
                       test_size= 0.33,
                       seed=123,
                       verbose=False,
                       generate_report=False):
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
    X_train, X_test, y_train, y_test = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=seed)

    # For some defualt value if not set in param
    default_param = {'importance_type': "total_gain",
                     'eval_metric': 'auc'}
    for key, value in default_param.items():
        param.setdefault(key, value)

    xgb_model = XGBClassifier(**param)
    xgb_model = xgb_model.fit(X_train, y_train,
                                eval_set=[(X_train[varlist], y_train),
                                          (X_test[varlist], y_test),
                                            ],
                                verbose=verbose)

    train_pred = xgb_model.predict_proba(X_train[varlist])[:,1]
    test_pred = xgb_model.predict_proba(X_test[varlist])[:,1]
    df_oot.loc[:,'pred'] = xgb_model.predict_proba(df_oot[varlist])[:,1]
    if oot_seg is not None:
        perf = df_oot.groupby(oot_seg).apply(lambda x:auc_ks(x[dep],x['pred'])).to_dict()
    else:
        perf = {'oot':auc_ks(df_oot[dep],df_oot['pred'])}
    perf['dev'] = auc_ks(y_train,train_pred)
    perf['test'] = auc_ks(y_test,test_pred)
    return xgb_model, perf

def hyperopt_search(df_dev,
                    df_oot,
                    dep,
                    param_space,
                    ntrials,
                    varlist=None,
                    oot_seg=None,
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
                                    param=model_param,
                                    varlist=varlist,
                                    oot_seg=oot_seg,
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
                                    varlist=varlist,
                                    oot_seg=oot_seg,
                                    test_size= test_size,
                                    seed=seed)
    
    return model, best, perf

def iter_random_search(df_dev,
                        df_oot,
                        dep,
                        param_space,
                        ntrials,
                        varlist,
                        stop=30,
                        keep_importance = 1,
                        oot_seg=None,
                        test_size= 0.33,
                        seed=123,
                        generate_report=False):

    random_integer = random.randint(10000, 20000)
    os.makedirs('var', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs('report', exist_ok=True)
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
                                             oot_seg=oot_seg,
                                             test_size= test_size,
                                             seed=seed)

        imp_df = pd.DataFrame({'col':df_dev[varlist].columns,
                  'imp':model.feature_importances_})
        imp_df_sort = imp_df.sort_values('imp',ascending=False)
        imp_df_sort['cum_imp'] = imp_df_sort['imp'].cumsum()
        imp_df.to_csv(f"var/round_{random_integer}_{len(imp_df)}.csv", index=False)
        print(f'round_{len(imp_df)}:', perf)
        print(f'round_{len(imp_df)}:', param)
        var_next = imp_df_sort[(imp_df_sort['cum_imp']<=keep_importance)&(imp_df_sort['imp']>0)]['col'].to_list()

        model.save_model(f"model/model_{random_integer}_{len(imp_df)}.json")
        gc.collect()
        if generate_report:
            X_train, X_test, y_train, y_test = train_test_split(df_dev[varlist], df_dev[dep], test_size=test_size, random_state=seed)
            df_report = df_dev.copy()
            df_report.loc[X_train.index,'seg'] = 'dev'
            df_report.loc[X_test.index,'seg'] = 'val'
            df_report = pd.concat([df_report.reset_index(drop=True), df_oot.reset_index(drop=True)],axis=0)
            df_report['seg'] = df_report['seg'].fillna('oot')
            df_report[['seg','t0_cnt']+varlist].to_csv("./report/_tmp_output_variable.csv")
            mdlr =model_reporter("./report/_tmp_output_variable.csv",['seg'],['dev'],dep,f"model/model_{random_integer}_{len(imp_df)}.json",{},f'report/model_{random_integer}_{len(imp_df)}.xlsx', scring=True, scr_logbase=log_score)
            mdlr.run()

        varlist = var_next
    return

# def objective(trial, df_dev, df_oot, dep, varlist,oot_seg,test_size,seed):
#     """
#     Objective function that Optuna will optimize.
#     This function defines the hyperparameter search space and trains/evaluates the XGBoost model.
#     """
    
#     # Suggest hyperparameters for XGBoost
#     param = {
#         'max_depth': trial.suggest_int('max_depth', 3, 5),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log= True),
#         'n_estimators': trial.suggest_int('n_estimators', 50, 80),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 0.8),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 10, 100),
#         'reg_lambda': trial.suggest_float('reg_lambda', 10, 100),
#         'eval_metric': 'auc',
#         'scale_pos_weight': 0.2,
#     }

#     # Train the XGBoost model
#     _,perf = train_single_model(df_dev=df_dev,
#                                 df_oot=df_oot,
#                                 dep=dep,
#                                 param=param,
#                                 varlist=varlist,
#                                 oot_seg=oot_seg,
#                                 test_size= test_size,
#                                 seed=seed)
    
#     # Evaluate accuracy
#     accuracy = 1 - perf['test'][0]
    
#     # Return the accuracy as the objective to maximize
#     return accuracy

# def optimize_hyperparameters(df_dev, df_oot, dep, varlist=None,oot_seg=None,test_size=0.33,seed=123,ntrials=60):
#     """Use Optuna to optimize hyperparameters for XGBoost."""
#     # Load the data

#     # Define the Optuna study (for maximization of accuracy)
#     study = optuna.create_study(direction='maximize')
    
#     # Optimize using the objective function
#     study.optimize(lambda trial: objective(trial, df_dev, df_oot, dep, varlist,oot_seg,test_size, seed), n_trials=ntrials)
    
#     # Print the best trial and hyperparameters
#     print(f'Best trial: {study.best_trial.number}')
#     print(f'Best accuracy: {study.best_trial.value}')
#     print(f'Best hyperparameters: {study.best_trial.params}')
    
#     return study.best_trial