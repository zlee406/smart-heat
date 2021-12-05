import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import os, sys, pathlib, importlib, pickle
import sklearn.ensemble
import sklearn.linear_model
import xgboost as xgb
sys.path.append(f'{pathlib.Path(os.path.abspath("")).parents[0]}')
import read, cop, process, MLPrograms
importlib.reload(read)
importlib.reload(cop)
importlib.reload(process)

import wandb

wandb.init(project="DYD Heating Demand Prediction")
config = wandb.config
# Import Data

location = 'NY'
local = False
DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[1]}/Data Files'
# df_list_hp = pickle.load(open(f'{DATA_DIR}/DF Lists/df_list_{location}.sav', 'rb'))
# df_list_gas = pickle.load(open(f'{DATA_DIR}/DF Lists/df_list_gas_{location}.sav', 'rb'))
if not local:
    grouped_df_hp = pickle.load(open(f'{DATA_DIR}/DF Lists/grouped_df_hp_{location}.sav', 'rb'))
    grouped_df_gas = pickle.load(open(f'{DATA_DIR}/DF Lists/grouped_df_gas_{location}.sav', 'rb'))
else:
    grouped_df_hp  = pickle.load(open(f'{DATA_DIR}/DF Lists/grouped_loc_df_hp_{location}.sav', 'rb'))
    grouped_df_gas  = pickle.load(open(f'{DATA_DIR}/DF Lists/grouped_loc_df_gas_{location}.sav', 'rb'))

# Resample and Generate Features
DT = '15T'
grouped_df_gas_opt = grouped_df_gas.loc[:, ['T_ctrl_C', 'GHI_(kW/m2)',  'Wind Speed', 'T_out_C',
                                            'T_stp_heat', 'effectiveGasHeat']]
grouped_df_gas_opt = grouped_df_gas_opt.interpolate(method='linear', limit=3)

if local:
    level_values = grouped_df_gas_opt.index.get_level_values
    grouped_df_gas_opt = grouped_df_gas_opt.groupby([level_values(i) for i in [1]]
                                                    +[pd.Grouper(freq=DT, level=-2)]).mean()
else:
    grouped_df_gas_opt = grouped_df_gas_opt.groupby(pd.Grouper(freq=DT)).mean()

grouped_df_gas_opt['T_ctrl_C_t1'] = grouped_df_gas_opt['T_ctrl_C'].shift(-1)
grouped_df_gas_opt['MOD'] = (grouped_df_gas_opt.index.get_level_values('DateTime').minute
                             + 60* grouped_df_gas_opt.index.get_level_values('DateTime').hour)
dr = grouped_df_gas_opt.index.get_level_values('DateTime')
grouped_df_gas_opt['Holiday'] = dr.isin(calendar().holidays(start=dr.min(), end=dr.max()))

# Get Supervised DF
n_out = 1
n_in = config.n_in

grouped_df_gas_opt_sup, n_vars = MLPrograms.series_to_supervised(grouped_df_gas_opt,
                                                                 n_in=n_in,
                                                                 n_out=n_out,
                                                                 )

labelcols = ['effectiveGasHeat(t+' + str(i) + ')' for i in range(0, n_out)]
dropcols = []

for i in range(0, n_out):
    dropcols.append('effectiveGasHeat(t+' + str(i) + ')')
for i in range(-n_in, 0):
    dropcols.append('effectiveGasHeat(t' + str(i) + ')')
    dropcols.append('Holiday(t' + str(i) + ')')
    dropcols.append('MOD(t' + str(i) + ')')

labels_df = grouped_df_gas_opt_sup[labelcols]
features_df = grouped_df_gas_opt_sup.drop(columns=dropcols, errors='ignore')
labels = labels_df.values.ravel()
features = features_df.values

# Split Into K Folds
n_splits = 5
kf = sklearn.model_selection.KFold(n_splits=n_splits)
kf.get_n_splits(features)
test_array = np.zeros(n_splits)
train_array = np.zeros(n_splits)

model_type = config.model_type


# Get Params
np.random.seed(config.seed)
max_depth = np.random.randint(1, 10)
eta = np.random.choice([.1, .2, .3])
min_child_weight = np.random.randint(1, 10)
subsample = np.random.uniform(.5, 1)
early_stopping_rounds = np.random.randint(10, 50)

xgb_params = {
    'tree_method': 'gpu_hist',
    'max_depth': max_depth,
    'eta': eta,
    'objective': 'reg:squarederror',
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    # 'colsample_bytree': config.colsample_bytree,
    'eval_metric': 'rmse',
}
ridge_alpha = np.random.choice([.1, 1, 10, 100])
lasso_alpha = np.random.choice([.01, .1, .25, .5])
trees = np.random.randint(1, 300)
model_type='XG'
# Train
for n, (train_index, test_index) in enumerate(kf.split(features)):
    # if model_type == 'linear':
    #     model = sklearn.linear_model.LinearRegression()
    #     result = model.fit(features[train_index], labels[train_index])
    #
    # if model_type == 'poly':
    #     model = sklearn.linear_model.LinearRegression()
    #     result = model.fit(features[train_index], labels[train_index])

    if model_type == 'ridge':
        model = sklearn.linear_model.Ridge(alpha=ridge_alpha)
        result = model.fit(features[train_index], labels[train_index])
        wandb.log({f'Ridge Alpha': ridge_alpha})

        # Test Model
        train_preds = model.predict(features[train_index])
        test_preds = model.predict(features[test_index])

    if model_type == 'lasso':
        model = sklearn.linear_model.Lasso(alpha=lasso_alpha, max_iter=1000000)
        result = model.fit(features[train_index], labels[train_index])
        wandb.log({f'Lasso Alpha': lasso_alpha})

        # Test Model
        train_preds = model.predict(features[train_index])
        test_preds = model.predict(features[test_index])

    if model_type == 'RF':
        model = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators=trees)
        model.fit(features[train_index], labels[train_index])
        wandb.log({'Trees': trees})

        # Test Model
        train_preds = model.predict(features[train_index])
        test_preds = model.predict(features[test_index])

    if model_type == 'XG':

        dtrain = xgb.DMatrix(features[train_index], label=labels[train_index])
        dtest = xgb.DMatrix(features[test_index], label=labels[test_index])
        model = xgb.train(xgb_params, dtrain,
                          num_boost_round=9999,
                          evals=[(dtest, 'Test')],
                          early_stopping_rounds=early_stopping_rounds,
                          )

        dxtrain = xgb.DMatrix(features[train_index])
        dxtest = xgb.DMatrix(features[test_index])
        train_preds = model.predict(dxtrain)
        test_preds = model.predict(dxtest)

        wandb.log({
            'max_depth': max_depth,
            'eta': eta,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            # 'colsample_bytree': config.colsample_bytree,
            'eval_metric': 'rmse',
        })

    # Get Error and Print
    test_array[n] = sklearn.metrics.mean_squared_error(test_preds, labels[test_index])
    train_array[n] = sklearn.metrics.mean_squared_error(train_preds, labels[train_index])

    print(f'{model_type.capitalize()} Val error {n}: {test_array[n]}')
    print(f'{model_type.capitalize()} Train error {n}: {train_array[n]}')

    wandb.log({f'Val MSE {n}': test_array[n],
               f'Train MSE  {n}': train_array[n],
               })
wandb.log({f'Ave Val MSE': np.mean(test_array),
           f'Ave Train MSE': np.mean(train_array),
           f'Model Type': model_type})