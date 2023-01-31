import numpy as np
import pandas as pd

from models.model_xgb import ModelXGB
from models.runner_2 import Runner
from models.util import Submission

# 数値の特徴量をカテゴリ化した特徴量を追加した推論結果を求める。
if __name__ == '__main__':

    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 350

    # 特徴量の指定
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # 目的変数のカラム名の指定
    target_name = 'Survived'

    # xgboostによる学習・予測
    runner = Runner('xgb2', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('xgb2', target_name)

    '''
    # (参考）xgboostによる学習・予測 - 学習データ全体を使う場合
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    '''