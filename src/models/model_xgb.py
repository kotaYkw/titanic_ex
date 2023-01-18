import os

import numpy as np
import pandas as pd
import xgboost as xgb

from models.model import Model
from models.util import Util

class ModelXGB(Model):
    """xgboostモデルクラス"""

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        validation = va_x is not None
        dtrain = xgb.DMatrix(tr_x, label=tr_y, enable_categorical=True)
        if validation:
            dvalid = xgb.DMatrix(va_x, label=va_y, enable_categorical=True)
        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        # 学習
        if validation:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            watch_list = [(dtrain, 'train'), (dvalid, 'eval')]
            self.model = xgb.train(params, dtrain, num_round, evals=watch_list,
                                   early_stopping_rounds=early_stopping_rounds)
        else:
            watch_list = [(dtrain, 'train')]
            self.model = xgb.train(params, dtrain, num_round, evals=watch_list)

    def predict(self, te_x):
        d_test = xgb.DMatrix(te_x, enable_categorical=True)
        return self.model.predict(d_test, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)

