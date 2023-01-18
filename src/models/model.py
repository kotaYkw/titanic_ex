from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

class Model(metaclass=ABCMeta):
    """モデルのメソッドを持つ抽象クラス

    Attributes:
        run_fold_name (str): ランの名前とfoldの番号を組み合わせた名前
        prms (dict): ハイパーパラメータ
    """

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """コンストラクタ

        Args:
            run_fold_name (str): ランの名前とfoldの番号を組み合わせた名前
            prms (dict): ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.DataFrame,
              va_x: pd.DataFrame, va_y: pd.DataFrame) -> None:
        """モデルの学習を行い、学習済みのモデルを保存する。

        Args:
            tr_x (pd.DataFrame): 学習データの特徴量
            tr_y (pd.DataFrame): 学習データの目的変数
            va_x (pd.DataFrame): 検証データの特徴量
            va_y (pd.DataFrame): 検証データの目的変数
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """学習済みのモデルでの予測値を返す。

        Args:
            te_x (pd.DataFrame): 検証データやテストデータの特徴量

        Returns:
            np.array: 予測値
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """モデルの保存を行う
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """モデルの読み込みを行う
        """
        pass
