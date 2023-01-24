import numpy as np
import pandas as pd
from models.model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import Callable, List, Optional, Tuple, Union

from models.util import Logger, Util

logger = Logger()


class Runner:
    """クロスバリデーションなども含めた学習・評価・予測を行うクラス

    Attributes:
        run_name (str): ランの名前
        model_cls (Callable[[str, dict]): モデルのクラス
        features (List[str]) : 特徴量のリスト
        params (dict): ハイパーパラメータ
    """

    def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict) -> None:
        """コンストラクタ

        Args
            run_name (str): ランの名前
            model_cls (Callable[[str, dict]): モデルのクラス
            features (List[str]) : 特徴量のリスト
            params (dict): ハイパーパラメータ
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features
        self.params = params
        self.n_fold = 4

    def train_fold(self, i_fold: Union[int, str]) -> \
            Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う
        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        Args:
            i_fold (Union[int, str]): foldの番号（すべてのときには'all'とする）

        Returns:
            Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
                （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        validation = i_fold != 'all'
        train_x = self.load_x_train()
        train_x.info()
        train_y = self.load_y_train()

        if validation:
            # 学習データ・検証データをセットする
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
            # 学習を行う
            model = self.build_model(i_fold)
            model.train(tr_x, tr_y, va_x, va_y)
            # 検証データへの予測・評価を行う
            va_pred = model.predict(va_x)
            score = log_loss(va_y, va_pred, eps=1e-15, normalize=True)
            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_pred, score
        else:
            # 学習データ全てで評価を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y)
            # モデルを返す
            return model, None, None, None

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う
        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        va_idxes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            scores.append(score)
            preds.append(va_pred)

        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        Util.dump(preds, f'model/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う
        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f'model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""
        logger.info(f'{self.run_name} - start training all')

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う
        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f'model/pred/{self.run_name}-test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う
        
        Args:
            i_fold (Union[int, str]): foldの番号

        Returns:
            Model: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_cls(run_fold_name, self.params)

    def preprocess(self, train_df) -> pd.DataFrame:
        """データの前処理を行う
        
        Args:
            train_df (pd.DataFrame): 前処理を行うデータフレーム

        Returns:
            pd.DataFrame: 前処理済みのデータフレーム
        """
        train_df['Sex'] = train_df['Sex'].astype("category")
        train_df['Pclass'] = train_df['Pclass'].astype("category")
        train_df['Age'] = train_df['Age'].fillna(-1)
        train_df['Age'] = train_df['Age'].astype("int")
        train_df['SibSp'] = train_df['SibSp'].astype("int")
        train_df['Parch'] = train_df['Parch'].astype("int")
        train_df['Embarked'] = train_df['Embarked'].astype("category")
        return train_df

    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む
        
        Returns:
            pd.DataFrame: 学習データの特徴量
        """
        # 学習データの読込を行う
        # 列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        # 毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）
        df = pd.read_csv('input/train.csv')[self.features]
        df = self.preprocess(df)
        return df

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む

        Returns:
            pd.Series: 学習データの目的変数
        """
        # 目的変数の読込を行う
        train_y = pd.read_csv('input/train.csv')['Survived']
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む

        Returns:
            pd.DataFrame: テストデータの特徴量
        """
        df = pd.read_csv('input/test.csv')[self.features]
        df = self.preprocess(df)
        return df

    def load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        Args:
            i_fold (int): foldの番号

        Returns:
            np.array: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]