import datetime
import logging
import os

import numpy as np
import pandas as pd
import joblib
from xgboost import Booster


class Util:

    @classmethod
    def dump(cls, value: Booster, path: str) -> None:
        """モデルを特定のパスにpklファイルとしてダンプする

        Args:
            value (Booster): booster model
            path (str): 保存先のファイルパス
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path: str) -> Booster:
        """モデルをロードする

        Args:
            path (str): 保存先のファイルパス

        Returns:
            Booster: booster model
        """
        return joblib.load(path)


class Logger:

    def __init__(self, general_name='general', result_name='result'):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler('logs/'+ general_name + '.log')
        file_result_handler = logging.FileHandler('logs/' + result_name + '.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message: str) -> None:
        """時刻をつけてコンソールとログに出力

        Args:
            message (str): ログメッセージ
        """
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message: str) -> None:
        """コンソールとログに結果を出力

        Args:
            message (str): ログメッセージ
        """
        self.result_logger.info(message)

    def result_ltsv(self, dic: dict) -> None:
        """結果をLTSV形式で出力

        Args:
            dic (dict): ログと値の辞書
        """
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name: str, scores: list) -> None:
        """計算結果をコンソールと計算結果用ログに出力

        Args:
            run_name (str): ランの名前
            scores (list): スコアのリスト
        """
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self) -> str:
        """現在時刻を文字列で返す

        Returns:
            str: 現在時刻の文字列表現
        """
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic: dict) -> str:
        """Labeled Tab-separated Values (LTSV)、
           コロンで区切られたラベルと値の組み合わせ(key:value)を
           タブ区切りで表現したフォーマットに出力
        
        Returns:
            str: LTSV形式の文字列
        """
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:

    @classmethod
    def create_submission(cls, run_name: str) -> None:
        """提出形式に則った結果ファイルを出力

        Args:
            run_name (str): ランの名前
        """
        submission = pd.read_csv('input/sampleSubmission.csv')
        pred = Util.load(f'model/pred/{run_name}-test.pkl')
        # for i in range(pred.shape[1]):
        #     submission[f'Class_{i + 1}'] = pred[:, i]
        submission.to_csv(f'submission/{run_name}.csv', index=False)