# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np

"""
sklearn 
Module          Ex.     
datasets        load_iris       アヤメのデータの読み込み
                load_digits     手書き画像データの読み込み
svm             SVC             分類用のサポートベクトルマシン
neural_network  MLPClassifier   分類用の多層ニューラルネットワーク
deomposition    PCA             主成分分析
linear_model    Perceptron      単純パーセプトロン
"""

# アヤメのデータをロードし、変数irisに格納
iris = datasets.load_iris()

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = iris.data
y = iris.target

# Xとyをそのまま表示
print(X)
print(y)

# Xとyの次元を表示。それぞれ(150, 4)と(150,)となる。
# サンプル数150, 特徴量の次元4, という意味
print(X.shape)
print(y.shape)

# サンプル数と特徴量の次元の取り出し方法
(n_samples, n_features) = X.shape
print('サンプル数: {0}'.format(n_samples))
print('特徴量の次元: {0}'.format(n_features))
print(f'項目数: {len(X.shape)}')

# クラス数の取り出し方法
n_classes = len(np.unique(y))
print('クラス数: {0}'.format(n_classes))
