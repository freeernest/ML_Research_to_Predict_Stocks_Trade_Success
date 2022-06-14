import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from seaborn import heatmap

df = pd.read_csv('results.csv')
print(len(df))
print(len(df.loc[(df['MACD()'] == 0)
                 | (df['MACDdiff'] == 0)
                 | (df['MACD().Avg'] == 0)
                 | (df['RSI()'] == 0)
                 | (df['ExpAverage(close, length = 9)'] == 0)
                 | (df['ExpAverage(close, length = 21)'] == 0)
                 | (df['ExpAverage(close, length = 34)'] == 0)
                 | (df['ExpAverage(close, length = 55)'] == 0)
                 | (df['ExpAverage(close, length = 88)'] == 0)
                 | (df['ExpAverage(close, length = 100)'] == 0)
                 | (df['BollingerBands().UpperBand'] == 0)
                 | (df['BollingerBands().LowerBand'] == 0)
                 | (df['CCI()'] == 0)
                 | (df['StochasticFull().FullD'] == 0)
                 | (df['StochasticFull().FullK'] == 0)
                 | (df['imp_volatility'] == 0)
                 | (df['volume'] == 0)
                 | (df['close'] == 0)
                 | (df['GetTime()'] == 0)]))

df_no_missing = df.loc[(df['MACD()'] != 0)
                       & (df['MACDdiff'] != 0)
                       & (df['MACD().Avg'] != 0)
                       & (df['RSI()'] != 0)
                       & (df['ExpAverage(close, length = 9)'] != 0)
                       & (df['ExpAverage(close, length = 21)'] != 0)
                       & (df['ExpAverage(close, length = 34)'] != 0)
                       & (df['ExpAverage(close, length = 55)'] != 0)
                       & (df['ExpAverage(close, length = 88)'] != 0)
                       & (df['ExpAverage(close, length = 100)'] != 0)
                       & (df['BollingerBands().UpperBand'] != 0)
                       & (df['BollingerBands().LowerBand'] != 0)
                       & (df['CCI()'] != 0)
                       & (df['StochasticFull().FullD'] != 0)
                       & (df['StochasticFull().FullK'] != 0)
                       & (df['imp_volatility'] != 0)
                       & (df['volume'] != 0)
                       & (df['close'] != 0)
                       & (df['GetTime()'] != 0)]

print(len(df_no_missing))

df_profitable = df_no_missing[df_no_missing['result_label'] == 1]
df_non_profitable = df_no_missing[df_no_missing['result_label'] == 0]

df_profitable_downsampled = resample(df_profitable, replace=False, n_samples=550)
df_non_profitable_downsampled = resample(df_non_profitable, replace=False, n_samples=550)

df_downsampled = pd.concat([df_profitable_downsampled, df_non_profitable_downsampled])

# X = df_downsampled.drop(['result_label',
#                          'MACD().Avg',
#                          'ExpAverage(close, length = 9)',
#                          'ExpAverage(close, length = 21)',
#                          'ExpAverage(close, length = 34)',
#                          'ExpAverage(close, length = 55)',
#                          'ExpAverage(close, length = 88)',
#                          'ExpAverage(close, length = 100)',
#                          'BollingerBands().UpperBand',
#                          'BollingerBands().LowerBand',
#                          'StochasticFull().FullD',
#                          'StochasticFull().FullK',
#                          'imp_volatility',
#                          'volume',
#                          'close',
#                          'GetTime()'],
#                         axis=1).copy()

X = df_downsampled.drop('result_label', axis=1).copy()
print(X.head())

y = df_downsampled['result_label'].copy()
print(y.head())

print(len(y.loc[y == 1]))
print(len(y.loc[y == 0]))

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

print(len(y_test.loc[y_test == 1]))
print(len(y_test.loc[y_test == 0]))
# print(X_train_scaled)
# print(X_test_scaled)
# print(X_test_scaled)

clf_svm = SVC(C=100000000)
clf_svm = clf_svm.fit(X_train_scaled, y_train)

# print(clf_svm.predict(X_test_scaled))

plot_confusion_matrix(clf_svm,
                      X_test_scaled,
                      y_test,
                      values_format='d',
                      display_labels=["Non Profitable", "Profitable"])

plt.show()



# clf_forest = RandomForestClassifier(n_estimators=100)
# clf_forest.fit(X_train_scaled, y_train)
#
# plot_confusion_matrix(clf_forest,
#                       X_test_scaled,
#                       y_test,
#                       values_format='d',
#                       display_labels=["Non Profitable", "Profitable"])
# plt.show()

# param_grid = [
#     {'C': [0.5, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000],
#      "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#      'kernel': ['rbf']}
# ]
#
# optimal_params = GridSearchCV(
#     SVC(),
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     verbose=100
# )

# optimal_params.fit(X_train_scaled, y_train)
#
# print(optimal_params.best_params_)

# param_grid = [
#     {'C': [10000000],
#      "gamma": [1e-3, 1e-2, 1e-1],
#      'kernel': ['rbf']}
# ]
#
# optimal_params = GridSearchCV(
#     SVC(),
#     param_grid,
#     cv=1,
#     scoring='accuracy',
#     verbose=100
# )
#
# optimal_params.fit(X_train_scaled, y_train)
#
# print(optimal_params.best_params_)

# clf_svm_2 = SVC(C=0.5, gamma=1)
# clf_svm_2 = SVC(C=100, gamma=0.001)
# clf_svm_2 = SVC(C=1000, gamma=1e-8)
# clf_svm_2 = SVC(C=1000000, gamma=0.0001)
# clf_svm_2 = SVC(C=1000, gamma=0.1)
# clf_svm_2 = clf_svm_2.fit(X_train_scaled, y_train)
#
# plot_confusion_matrix(clf_svm_2,
#                       X_test_scaled,
#                       y_test,
#                       values_format='d',
#                       display_labels=["Non Profitable", "Profitable"])
#
# plt.show()
