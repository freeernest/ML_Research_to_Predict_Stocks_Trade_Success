from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas import Series, DataFrame
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import scale, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from seaborn import heatmap
import openpyxl
import pickle

RESULTS_NEW_CSV = 'results_new_tradingview_000_percents_new.csv'

FINALIZED_MODEL___SAV = 'finalized_model_tradingview.sav'

AS_PROFITABLE_NEW_CSV = "results_of_only_predicted_as_profitable.csv"


def load_report_prepare_model_and_print_results():
    df = pd.read_csv(RESULTS_NEW_CSV)
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
    print(df_no_missing.head())

    df_profitable = df_no_missing[df_no_missing['result_label'] == 1]
    df_non_profitable = df_no_missing[df_no_missing['result_label'] == 0]

    # show_full_confusion_matrix(df_no_missing)

    print(df_profitable.head())
    print('df_profitable length' + str(len(df_profitable)))
    print('df_non_profitable length' + str(len(df_non_profitable)))

    df_profitable_downsampled = resample(df_profitable, replace=False, n_samples=10691)
    df_non_profitable_downsampled = resample(df_non_profitable, replace=False, n_samples=10691)

    df_downsampled = pd.concat([df_profitable_downsampled, df_non_profitable_downsampled])
                                # df_profitable_downsampled, df_non_profitable_downsampled,
                                # df_profitable_downsampled, df_non_profitable_downsampled,
                                # df_profitable_downsampled, df_non_profitable_downsampled,
                                # df_profitable_downsampled, df_non_profitable_downsampled,
                                # df_profitable_downsampled, df_non_profitable_downsampled,
                                # df_profitable_downsampled, df_non_profitable_downsampled])

    # df_downsampled = df_no_missing

    print(df_downsampled.head())
    # x = df_downsampled.drop(['result_label',
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

    print("Number columns " + str(len(df_downsampled.columns)))

    x = df_downsampled.drop(['result_label', 'profit_loss'], axis=1).copy()
    print(x.head())

    y = df_downsampled['result_label'].copy()
    print(y.head())

    print("Number of '1' labels " + str(len(y.loc[y == 1])))
    print("Number of '0' labels " + str(len(y.loc[y == 0])))

    standardScaler = StandardScaler()
    standardScaler.fit_transform(df_no_missing.drop(['result_label', 'profit_loss'], axis=1).copy())
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    X_train_scaled = standardScaler.transform(X_train)
    X_test_scaled = standardScaler.transform(X_test)

    pickle.dump(standardScaler, open('standardScaler.sav', 'wb'))
    # print(len(X_train_scaled[0]))

    print("Number of '1' labels in test data " + str(len(y_test.loc[y_test == 1])))
    print("Number of '0' labels in test data " + str(len(y_test.loc[y_test == 0])))

    # perform_cross_validation_and_print_optimal_parameters(X_train_scaled, y_train)
    clf_svm: SVC = train_SVM_and_show_its_confusion_matrix(X_test_scaled, X_train_scaled, y_test, y_train)


    # save the model to disk
    filename = FINALIZED_MODEL___SAV
    pickle.dump(clf_svm, open(filename, 'wb'))

    # load the model from disk
    clf_svm: SVC = pickle.load(open(filename, 'rb'))


    show_full_confusion_matrix(df_no_missing)

    write_down_only_the_rows_which_predicted_as_profitable(clf_svm, df_no_missing, standardScaler)

    print_profit_sum(AS_PROFITABLE_NEW_CSV)
    print_profit_sum(RESULTS_NEW_CSV)

    # prepare_and_show_pca_diagram(X_train_scaled, y_train)
    # train_random_forest_and_show_its_confusion_matrix(X_test_scaled, X_train_scaled, y_test, y_train)
    # k_fold_cross_validation(x, y)


def print_profit_sum(file_path : str):
    profitable_df = pd.read_csv(file_path)
    print('profitable_df: \n')
    # print(profitable_df.head(20))
    profitable_df_sum = profitable_df['profit_loss'].copy().sum()
    profitable_df_size = profitable_df['profit_loss'].copy().size
    print(file_path + '\n profitable_df_sum = ' + str(profitable_df_sum))
    print(file_path + '\n profitable_df_size = ' + str(profitable_df_size))


def write_down_only_the_rows_which_predicted_as_profitable(clf_svm, df_no_missing, standardScaler : StandardScaler):
    results = open(AS_PROFITABLE_NEW_CSV, "a")
    results.truncate(0)
    header = 'MACD(),MACDdiff,MACD().Avg,RSI(),"ExpAverage(close, length = 9)","ExpAverage(close, length = 21)",' \
             '"ExpAverage(close, length = 34)","ExpAverage(close, length = 55)","ExpAverage(close, length = 88)",' \
             '"ExpAverage(close, length = 100)",BollingerBands().UpperBand,BollingerBands().LowerBand,CCI(),' \
             'StochasticFull().FullD,StochasticFull().FullK,imp_volatility,volume,close,GetTime(),result_label,profit_loss,predicted_result'
    results.write(header + "\n")
    print('df_no_missing \n')
    print(df_no_missing.head())
    df_no_missing_x = df_no_missing.drop(['result_label', 'profit_loss'], axis=1).copy()
    print(df_no_missing_x.head())
    df_no_missing_y = df_no_missing.loc[:, 'result_label': 'profit_loss']
    print(df_no_missing_y.head())
    df_no_missing_scaled_x = standardScaler.transform(df_no_missing_x)
    df_no_missing_scaled_x_df = pd.DataFrame(df_no_missing_scaled_x, columns=df_no_missing_x.columns)
    print('df_no_missing_scaled_x_df \n')
    print(df_no_missing_scaled_x_df.head())
    df_no_missing_scaled_x_df.reset_index(drop=True, inplace=True)
    df_no_missing_y.reset_index(drop=True, inplace=True)
    df_no_missing_scaled_df = pd.concat([df_no_missing_scaled_x_df, df_no_missing_y], 1)
    print('df_no_missing_scaled_df \n')
    print(df_no_missing_scaled_df.head())
    counter_of_0 = 0
    counter_of_1 = 0
    for index, row in df_no_missing_scaled_df.iterrows():

        shortened_row = row.drop('result_label').drop('profit_loss').copy()
        row_for_prediction = shortened_row.values.reshape(1, -1)
        # print(str(row_for_prediction))
        # print(str(row['profit_loss']))
        prediction = clf_svm.predict(row_for_prediction)
        # print(str(prediction[0]))

        if prediction[0] == 0:
            counter_of_0 += 1
            # row['profit_loss'] = 0
        else:
            pd.DataFrame(row).T.to_csv(results,
                                       index=False,
                                       header=False,
                                       mode='a')
            # results.write("," + str(prediction) + "\n")
            counter_of_1 += 1


    print('counter_of_0 ' + str(counter_of_0))
    print('counter_of_1 ' + str(counter_of_1))


def prepare_and_show_pca_diagram(X_train_scaled, y_train):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    # show_pca_relevance_graph(pca)
    train_pc1_coords = X_train_pca[:, 0]
    train_pc2_coords = X_train_pca[:, 1]
    pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))
    # perform_cross_validation_and_print_optimal_parameters(pca_train_scaled, y_train)
    # print("Starting SVM training for PCA scaled data")
    # clf_svm_pca = SVC(C=100000, gamma=1)
    # clf_svm_pca.fit(pca_train_scaled, y_train)
    # print("Finished SVM training for PCA scaled data")
    # save the model to disk
    filename_PCA = 'finalized_model_PCA.sav'
    # pickle.dump(clf_svm_pca, open(filename_PCA, 'wb'))
    # load the model from disk
    clf_svm_pca = pickle.load(open(filename_PCA, 'rb'))
    X_train_pca = pca.transform(X_train_scaled)
    print(str(X_train_scaled.shape))
    test_pc1_coords = X_train_pca[:, 0]
    test_pc2_coords = X_train_pca[:, 1]
    x_min = test_pc1_coords.min() - 1
    x_max = test_pc1_coords.max() + 1
    y_min = test_pc2_coords.min() - 1
    y_max = test_pc2_coords.max() + 1
    xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                         np.arange(start=y_min, stop=y_max, step=0.1))
    Z = clf_svm_pca.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, Z, alpha=0.1)
    cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
    print(str(y_train.shape))
    print(str(test_pc1_coords.shape))
    print(str(test_pc2_coords.shape))
    scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train,
                         cmap=cmap,
                         s=100,
                         edgecolors='k',
                         alpha=0.7)
    legend = ax.legend(scatter.legend_elements()[0],
                       scatter.legend_elements()[1],
                       loc="upper right")
    legend.get_texts()[0].set_text("Non Profitable")
    legend.get_texts()[1].set_text("Profitable")
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_title('Decision surface using the PCA transformed/projected features')
    plt.show()


def perform_cross_validation_and_print_optimal_parameters(pca_train_scaled, y_train):
    param_grid = [
        {'C': [1, 10, 100, 1000, 10000, 100000],
         "gamma": ['scale', 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
         'kernel': ['rbf']}
    ]
    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=100
    )
    optimal_params.fit(pca_train_scaled, y_train)
    print("CV optimal parameters: " + str(optimal_params.best_params_))


def train_random_forest_and_show_its_confusion_matrix(X_test_scaled, X_train_scaled, y_test, y_train):
    clf_forest = RandomForestClassifier(n_estimators=10000)
    clf_forest.fit(X_train_scaled, y_train)
    plot_confusion_matrix(clf_forest,
                          X_test_scaled,
                          y_test,
                          values_format='d',
                          display_labels=["Non Profitable", "Profitable"])
    plt.show()


def show_pca_relevance_graph(pca):
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # labels = [str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Screen Plot')
    plt.show()


def train_SVM_and_show_its_confusion_matrix(X_test_scaled, X_train_scaled, y_test, y_train):
    print("Partial Matrix")
    clf_svm = SVC(C=100000)
    clf_svm = clf_svm.fit(X_train_scaled, y_train)
    # print(clf_svm.predict(X_test_scaled))
    plot_confusion_matrix(clf_svm,
                          X_test_scaled,
                          y_test,
                          values_format='d',
                          display_labels=["Non Profitable", "Profitable"])
    plt.show()
    print("Partial Matrix")
    return clf_svm

def show_full_confusion_matrix(df_no_missing):
    print("Full Matrix")
    clf_svm: SVC = pickle.load(open(FINALIZED_MODEL___SAV, 'rb'))
    # print(clf_svm.predict(X_test_scaled))
    X = df_no_missing.drop(['result_label', 'profit_loss'], axis=1).copy()
    y = df_no_missing['result_label'].copy()

    standardScaler: StandardScaler = pickle.load(open("standardScaler.sav", 'rb'))

    X_scaled = standardScaler.transform(X)
    print("Full Matrix")
    plot_confusion_matrix(clf_svm,
                          X_scaled,
                          y,
                          values_format='d',
                          display_labels=["Non Profitable", "Profitable"])
    plt.show()
    print("Full Matrix")

def k_fold_cross_validation(x: DataFrame, y: DataFrame):
    # Implementing cross validation

    k = 10
    kf = KFold(n_splits=k, random_state=None)
    model = SVC(C=1000000)

    acc_score = []
    x_scaled = scale(x)

    # for train_index , test_index in kf.split(x_scaled):
    #     X_train, X_test = x_scaled[train_index, :], x_scaled[test_index, :]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     model.fit(X_train, y_train)
    #     pred_values = model.predict(X_test)
    #
    #     acc = accuracy_score(pred_values, y_test)
    #     acc_score.append(acc)
    #
    # avg_acc_score = sum(acc_score)/k
    #
    # print('accuracy of each fold - {}'.format(acc_score))
    # print('Avg accuracy : {}'.format(avg_acc_score))

    result = cross_val_score(model, x_scaled, y, cv=kf, verbose=100)
    print("Avg accuracy: {}".format(result.mean()))


if __name__ == '__main__':
    load_report_prepare_model_and_print_results()
