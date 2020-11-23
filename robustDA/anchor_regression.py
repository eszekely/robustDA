#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy import stats

from robustDA.process_cmip6 import read_files_cmip6
from robustDA.processing import split_train_test, split_folds_CV
from robustDA.plots import make_plots, plot_CV
from robustDA.utils import helpers


def standardize(dict_models):
    X_train = dict_models["X_train"]
    y_train = dict_models["y_train"]
    y_anchor_train = dict_models["y_anchor_train"]

    X_test = dict_models["X_test"]
    y_test = dict_models["y_test"]
    y_anchor_test = dict_models["y_anchor_test"]

    # Create a scaler object
    sc_X = StandardScaler(with_mean=True, with_std=True)
    sc_y = StandardScaler(with_mean=True, with_std=True)
    sc_y_anchor = StandardScaler(with_mean=True, with_std=True)
    sc_X_test = StandardScaler(with_mean=True, with_std=True)
    sc_y_test = StandardScaler(with_mean=True, with_std=True)
    sc_y_anchor_test = StandardScaler(with_mean=True, with_std=True)

    # Fit the scaler to the data and transform
    X_train_std = sc_X.fit_transform(X_train.values)
    y_train_std = sc_y.fit_transform(y_train.values)
    y_anchor_train_std = sc_y_anchor.fit_transform(y_anchor_train.values)
    X_test_std = sc_X_test.fit_transform(X_test.values)
    y_test_std = sc_y_test.fit_transform(y_test.values)
    y_anchor_test_std = sc_y_anchor_test.fit_transform(y_anchor_test.values)

    X = X_train_std
    y = y_train_std
    y_anchor = y_anchor_train_std
    Xt = X_test_std
    yt = y_test_std
    yt_anchor = y_anchor_test_std

    std_X_train = X_train.std(axis=0)

    return X, y, y_anchor, Xt, yt, yt_anchor, std_X_train


def build_column_space(y_anchor, h_anchors):
    """
    Build the column space of A using nonlinear functions
    for a given (one) anchor source
    """
    A_h = np.zeros((y_anchor.shape[0], len(h_anchors) + 1))
    A_h[:, 0] = y_anchor.reshape(-1)

    if len(h_anchors) > 0:
        for i in range(len(h_anchors)):
            A_h[:, i + 1] = helpers.nonlinear_anchors(y_anchor, h_anchors[i]).reshape(-1)

    A_h = np.mat(A_h)  # needed for matrix multiplication
    PA = A_h * np.linalg.inv(np.transpose(A_h) * A_h) * np.transpose(A_h)

    return PA


def projection_column_space(X, y, gamma, PA):
    N = X.shape[0]

    X_PA = (np.mat(np.identity(N)) - PA) * np.mat(X) + np.sqrt(
        gamma
    ) * PA * np.mat(X)

    y_PA = (np.mat(np.identity(N)) - PA) * np.mat(y) + np.sqrt(
        gamma
    ) * PA * np.mat(y)

    return X_PA, y_PA


def transformed_anchor_matrices(X, y, y_anchor, gamma, h_anchors):
    """ Build the transformed anchor matrices"""

    PA = build_column_space(y_anchor, h_anchors)

    X_PA, y_PA = projection_column_space(X, y, gamma, PA)

    return X_PA, y_PA


def anchor_regression_estimator(dict_models, gamma, h_anchors, regLambda, method="ridge"):
    """ Standardize the data before anchor regression """
    X, y, y_anchor, Xt, yt, yt_anchor, std_X_train = standardize(dict_models)

    """ Use either the ridge implementation or direct implementation.
    Ridge is usually faster and used by default.
    """

    PA = build_column_space(y_anchor, h_anchors)

    N, p = X.shape
    if method == "ridge":
        X_PA, y_PA = projection_column_space(X, y, gamma, PA)

        regr = linear_model.Ridge(alpha=N * regLambda / 2)
        regr.fit(X_PA, y_PA)

        coefStd = regr.coef_
        coefRaw = coefStd / np.array(std_X_train).reshape(1, p)

        y_test_pred = regr.predict(Xt)

    elif method == "direct":
        PA_C = np.identity(N) - PA

        D_lambda = (
            gamma * np.mat(X.T) * PA * np.mat(X)
            + np.mat(X.T) * np.mat(PA_C) * np.mat(X)
            + N * regLambda / 2 * np.identity(p)
        )

        d = gamma * np.mat(X.T) * PA * np.mat(y) + np.mat(X.T) * PA_C * np.mat(
            y
        )

        coefStd = np.linalg.inv(D_lambda) * d
        coefStd = np.asarray(coefStd.T)
        coefRaw = coefStd.T / np.array(std_X_train).reshape(p, 1)

        y_test_pred = np.array(np.matmul(Xt, coefStd.T))

    return coefRaw, y_test_pred


def run_anchor_regression_all(
    params_climate, params_anchor, display_CV_plot=False
):

    target = params_climate["target"]
    anchor = params_climate["anchor"]
    gamma = params_anchor["gamma"]
    h_anchors = params_anchor["h_anchors"]
    
    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    dict_models = split_train_test(
        modelsDataList, modelsInfoFrame, target, anchor
    )

    print(dict_models["trainModels"])
    print(dict_models["testModels"])
    
    for i in range(len(gamma)):
        print("Gamma: " + str(gamma[i]))

        if gamma[i] == 1:
            tmp_h_anchors = []
            run_anchor_regression(
                modelsDataList,
                modelsInfoFrame,
                dict_models,
                params_climate,
                gamma[i],
                tmp_h_anchors,
                display_CV_plot,) 
       
        else: # gamma different from 1
            """ Run anchor regression with the 
            nonlinear anchors
            """
            for j in range(len(h_anchors) + 1):
                tmp_h_anchors = h_anchors[: j]
                print(" ---- " + str(tmp_h_anchors))
                run_anchor_regression(
                    modelsDataList,
                    modelsInfoFrame,
                    dict_models,
                    params_climate,
                    gamma[i],
                    tmp_h_anchors,
                    display_CV_plot,) 

                
def run_anchor_regression(
    modelsDataList,
    modelsInfoFrame,
    dict_models,
    params_climate,
    gamma,
    h_anchors,
    display_CV_plot,
):
    
    lambdaSelAll, _, _ = cross_validation_anchor_regression(
        modelsDataList,
        modelsInfoFrame,
        dict_models,
        params_climate,
        gamma,
        h_anchors,
        display_CV_plot,
    )

    anchor_regression(
        dict_models, gamma, h_anchors, lambdaSelAll[1], params_climate, 1
    )

    anchor_regression(
        dict_models, gamma, h_anchors, lambdaSelAll[2], params_climate, 2
    )

    
def anchor_regression(
    dict_models,
    gamma,
    h_anchors,
    lambdaSel,
    params_climate, 
    nbSem,
):
    
    coefRaw, y_test_pred = anchor_regression_estimator(
        dict_models, gamma, h_anchors, lambdaSel
    )

    filename = (
        "target_"
        + params_climate["target"]
        + "_"
        + "anchor_"
        + params_climate["anchor"]
        + "_"
        + "-".join(params_climate["variables"])
        + "_"
        + "-".join(params_climate["scenarios"])
        + "_"
        + str(params_climate["startDate"])
        + "_"
        + str(params_climate["endDate"])
        + "_"
        + "gamma_"
        + str(gamma)
        + "_"
        + "nonlinear-h_"
        + str(len(h_anchors)) 
        + "-".join(h_anchors)
        + "_"
        + "lambda_"
        + str(nbSem)
        + "SEM-"
        + str(np.round(lambdaSel, 2))
        + ".pdf"
    )

    make_plots(
        dict_models,
        coefRaw,
        y_test_pred,
        dict_models["y_anchor_test"],
        gamma,
        params_climate["target"],
        params_climate["anchor"],
        h_anchors,
        filename,
    )
            
                
def cross_validation_anchor_regression(
    modelsDataList,
    modelsInfoFrame,
    dict_models,
    params_climate,
    gamma,
    h_anchors,
    display_CV_plot=False,
):

    uniqueTrain = set(
        [
            dict_models["trainFiles"][i].split("_")[2][:3]
            for i in range(len(dict_models["trainFiles"]))
        ]
    )

    dict_folds = split_folds_CV(
        modelsDataList,
        modelsInfoFrame,
        dict_models,
        params_climate["target"],
        params_climate["anchor"],
        nbFoldsCV=len(uniqueTrain),
        displayModels=False,
    )

    lambdasCV = np.logspace(-2, 6, 100)

    nbFoldsCV = len(dict_folds["foldsData"])
    mse = np.zeros([len(lambdasCV), nbFoldsCV])

    for i in range(nbFoldsCV):
        X_val_df = dict_folds["foldsData"][i]
        y_val_df = dict_folds["foldsTarget"][i]
        y_anchor_val_df = dict_folds["foldsAnchor"][i]

        X_train_CV_df = pd.DataFrame()
        y_train_CV_df = pd.DataFrame()
        y_anchor_train_CV_df = pd.DataFrame()

        for j in range(nbFoldsCV):
            if j != i:
                X_train_CV_df = pd.concat(
                    [X_train_CV_df, dict_folds["foldsData"][j]], axis=0
                )
                y_train_CV_df = pd.concat(
                    [y_train_CV_df, dict_folds["foldsTarget"][j]], axis=0
                )
                y_anchor_train_CV_df = pd.concat(
                    [y_anchor_train_CV_df, dict_folds["foldsAnchor"][j]],
                    axis=0,
                )

        dict_models = {
            #             "trainModels": trainModels,
            #             "testModels": testModels,
            #             "trainFiles": trainFiles,
            #             "testFiles": testFiles,
            "X_train": X_train_CV_df,
            "y_train": y_train_CV_df,
            "y_anchor_train": y_anchor_train_CV_df,
            "X_test": X_val_df,
            "y_test": y_val_df,
            "y_anchor_test": y_anchor_val_df,
        }

        (
            X,
            y,
            y_anchor,
            X_val,
            y_val_true,
            y_anchor_val,
            std_X_train,
        ) = standardize(dict_models)

        X_PA, y_PA = transformed_anchor_matrices(X, y, y_anchor, gamma, h_anchors)

        for j in range(len(lambdasCV)):
            regr = linear_model.Ridge(alpha=X.shape[0] * lambdasCV[j] / 2)
            regr.fit(X_PA, y_PA)
            y_val_pred = regr.predict(X_val)
            mse[j][i] = np.mean((y_val_true - y_val_pred) ** 2)

    columnNames = ["MSE - Fold " + str(i) for i in range(nbFoldsCV)]
    mse_df = pd.DataFrame(mse, index=lambdasCV, columns=columnNames)
    mse_df["MSE - TOTAL"] = np.mean(mse, axis=1)
    mse_df.index.name = "Lambda"

    lambdasSelAll, sem_CV = choose_lambda(mse_df, lambdasCV)

    if display_CV_plot:
        filename = (
            "target_"
            + params_climate["target"]
            + "_"
            + "anchor_"
            + params_climate["anchor"]
            + "_"
            + "-".join(params_climate["variables"])
            + "_"
            + "-".join(params_climate["scenarios"])
            + "_"
            + str(params_climate["startDate"])
            + "_"
            + str(params_climate["endDate"])
            + "_"
            + "gamma_"
            + str(gamma)
            + "_"
            + "nonlinear-h_"
            + str(len(h_anchors)) 
            + "-".join(h_anchors)
            + "_CV.pdf"
        )
        plot_CV(mse_df, lambdasSelAll, sem_CV, filename)

    # mse_df.to_csv("./../output/data/MSE_rcp85_norm_train30_cv5_lambda0_5_100.csv")

    return lambdasSelAll, mse_df, sem_CV


def choose_lambda(mse_df, lambdasCV):
    mse_total = mse_df["MSE - TOTAL"]

    nbSEM = np.array([1, 2, 3])

    """ Compute Standard error of the mean (SEM): σ_μ = σ / \\sqrt{K} """
    sem_CV = np.zeros(mse_df.shape[0])
    for i in range(mse_df.shape[0]):
        sem_CV[i] = stats.sem(mse_df.iloc[i, :-1], ddof=0)

    lambdaOpt = mse_total[mse_total == np.min(mse_total)].index[0]
    mse_total_sel = mse_total[lambdaOpt:]

    lambdasSelAll = np.zeros(len(nbSEM) + 1)
    lambdasSelAll[0] = lambdaOpt

    """ Compute lambda for different SEMs """
    for j in range(len(nbSEM)):
        lambdaSel = mse_total_sel[
            mse_total_sel
            == np.max(
                mse_total_sel[
                    mse_total_sel
                    <= np.min(mse_total_sel)
                    + nbSEM[j] * stats.sem(mse_df.loc[lambdaOpt][:-1], ddof=0)
                ]
            )
        ].index[0]
        lambdasSelAll[j + 1] = lambdaSel

    return lambdasSelAll, sem_CV
