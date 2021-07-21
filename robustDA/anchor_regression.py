#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

from robustDA.process_cmip6 import read_files_cmip6, read_forcing_cmip6
from robustDA.processing import split_train_test, split_folds_CV
from robustDA.plots import (
    make_plots,
    plot_CV_sem,
    plot_CV_multipleMSE,
    plot_Pareto,
)
from robustDA.utils import helpers


def standardize(dict_models):
    X_train_std = StandardScaler().fit_transform(dict_models["X_train"].values)
    y_train_std = StandardScaler().fit_transform(dict_models["y_train"].values)
    y_anchor_train_ctd = StandardScaler(
        with_mean=True, with_std=False
    ).fit_transform(dict_models["y_anchor_train"].values)

    X_test_std = StandardScaler(with_mean=True, with_std=True).fit_transform(
        dict_models["X_test"].values
    )
    y_test_std = StandardScaler(with_mean=True, with_std=True).fit_transform(
        dict_models["y_test"].values
    )
    y_anchor_test_ctd = StandardScaler(
        with_mean=True, with_std=False
    ).fit_transform(dict_models["y_anchor_test"].values)

    std_X_train = dict_models["X_train"].values.std(axis=0)
    std_y_train = dict_models["y_train"].values.std()
    std_X_test = dict_models["X_test"].values.std(axis=0)
    std_y_test = dict_models["y_test"].values.std()

    return (
        X_train_std,
        y_train_std,
        y_anchor_train_ctd,
        X_test_std,
        y_test_std,
        y_anchor_test_ctd,
        std_X_train,
        std_y_train,
        std_X_test,
        std_y_test,
    )


def build_column_space(y_anchor, h_anchors):
    """
    Build the column space of A using nonlinear functions
    for a given (one) anchor source
    """
    if h_anchors is None:
        A_h = np.zeros((y_anchor.shape[0], 1))
    else:
        A_h = np.zeros((y_anchor.shape[0], len(h_anchors) + 1))

    A_h[:, 0] = y_anchor.reshape(-1)

    if h_anchors is not None:
        for i in range(len(h_anchors)):
            A_h[:, i + 1] = helpers.nonlinear_anchors(
                y_anchor, h_anchors[i]
            ).reshape(-1)

    A_h = np.mat(A_h)  # needed for matrix multiplication
    #     PA = A_h * np.linalg.inv(np.transpose(A_h) * A_h) * np.transpose(A_h)
    A_h = np.mat(
        StandardScaler(with_mean=True, with_std=False).fit_transform(A_h)
    )
    PA = (
        A_h
        * np.linalg.inv(np.transpose(A_h) * A_h)
        * np.transpose(A_h)
    )

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


def anchor_regression_estimator(
    dict_models, gamma, h_anchors, regLambda, method="ridge", pred="original"
):
    """ Standardize the data before anchor regression """
    X, y, y_anchor, Xt, yt, yt_anchor, std_X_train, std_y_train = standardize(
        dict_models
    )

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

    #     y_test_pred = (
    #         StandardScaler()
    #         .fit(dict_models["y_test"])
    #         .inverse_transform(y_test_pred)
    #     )

    mse = np.mean((yt - y_test_pred) ** 2)

    return coefRaw, yt, y_test_pred, mse


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
                display_CV_plot,
            )

        else:  # gamma different from 1
            """Run anchor regression with the
            nonlinear anchors
            """
            for j in range(len(h_anchors) + 1):
                tmp_h_anchors = h_anchors[:j]
                print(" ---- " + str(tmp_h_anchors))
                run_anchor_regression(
                    modelsDataList,
                    modelsInfoFrame,
                    dict_models,
                    params_climate,
                    gamma[i],
                    tmp_h_anchors,
                    display_CV_plot,
                )


def run_anchor_regression(
    modelsDataList,
    modelsInfoFrame,
    dict_models,
    params_climate,
    gamma,
    h_anchors,
    display_CV_plot,
):

    cv_vals = 50
    sel_method = "MSE"

    (
        lambdaSelAll,
        mse_df,
        sem_CV,
        corr_pearson,
        mi,
    ) = cross_validation_anchor_regression(
        modelsDataList,
        modelsInfoFrame,
        deepcopy(dict_models),
        params_climate,
        gamma,
        h_anchors,
        cv_vals,
        sel_method,
        display_CV_plot,
    )

    nbSem = 1
    anchor_regression(
        dict_models,
        gamma,
        h_anchors,
        lambdaSelAll[nbSem],
        params_climate,
        nbSem,
    )

    nbSem = 2
    anchor_regression(
        dict_models,
        gamma,
        h_anchors,
        lambdaSelAll[nbSem],
        params_climate,
        nbSem,
    )


def anchor_regression(
    dict_models,
    gamma,
    h_anchors,
    lambdaSel,
    params_climate,
    nbSem=2,
):

    coefRaw, y_test_pred, mse = anchor_regression_estimator(
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
        #         + str(nbSem)
        #         + "SEM-"
        + str(np.round(lambdaSel, 2))
        + "_noAhStd.pdf"
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
    cv_lambda_vals,
    sel_method="MSE",
    display_CV_plot=False,
):

    uniqueTrain = set(
        [
            dict_models["trainFiles"][i].split("_")[2][:3]
            for i in range(len(dict_models["trainFiles"]))
        ]
    )

    """ Leave-one-model-out CV """
    dict_folds = split_folds_CV(
        modelsDataList,
        modelsInfoFrame,
        dict_models,
        params_climate["target"],
        params_climate["anchor"],
        nbFoldsCV=len(uniqueTrain),
        displayModels=False,
    )

    lambdasCV = np.logspace(-2, 6, cv_lambda_vals)

    nbFoldsCV = len(dict_folds["foldsData"])
    mse = np.zeros([len(lambdasCV), nbFoldsCV])
    corr_pearson = np.zeros([len(lambdasCV), nbFoldsCV])
    mi = np.zeros([len(lambdasCV), nbFoldsCV])

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

        dict_models_CV = {
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
        ) = standardize(dict_models_CV)

        X_PA, y_PA = transformed_anchor_matrices(
            X, y, y_anchor, gamma, h_anchors
        )

        for j in range(len(lambdasCV)):
            regr = linear_model.Ridge(alpha=X.shape[0] * lambdasCV[j] / 2)
            regr.fit(X_PA, y_PA)
            y_val_pred = regr.predict(X_val)
            residuals = (y_val_true - y_val_pred).reshape(-1)

            mse[j][i] = np.mean((y_val_true - y_val_pred) ** 2)
            corr_pearson[j][i] = np.round(
                np.corrcoef(
                    np.transpose(y_anchor_val), np.transpose(residuals)
                )[0, 1],
                2,
            )
            mi[j][i] = np.round(
                mutual_info_regression(y_anchor_val, residuals)[0], 2
            )

    columnNames = ["MSE - Fold " + str(i) for i in range(nbFoldsCV)]
    mse_df = pd.DataFrame(mse, index=lambdasCV, columns=columnNames)
    mse_df["MSE - TOTAL"] = np.mean(mse, axis=1)
    mse_df.index.name = "Lambda"

    if sel_method == "pareto":
        lambdaSel, _, _, _ = choose_lambda_pareto(
            mse_df.iloc[:, -1].values,
            np.mean(corr_pearson, axis=1),
            mse_df.index,
            maxX=False,
            maxY=False,
        )
    elif sel_method == "multipleMSE":
        lambdaSel = choose_lambda_multipleMSE(mse_df, lambdasCV)

    elif sel_method == "MSE":
        lambdaSel, sem_CV = choose_lambda(mse_df, lambdasCV)

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
            + "_CV_noAhStd.pdf"
        )
        if sel_method == "multipleMSE":
            plot_CV_multipleMSE(
                mse_df,
                lambdaSel,
                filename,
                dict_folds["trainFolds"],
            )
        elif sel_method == "MSE":
            plot_CV_sem(
                mse_df,
                lambdaSel,
                sem_CV,
                filename,
                dict_folds["trainFolds"],
            )
        elif sel_method == "pareto":
            plot_Pareto(
                mse_df,
                lambdaSel,
                filename,
                dict_folds["trainFolds"],
            )

    return lambdaSel, mse_df, corr_pearson, mi, lambdasCV


def choose_lambda_pareto(
    Xs, Ys, lambdavals, maxX=True, maxY=True, plot_var=True
):
    """Pareto frontier selection process"""
    Xs = np.abs(Xs)
    Ys = np.abs(Ys)

    sorted_list = sorted(
        [[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY
    )
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    ideal = [min(Xs), min(Ys)]
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    dst = compute_distance(ideal, pf_X, pf_Y)
    ind = np.argmin(dst)
    lambdaSel = [
        lambdavals[i]
        for i in range(len(lambdavals))
        if (Xs[i] == pf_X[ind]) and (Ys[i] == pf_Y[ind])
    ][0]

    if plot_var:
        """Plotting process"""
        plt.scatter(Xs, Ys)
        plt.plot(pf_X, pf_Y)
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")

        plt.plot(ideal[0], ideal[1], "k*")
        plt.plot(pf_X[ind], pf_Y[ind], "ro")
        plt.show()

    return lambdaSel, pf_X, pf_Y, ind


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


def choose_lambda_multipleMSE(mse_df, lambdasCV):
    mse_total = mse_df["MSE - TOTAL"]

    mult = [1.05, 1.10]

    lambdaOpt = mse_total[mse_total == np.min(mse_total)].index[0]
    mse_total_sel = mse_total[lambdaOpt:]

    lambdasSelAll = np.zeros(len(mult) + 1)
    lambdasSelAll[0] = lambdaOpt

    """ Compute lambda for different multiplication factors """
    for j in range(len(mult)):
        lambdaSel = mse_total_sel[
            mse_total_sel
            == np.max(
                mse_total_sel[mse_total_sel <= mult[j] * np.min(mse_total_sel)]
            )
        ].index[0]
        lambdasSelAll[j + 1] = lambdaSel

    return lambdasSelAll


def subagging(params_climate, params_anchor, nbRuns):

    cv_vals = 10

    print("Subagging")

    target = params_climate["target"]
    anchor = params_climate["anchor"]
    startDate = params_climate["startDate"]
    endDate = params_climate["endDate"]
    gamma = params_anchor["gamma"][0]
    h_anchors = params_anchor["h_anchors"]

    grid = (72, 144)
    nbYears = endDate - startDate + 1

    mse_runs = np.zeros([nbRuns, cv_vals])
    corr_pearson_runs = np.zeros([nbRuns, cv_vals])
    mi_runs = np.zeros([nbRuns, cv_vals])
    coefRaw_runs = np.zeros([nbRuns, grid[0] * grid[1]])
    y_test_pred_runs = [[] for i in range(nbRuns)]
    lambdaSel_runs = np.zeros([nbRuns, 1])
    trainFiles = []
    testFiles = []

    yf = read_forcing_cmip6("historical", target, startDate, endDate)
    testStatistic_null = []
    testStatistic_alt = []

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    for r in range(nbRuns):
        print("---- Run = " + str(r) + " ----")

        dict_models = split_train_test(
            modelsDataList,
            modelsInfoFrame,
            params_climate["target"],
            params_climate["anchor"],
        )

        #         print(dict_models["trainModels"])
        #         print(dict_models["testModels"])

        trainFiles.append(dict_models["trainFiles"])
        testFiles.append(dict_models["testFiles"])

        (
            lambdaSel,
            mse_df,
            corr_pearson,
            mi,
            lambdasCV,
        ) = cross_validation_anchor_regression(
            modelsDataList,
            modelsInfoFrame,
            deepcopy(dict_models),
            params_climate,
            gamma,
            h_anchors,
            cv_vals,
            sel_method="MSE",
            display_CV_plot=False,
        )

        """ Train with the chosen lambda to learn the coefficients beta,
        and get the error of each run by testing with the test set
        """

        nbSem = 2
        coefRaw, y_test_pred, mse = anchor_regression_estimator(
            dict_models, gamma, h_anchors, lambdaSel[nbSem]
        )

        lambdaSel_runs[r] = lambdaSel[nbSem]
        mse_runs[r, :] = mse_df["MSE - TOTAL"].values
        corr_pearson_runs[r, :] = np.mean(corr_pearson, axis=1)
        mi_runs[r, :] = np.mean(mi, axis=1)
        coefRaw_runs[r, :] = coefRaw
        y_test_pred_runs[r].append(y_test_pred)

        for i in range(len(dict_models["testFiles"])):
            if dict_models["testFiles"][i].split("_")[3] == "piControl":
                testStatistic_null.append(
                    np.corrcoef(
                        np.transpose(
                            y_test_pred[i * nbYears : (i + 1) * nbYears]
                        ),
                        np.transpose(yf.values.reshape(-1, 1)),
                    )[0, 1]
                )
            elif dict_models["testFiles"][i].split("_")[3] != "piControl":
                testStatistic_alt.append(
                    np.corrcoef(
                        np.transpose(
                            y_test_pred[i * nbYears : (i + 1) * nbYears]
                        ),
                        np.transpose(yf.values.reshape(-1, 1)),
                    )[0, 1]
                )

    mse_runs_df = pd.DataFrame(mse_runs)
    mse_runs_df["Lambda selected"] = pd.DataFrame(lambdaSel_runs)
    mse_runs_df.index.name = "Run"

    filename = (
        "./../output/data/subagging_"
        + target
        + "_"
        + anchor
        + "_gamma_"
        + str(gamma)
        + "_square_noAhstd.pkl"
    )

    dirname = "./../output/data/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(filename, "wb") as f:
        pickle.dump(
            [
                lambdaSel_runs,
                coefRaw_runs,
                mse_runs_df,
                corr_pearson_runs,
                mi_runs,
                y_test_pred_runs,
                testStatistic_null,
                testStatistic_alt,
            ],
            f,
        )


def param_optimization(params_climate, params_anchor, cv_vals):

    print("Param optimization")
    target = params_climate["target"]
    anchor = params_climate["anchor"]
    gamma_vals = params_anchor["gamma"]
    h_anchors = params_anchor["h_anchors"]

    mse_gamma = np.zeros([len(gamma_vals), len(h_anchors) + 1, cv_vals])
    corr_gamma = np.zeros([len(gamma_vals), len(h_anchors) + 1, cv_vals])
    mi_gamma = np.zeros([len(gamma_vals), len(h_anchors) + 1, cv_vals])

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    dict_models = split_train_test(
        modelsDataList, modelsInfoFrame, target, anchor
    )

    for i in range(len(gamma_vals)):
        print("Gamma = " + str(gamma_vals[i]))
        for j in range(len(h_anchors)):
            tmp_h_anchors = h_anchors
            print(" ---- " + str(tmp_h_anchors))
            #         for j in range(len(h_anchors) + 1):
            #             tmp_h_anchors = h_anchors[:j]
            #             print(" ---- " + str(tmp_h_anchors))
            #             (
            #                 lambdaSelAll,
            #                 mse_df,
            #                 sem_CV,
            #                 corr_pearson,
            #                 mi,
            #             ) = cross_validation_anchor_regression(
            #                 modelsDataList,
            #                 modelsInfoFrame,
            #                 deepcopy(dict_models),
            #                 params_climate,
            #                 gamma_vals[i],
            #                 tmp_h_anchors,
            #                 cv_vals,
            #                 display_CV_plot=False,
            #             )

            (
                lambdaSel,
                mse_df,
                corr_pearson,
                mi,
            ) = cross_validation_anchor_regression(
                modelsDataList,
                modelsInfoFrame,
                deepcopy(dict_models),
                params_climate,
                gamma_vals[i],
                tmp_h_anchors,
                cv_vals,
                sel_method="pareto",
                display_CV_plot=False,
            )

            mse_gamma[i, j, :] = mse_df.iloc[:, -1].values
            corr_gamma[i, j, :] = np.mean(corr_pearson, axis=1)
            mi_gamma[i, j, :] = np.mean(mi, axis=1)

    dirname = "./../output/data/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    filename = (
        dirname
        + "param_optimization_target_"
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
        + "nonlinear-h_"
        + str(len(h_anchors))
        + "-".join(h_anchors)
        + ".pkl"
    )
    with open(filename, "wb") as f:
        pickle.dump(
            [
                mse_gamma,
                corr_gamma,
                mi_gamma,
                gamma_vals,
                h_anchors,
                cv_vals,
                dict_models,
            ],
            f,
        )


def cross_validation_gamma_lambda(
    modelsDataList,
    modelsInfoFrame,
    dict_models,
    params_climate,
    gamma_vals,
    cv_vals,
    h_anchors,
    display_CV_plot=False,
):

    nb_folds_CV = 3
    lambda_vals = np.logspace(0, 9, cv_vals)
    
    """ LOOCV """
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
        nbFoldsCV=nb_folds_CV, # len(uniqueTrain),
        displayModels=False,
    )

    grid = (72, 144)
    p = grid[0] * grid[1]

    rmse_bag_folds_lin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    corr_bag_folds_lin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    mi_bag_folds_lin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    coef_std_bag_folds_lin = np.zeros(
        [nb_folds_CV, len(gamma_vals), len(lambda_vals), p]
    )
    coef_raw_bag_folds_lin = np.zeros(
        [nb_folds_CV, len(gamma_vals), len(lambda_vals), p]
    )
    rmse_test_lin = np.zeros([len(gamma_vals), len(lambda_vals)])
    corr_test_lin = np.zeros([len(gamma_vals), len(lambda_vals)])
    mi_test_lin = np.zeros([len(gamma_vals), len(lambda_vals)])
    
    rmse_bag_folds_nonlin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    corr_bag_folds_nonlin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    corr2_bag_folds_nonlin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    mi_bag_folds_nonlin = np.zeros([nb_folds_CV, len(gamma_vals), len(lambda_vals)])
    coef_std_bag_folds_nonlin = np.zeros(
        [nb_folds_CV, len(gamma_vals), len(lambda_vals), p]
    )
    coef_raw_bag_folds_nonlin = np.zeros(
        [nb_folds_CV, len(gamma_vals), len(lambda_vals), p]
    )
    rmse_test_nonlin = np.zeros([len(gamma_vals), len(lambda_vals)])
    corr_test_nonlin = np.zeros([len(gamma_vals), len(lambda_vals)])
    corr2_test_nonlin = np.zeros([len(gamma_vals), len(lambda_vals)])
    mi_test_nonlin = np.zeros([len(gamma_vals), len(lambda_vals)])

    for k in range(nb_folds_CV):
        print(" === Fold " + str(k) + " ===", flush=True)
        X_val_df = dict_folds["foldsData"][k]
        y_val_df = dict_folds["foldsTarget"][k]
        y_anchor_val_df = dict_folds["foldsAnchor"][k]

        X_train_CV_df = pd.DataFrame()
        y_train_CV_df = pd.DataFrame()
        y_anchor_train_CV_df = pd.DataFrame()

        for m in range(nb_folds_CV):
            if m != k:
                X_train_CV_df = pd.concat(
                    [X_train_CV_df, dict_folds["foldsData"][m]], axis=0
                )
                y_train_CV_df = pd.concat(
                    [y_train_CV_df, dict_folds["foldsTarget"][m]], axis=0
                )
                y_anchor_train_CV_df = pd.concat(
                    [y_anchor_train_CV_df, dict_folds["foldsAnchor"][m]],
                    axis=0,
                )

        dict_models_CV = {
            "X_train": X_train_CV_df,
            "y_train": y_train_CV_df,
            "y_anchor_train": y_anchor_train_CV_df,
            "X_val": X_val_df,  # need to call it 'X_test' and 'y_test'
            "y_val": y_val_df,  # to be able to use the 'standardize' method
            "y_anchor_val": y_anchor_val_df,
        }

        #         (
        #             X,
        #             y,
        #             y_anchor,
        #             X_val,
        #             y_val_true,
        #             y_anchor_val,
        #             std_X_train,
        #             std_y_train,
        #         ) = standardize(dict_models_CV)

        X_train_std = StandardScaler().fit_transform(
            dict_models_CV["X_train"].values
        )
        sc_y_train = StandardScaler()
        y_train_std = sc_y_train.fit_transform(
            dict_models_CV["y_train"].values
        )
        y_anchor_train_ctd = StandardScaler(
            with_mean=True, with_std=False
        ).fit_transform(dict_models_CV["y_anchor_train"].values)

        sc_X_val = StandardScaler(with_mean=True, with_std=True)
        X_val_std = sc_X_val.fit_transform(dict_models_CV["X_val"].values)

        y_val = dict_models_CV["y_val"].values

        std_X_train = dict_models_CV["X_train"].values.std(axis=0)
        std_y_train = dict_models_CV["y_train"].values.std()

        for i in range(len(gamma_vals)):

            ''' Linear anchor '''
            X_PA, y_PA = transformed_anchor_matrices(
                X_train_std,
                y_train_std,
                y_anchor_train_ctd,
                gamma_vals[i],
                None,
            )

            for j in range(len(lambda_vals)):
                regr = linear_model.Ridge(alpha=lambda_vals[j])
                regr.fit(X_PA, y_PA)

                coef_std = regr.coef_
                coef_raw = (
                    coef_std
                    * std_y_train
                    / np.array(std_X_train).reshape(1, p)
                )

                y_val_pred_std = regr.predict(X_val_std)
                y_val_pred = sc_y_train.inverse_transform(y_val_pred_std)

                residuals = (y_val - y_val_pred).reshape(-1)

                coef_std_bag_folds_lin[k, i, j, :] = coef_std
                coef_raw_bag_folds_lin[k, i, j, :] = coef_raw

                rmse_bag_folds_lin[k, i, j] = np.sqrt(np.mean(residuals ** 2))

                corr_bag_folds_lin[k, i, j] = np.round(
                    np.corrcoef(
                        np.transpose(dict_models_CV["y_anchor_val"].values),
                        np.transpose(residuals),
                    )[0, 1],
                    2,
                )
                
                mi_bag_folds_lin[k, i, j] = np.round(
                    mutual_info_regression(
                        dict_models_CV["y_anchor_val"].values, residuals
                    )[0],
                    2,
                )
                
            ''' Nonlinear anchor '''
            X_PA, y_PA = transformed_anchor_matrices(
                X_train_std,
                y_train_std,
                y_anchor_train_ctd,
                gamma_vals[i],
                h_anchors,
            )

            for j in range(len(lambda_vals)):
                regr = linear_model.Ridge(alpha=lambda_vals[j])
                regr.fit(X_PA, y_PA)

                coef_std = regr.coef_
                coef_raw = (
                    coef_std
                    * std_y_train
                    / np.array(std_X_train).reshape(1, p)
                )

                y_val_pred_std = regr.predict(X_val_std)
                y_val_pred = sc_y_train.inverse_transform(y_val_pred_std)

                residuals = (y_val - y_val_pred).reshape(-1)

                coef_std_bag_folds_nonlin[k, i, j, :] = coef_std
                coef_raw_bag_folds_nonlin[k, i, j, :] = coef_raw

                rmse_bag_folds_nonlin[k, i, j] = np.sqrt(np.mean(residuals ** 2))

                corr_bag_folds_nonlin[k, i, j] = np.round(
                    np.corrcoef(
                        np.transpose(dict_models_CV["y_anchor_val"].values),
                        np.transpose(residuals),
                    )[0, 1],
                    2,
                )
                
                corr2_bag_folds_nonlin[k, i, j] = np.round(
                    np.corrcoef(
                        np.transpose(dict_models_CV["y_anchor_val"].values ** 2),
                        np.transpose(residuals),
                    )[0, 1],
                    2,
                )
                    
                mi_bag_folds_nonlin[k, i, j] = np.round(
                    mutual_info_regression(
                        dict_models_CV["y_anchor_val"].values, residuals
                    )[0],
                    2,
                )
                
    rmse_bag_av_lin = np.mean(rmse_bag_folds_lin, axis=0)
    corr_bag_av_lin = np.mean(corr_bag_folds_lin, axis=0)
    mi_bag_av_lin = np.mean(mi_bag_folds_lin, axis=0)

    rmse_bag_av_nonlin = np.mean(rmse_bag_folds_nonlin, axis=0)
    corr_bag_av_nonlin = np.mean(corr_bag_folds_nonlin, axis=0)
    corr2_bag_av_nonlin = np.mean(corr2_bag_folds_nonlin, axis=0)
    mi_bag_av_nonlin = np.mean(mi_bag_folds_nonlin, axis=0)

    coef_std_bag_av_lin = np.mean(coef_std_bag_folds_lin, axis=0)
    coef_raw_bag_av_lin = np.mean(coef_raw_bag_folds_lin, axis=0)
    
    coef_std_bag_av_nonlin = np.mean(coef_std_bag_folds_nonlin, axis=0)
    coef_raw_bag_av_nonlin = np.mean(coef_raw_bag_folds_nonlin, axis=0)


    """ Evaluation on the held-out test data for all parameter values"""
    sc_y_train = StandardScaler(with_mean=True, with_std=True)
    y_train_std = sc_y_train.fit_transform(dict_models["y_train"].values)

    sc_X_test = StandardScaler(with_mean=True, with_std=True)
    X_test_std = sc_X_test.fit_transform(dict_models["X_test"].values)

    y_test = dict_models["y_test"].values
    
#     sc_y_test = StandardScaler(with_mean=True, with_std=True)
#     y_test_std = sc_y_test.fit_transform(dict_models["y_test"].values)


    for i in range(len(gamma_vals)):
        ''' Linear anchor '''
        for j in range(len(lambda_vals)):
            coef_std = coef_std_bag_av_lin[i, j, :].reshape(1, -1)
            y_test_pred_std = np.array(
                np.matmul(X_test_std, np.transpose(coef_std))
            )

            y_test_pred = sc_y_train.inverse_transform(y_test_pred_std)
            residuals = (y_test - y_test_pred).reshape(-1)

            rmse_test_lin[i, j] = np.sqrt(np.mean(residuals ** 2))

            corr_test_lin[i, j] = np.round(
                np.corrcoef(
                    np.transpose(dict_models["y_anchor_test"].values),
                    np.transpose(residuals),
                )[0, 1],
                2,
            )

            mi_test_lin[i, j] = np.round(
                mutual_info_regression(
                    dict_models["y_anchor_test"].values, residuals
                )[0],
                2,
            )
        
        ''' Nonlinear anchor '''
        for j in range(len(lambda_vals)):
            coef_std = coef_std_bag_av_nonlin[i, j, :].reshape(1, -1)
            y_test_pred_std = np.array(
                np.matmul(X_test_std, np.transpose(coef_std))
            )

            y_test_pred = sc_y_train.inverse_transform(y_test_pred_std)
            residuals = (y_test - y_test_pred).reshape(-1)

            rmse_test_nonlin[i, j] = np.sqrt(np.mean(residuals ** 2))

            corr_test_nonlin[i, j] = np.round(
                np.corrcoef(
                    np.transpose(dict_models["y_anchor_test"].values),
                    np.transpose(residuals),
                )[0, 1],
                2,
            )

            corr2_test_nonlin[i, j] = np.round(
                np.corrcoef(
                    np.transpose(dict_models["y_anchor_test"].values ** 2),
                    np.transpose(residuals),
                )[0, 1],
                2,
            )
            
            mi_test_nonlin[i, j] = np.round(
                mutual_info_regression(
                    dict_models["y_anchor_test"].values, residuals
                )[0],
                2,
            )


    """ Parameter optimization with Pareto for anchor regression """
#     if len(h_anchors) == 0:
    (
        ind_gamma_opt_lin,
        ind_lambda_opt_lin,
        ind_vect_ideal_obj1_lin,
        ind_vect_ideal_obj2_lin,
    ) = choose_gamma_lambda_pareto(
        rmse_bag_av_lin,
        corr_bag_av_lin,
        maxX=False,
        maxY=False,
    )
#     else:
    (
        ind_gamma_opt_nonlin,
        ind_lambda_opt_nonlin,
        ind_vect_ideal_obj1_nonlin,
        ind_vect_ideal_obj2_nonlin,
        ind_vect_ideal_obj3_nonlin,
    ) = choose_gamma_lambda_pareto_3(
        rmse_bag_av_nonlin,
        corr_bag_av_nonlin,
        corr2_bag_av_nonlin,
        maxX=False,
        maxY=False,
        maxZ=False,
    )

    coef_raw_opt_lin = coef_raw_bag_av_lin[ind_gamma_opt_lin, ind_lambda_opt_lin, :]
    coef_std_opt_lin = coef_std_bag_av_lin[ind_gamma_opt_lin, ind_lambda_opt_lin, :]

    coef_raw_opt_nonlin = coef_raw_bag_av_nonlin[ind_gamma_opt_nonlin, ind_lambda_opt_nonlin, :]
    coef_std_opt_nonlin = coef_std_bag_av_nonlin[ind_gamma_opt_nonlin, ind_lambda_opt_nonlin, :]

    """ Parameter optimization with Pareto for ridge regression """
#     if len(h_anchors) == 0:
    (
        _,
        ind_lambda_opt_ridge_lin,
        _,
        _,
    ) = choose_gamma_lambda_pareto(
        rmse_bag_av_lin[0,:].reshape(1, -1),
        corr_bag_av_lin[0,:].reshape(1, -1),
        maxX=False,
        maxY=False,
    )
#     else:
    (
        _,
        ind_lambda_opt_ridge_nonlin,
        _,
        _,
        _,
    ) = choose_gamma_lambda_pareto_3(
        rmse_bag_av_nonlin[0,:].reshape(1, -1),
        corr_bag_av_nonlin[0,:].reshape(1, -1),
        corr2_bag_av_nonlin[0,:].reshape(1, -1),
        maxX=False,
        maxY=False,
        maxZ=False,
    )

    coef_raw_opt_ridge_lin = coef_raw_bag_av_lin[0, ind_lambda_opt_ridge_lin, :]
    coef_std_opt_ridge_lin = coef_std_bag_av_lin[0, ind_lambda_opt_ridge_lin, :]
    
    coef_raw_opt_ridge_nonlin = coef_raw_bag_av_nonlin[0, ind_lambda_opt_ridge_nonlin, :]
    coef_std_opt_ridge_nonlin = coef_std_bag_av_nonlin[0, ind_lambda_opt_ridge_nonlin, :]
    
    return (
        rmse_bag_av_lin,
        corr_bag_av_lin,
        mi_bag_av_lin,
        coef_raw_opt_lin,
        coef_std_opt_lin,
        coef_raw_opt_ridge_lin,
        coef_std_opt_ridge_lin,
        rmse_test_lin,
        corr_test_lin,
        mi_test_lin,
        ind_gamma_opt_lin,
        ind_lambda_opt_lin,
        ind_lambda_opt_ridge_lin,
        ind_vect_ideal_obj1_lin,
        ind_vect_ideal_obj2_lin,
        rmse_bag_av_nonlin,
        corr_bag_av_nonlin,
        corr2_bag_av_nonlin,
        mi_bag_av_nonlin,
        coef_raw_opt_nonlin,
        coef_std_opt_nonlin,
        coef_raw_opt_ridge_nonlin,
        coef_std_opt_ridge_nonlin,
        rmse_test_nonlin,
        corr_test_nonlin,
        corr2_test_nonlin,
        mi_test_nonlin,
        ind_gamma_opt_nonlin,
        ind_lambda_opt_nonlin,
        ind_lambda_opt_ridge_nonlin,
        ind_vect_ideal_obj1_nonlin,
        ind_vect_ideal_obj2_nonlin,
        ind_vect_ideal_obj3_nonlin,
    )


def compute_distance(vect_ideal, vect_Nadir, pf_X, pf_Y):
    d = np.zeros([len(pf_X), 1])
    for i in range(len(pf_X)):
        d[i] = np.sqrt(
            0.5
            * ((pf_X[i] - vect_ideal[0]) / (vect_Nadir[0] - vect_ideal[0]))
            ** 2
            + 0.5
            * ((pf_Y[i] - vect_ideal[1]) / (vect_Nadir[1] - vect_ideal[1]))
            ** 2
        )

    return d


def compute_distance_3(vect_ideal, vect_Nadir, pf_X, pf_Y, pf_Z):
    d = np.zeros([len(pf_X), 1])
    for i in range(len(pf_X)):
        d[i] = np.sqrt(
            0.5
            * ((pf_X[i] - vect_ideal[0]) / (vect_Nadir[0] - vect_ideal[0]))
            ** 2
            + 0.25
            * ((pf_Y[i] - vect_ideal[1]) / (vect_Nadir[1] - vect_ideal[1]))
            ** 2
            + 0.25
            * ((pf_Z[i] - vect_ideal[2]) / (vect_Nadir[2] - vect_ideal[2]))
            ** 2
        )

    return d


def choose_gamma_lambda_pareto(
    Xs, Ys, maxX=True, maxY=True, plot=False, objective1=None, objective2=None
):
    """Pareto frontier selection process"""
    Xsa = np.abs(Xs)
    Xsv = Xsa.reshape(-1, 1)
    Ysa = np.abs(Ys)
    Ysv = Ysa.reshape(-1, 1)
    sorted_list = sorted(
        [[Xsv[i], Ysv[i]] for i in range(len(Xsv))], reverse=maxY
    )
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]

    vect_ideal_abs = np.zeros(2)
    vect_Nadir_abs = np.zeros(2)

    min_obj1 = min(Xsv)
    max_obj1 = max(Xsv)
    min_obj2 = min(Ysv)
    max_obj2 = max(Ysv)

    for i in range(Xs.shape[0]):  # nb gamma values
        for j in range(Xs.shape[1]):  # nb lambda values
            if Xsa[i, j] == min_obj1:
                ind_vect_ideal_obj1 = np.array([i, j])
                vect_ideal_abs[0] = Xsa[i, j]
            if Xsa[i, j] == max_obj1:
                #                 ind_vect_Nadir_obj1 = np.array([i, j])
                vect_Nadir_abs[0] = Xsa[i, j]
            if Ysa[i, j] == min_obj2:
                ind_vect_ideal_obj2 = np.array([i, j])
                vect_ideal_abs[1] = Ysa[i, j]
            if Ysa[i, j] == max_obj2:
                #                 ind_vect_Nadir_obj2 = np.array([i, j])
                vect_Nadir_abs[1] = Ysa[i, j]

    dst = compute_distance(vect_ideal_abs, vect_Nadir_abs, pf_X, pf_Y)
    ind = np.argmin(dst)

    #     """Plotting process"""
    #     if plot:
    #         plt.scatter(Xsv, Ysv)
    #         plt.plot(pf_X, pf_Y)
    #         plt.xlabel(objective1)
    #         plt.ylabel(objective2)
    #         plt.plot(ideal[0], ideal[1], "k*")
    #         plt.plot(pf_X[ind], pf_Y[ind], "ro")
    #         plt.show()

    for ind_opt_gamma in range(Xs.shape[0]):  # nb gamma values
        for ind_opt_lambda in range(Xs.shape[1]):  # nb lambda values
            if (Xsa[ind_opt_gamma, ind_opt_lambda] == pf_X[ind]) and (
                Ysa[ind_opt_gamma, ind_opt_lambda] == pf_Y[ind]
            ):
                return (
                    ind_opt_gamma,
                    ind_opt_lambda,
                    ind_vect_ideal_obj1,
                    ind_vect_ideal_obj2,
                )

            
def choose_gamma_lambda_pareto_3(
    Xs, Ys, Zs, maxX=True, maxY=True, maxZ=True, plot=False, objective1=None, objective2=None, objective3=None
):
    """Pareto frontier selection process"""
    Xsa = np.abs(Xs)
    Xsv = Xsa.reshape(-1, 1)
    Ysa = np.abs(Ys)
    Ysv = Ysa.reshape(-1, 1)
    Zsa = np.abs(Zs)
    Zsv = Zsa.reshape(-1, 1)
    
    sorted_list = sorted(
        [[Xsv[i], Ysv[i], Zsv[i]] for i in range(len(Xsv))], reverse=maxY
    )
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    pf_Z = [pair[2] for pair in pareto_front]

    vect_ideal_abs = np.zeros(3)
    vect_Nadir_abs = np.zeros(3)

    min_obj1 = min(Xsv)
    max_obj1 = max(Xsv)
    min_obj2 = min(Ysv)
    max_obj2 = max(Ysv)
    min_obj3 = min(Zsv)
    max_obj3 = max(Zsv)

    for i in range(Xs.shape[0]):  # nb gamma values
        for j in range(Xs.shape[1]):  # nb lambda values
            if Xsa[i, j] == min_obj1:
                ind_vect_ideal_obj1 = np.array([i, j])
                vect_ideal_abs[0] = Xsa[i, j]
            if Xsa[i, j] == max_obj1:
                #                 ind_vect_Nadir_obj1 = np.array([i, j])
                vect_Nadir_abs[0] = Xsa[i, j]
            if Ysa[i, j] == min_obj2:
                ind_vect_ideal_obj2 = np.array([i, j])
                vect_ideal_abs[1] = Ysa[i, j]
            if Ysa[i, j] == max_obj2:
                #                 ind_vect_Nadir_obj2 = np.array([i, j])
                vect_Nadir_abs[1] = Ysa[i, j]
            if Zsa[i, j] == min_obj3:
                ind_vect_ideal_obj3 = np.array([i, j])
                vect_ideal_abs[2] = Zsa[i, j]
            if Zsa[i, j] == max_obj3:
                #                 ind_vect_Nadir_obj2 = np.array([i, j])
                vect_Nadir_abs[2] = Zsa[i, j]

    dst = compute_distance_3(vect_ideal_abs, vect_Nadir_abs, pf_X, pf_Y, pf_Z)
    ind = np.argmin(dst)

    #     """Plotting process"""
    #     if plot:
    #         plt.scatter(Xsv, Ysv)
    #         plt.plot(pf_X, pf_Y)
    #         plt.xlabel(objective1)
    #         plt.ylabel(objective2)
    #         plt.plot(ideal[0], ideal[1], "k*")
    #         plt.plot(pf_X[ind], pf_Y[ind], "ro")
    #         plt.show()

    for ind_opt_gamma in range(Xs.shape[0]):  # nb gamma values
        for ind_opt_lambda in range(Xs.shape[1]):  # nb lambda values
            if (Xsa[ind_opt_gamma, ind_opt_lambda] == pf_X[ind]) and (
                Ysa[ind_opt_gamma, ind_opt_lambda] == pf_Y[ind]) and (
                Zsa[ind_opt_gamma, ind_opt_lambda] == pf_Z[ind]
            ):
                return (
                    ind_opt_gamma,
                    ind_opt_lambda,
                    ind_vect_ideal_obj1,
                    ind_vect_ideal_obj2,
                    ind_vect_ideal_obj3,
                )


def choose_gamma_h_lambda_pareto(
    Xs, Ys, gamma_vals, h_anchors, lambdavals, maxX=True, maxY=True
):
    """Pareto frontier selection process"""
    Xs = np.abs(Xs)
    Xsv = Xs.reshape(-1, 1)
    Ys = np.abs(Ys)
    Ysv = Ys.reshape(-1, 1)
    sorted_list = sorted(
        [[Xsv[i], Ysv[i]] for i in range(len(Xsv))], reverse=maxY
    )
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    """Plotting process"""
    plt.scatter(Xsv, Ysv)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")

    ideal = [min(Xsv), min(Ysv)]
    dst = helpers.compute_distance(ideal, pf_X, pf_Y)
    ind = np.argmin(dst)

    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[1]):
            for k in range(lambdavals.shape[0]):
                if (Xs[i, j, k] == pf_X[ind]) and (Ys[i, j, k] == pf_Y[ind]):
                    gammaSel = gamma_vals[i]
                    hSel = h_anchors[:j]
                    lambdaSel = lambdavals[k]

    plt.plot(ideal[0], ideal[1], "k*")
    plt.plot(pf_X[ind], pf_Y[ind], "ro")
    plt.show()

    return gammaSel, hSel, lambdaSel, pf_X, pf_Y


def param_optimization_gamma(params_climate, params_anchor):

    target = params_climate["target"]
    anchor = params_climate["anchor"]
    gamma = params_anchor["gamma"][0]
    h_anchors = params_anchor["h_anchors"]

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    dict_models = split_train_test(
        modelsDataList, modelsInfoFrame, target, anchor
    )

    cv_vals = 5

    (
        lambdaSelAll,
        mse_df,
        sem_CV,
        corr_pearson,
        mi,
    ) = cross_validation_anchor_regression(
        modelsDataList,
        modelsInfoFrame,
        deepcopy(dict_models),
        params_climate,
        gamma,
        h_anchors,
        cv_vals,
        display_CV_plot=False,
    )

    dirname = "./../output/data/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    filename = (
        dirname
        + "param_optimization_gamma_target_"
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
        + ".pkl"
    )
    with open(filename, "wb") as f:
        pickle.dump([mse_df, corr_pearson, mi, gamma, h_anchors], f)
