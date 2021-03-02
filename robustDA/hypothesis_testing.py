import numpy as np
import pandas as pd
import pickle
import os
import sys

from copy import deepcopy

from robustDA.process_cmip6 import read_files_cmip6, read_forcing_cmip6
from robustDA.processing import split_train_test
from robustDA.anchor_regression import (
    cross_validation_anchor_regression,
    anchor_regression_estimator,
)


def test_DA(params_climate, params_anchor):

    target = params_climate["target"]
    anchor = params_climate["anchor"]
    gamma = params_anchor["gamma"][0]
    h_anchors = params_anchor["h_anchors"]

    B = 2
    cv_vals = 3
    grid = (72, 144)

    mse_runs = np.zeros([B, cv_vals])
    corr_pearson_runs = np.zeros([B, cv_vals])
    mi_runs = np.zeros([B, cv_vals])
    coefRaw_runs = np.zeros([B, grid[0] * grid[1]])
    y_test_pred_runs = [[] for i in range(B)]
    lambdaSel_runs = np.zeros([B, 1])

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    nbFiles = len(modelsDataList)
    models = sorted(set([modelsDataList[i].model for i in range(nbFiles)]))

    alpha_bagging = np.zeros(len(models))
    power_bagging = np.zeros(len(models))
    nb_models_bagging = np.zeros(len(models))

    filename = "HT_" + target + "_" + anchor + "_gamma_" + str(gamma)

    sys.stdout = open("./../output/logFiles/" + filename + ".log", "w")

    for b in range(B):
        print("\n\n ============== Bag " + str(b) + " =================== \n")
        dict_models = split_train_test(
            modelsDataList, modelsInfoFrame, target, anchor
        )

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
            display_CV_plot=True,
        )

        coefRaw, y_test_pred, mse = anchor_regression_estimator(
            dict_models, gamma, h_anchors, lambdaSel[2]
        )

        lambdaSel_runs[b] = lambdaSel[2]
        mse_runs[b, :] = mse_df["MSE - TOTAL"].values
        corr_pearson_runs[b, :] = np.mean(corr_pearson, axis=1)
        mi_runs[b, :] = np.mean(mi, axis=1)
        coefRaw_runs[b, :] = coefRaw
        y_test_pred_runs[b].append(y_test_pred)

        alpha_per_bag, power_per_bag, nb_models_per_bag = test_DA_per_bag(
            params_climate, models, dict_models, y_test_pred
        )

        alpha_bagging = alpha_bagging + alpha_per_bag
        power_bagging = power_bagging + power_per_bag
        nb_models_bagging = nb_models_bagging + nb_models_per_bag

    alpha_bagging = np.array(
        [
            alpha_bagging[i] / nb_models_bagging[i]
            if nb_models_bagging[i] != 0
            else 0
            for i in range(len(models))
        ]
    )
    power_bagging = np.array(
        [
            power_bagging[i] / nb_models_bagging[i]
            if nb_models_bagging[i] != 0
            else 0
            for i in range(len(models))
        ]
    )

    mse_runs_df = pd.DataFrame(mse_runs)
    mse_runs_df["Lambda selected"] = pd.DataFrame(lambdaSel_runs)
    mse_runs_df.index.name = "Run"

    dirname = "./../output/data/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    with open(dirname + filename + ".pkl", "wb") as f:
        pickle.dump(
            [
                lambdaSel_runs,
                coefRaw_runs,
                mse_runs_df,
                corr_pearson_runs,
                mi_runs,
                y_test_pred_runs,
                alpha_bagging,
                power_bagging,
                nb_models_bagging,
                models,
            ],
            f,
        )

    sys.stdout.close()

    return alpha_bagging, power_bagging, nb_models_bagging, models


def test_DA_per_bag(params_climate, models, dict_models, y_test_pred):
    target = params_climate["target"]
    start_date = params_climate["startDate"]
    end_date = params_climate["endDate"]

    alpha_per_bag = np.zeros(len(models))
    power_per_bag = np.zeros(len(models))
    nb_models_per_bag = np.zeros(len(models))

    yf = read_forcing_cmip6("historical", target, start_date, end_date)
    nb_years = end_date - start_date + 1

    test_models = set(
        [
            dict_models["testFiles"][i].split("_")[2][:3]
            for i in range(len(dict_models["testFiles"]))
        ]
    )

    print(test_models)

    """ For each model in the test set, build the null from the rest of the models,
    compute the threshold, then compute \alpha and \beta """
    for i in range(len(models)):
        if models[i] in test_models:
            ts_null = []
            test_model_files = []
            test_model_ts_vals = []

            for j in range(len(dict_models["testFiles"])):
                #                 print(dict_models["testFiles"][j])
                ts = np.corrcoef(
                    np.transpose(
                        y_test_pred[j * nb_years : (j + 1) * nb_years]
                    ),
                    np.transpose(yf.values.reshape(-1, 1)),
                )[0, 1]

                if dict_models["testFiles"][j].split("_")[2][:3] != models[i]:
                    if (
                        dict_models["testFiles"][j].split("_")[3]
                        == "piControl"
                    ):
                        ts_null.append(ts)
                else:
                    test_model_files.append(dict_models["testFiles"][j])
                    test_model_ts_vals.append(ts)

            ts_null_mean = np.mean(ts_null)
            ts_null_std = np.std(ts_null)
            print(
                str(i)
                + " --- "
                + models[i]
                + " --- mean = "
                + str(ts_null_mean)
                + " --- std = "
                + str(ts_null_std)
            )

            alpha = 0
            power = 0
            nb_control_runs = 0
            nb_forced_runs = 0
            for k in range(len(test_model_files)):
                print(test_model_files[k])
                if (
                    test_model_ts_vals[k] > ts_null_mean + 2 * ts_null_std
                    or test_model_ts_vals[k] < ts_null_mean - 2 * ts_null_std
                ):
                    test_val = 1  # Reject H0
                else:
                    test_val = 0

                # extend for other forcings
                if test_model_files[k].split("_")[3] == "piControl":
                    nb_control_runs = nb_control_runs + 1
                    if test_val == 1:
                        alpha = alpha + 1
                elif test_model_files[k].split("_")[3] != "piControl":
                    nb_forced_runs = nb_forced_runs + 1
                    if test_val == 1:
                        power = power + 1

                print(
                    test_model_files[k].split("_")[3]
                    + " --- test = "
                    + str(test_val)
                )

            print("nb_controls = " + str(nb_control_runs))
            print("nb_forced = " + str(nb_forced_runs))
            print("alpha = " + str(alpha))
            print("power = " + str(power))

            if nb_control_runs != 0:
                alpha_per_bag[i] = alpha / nb_control_runs
            if nb_forced_runs != 0:
                power_per_bag[i] = power / nb_forced_runs
            nb_models_per_bag[i] = 1

    print(alpha_per_bag)
    print(power_per_bag)

    return alpha_per_bag, power_per_bag, nb_models_per_bag
