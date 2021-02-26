import numpy as np
import pickle
import os

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

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    B = 20

    nbFiles = len(modelsDataList)
    models = sorted(set([modelsDataList[i].model for i in range(nbFiles)]))

    alpha_bagging = np.zeros(len(models))
    power_bagging = np.zeros(len(models))
    nb_models_bagging = np.zeros(len(models))

    for b in range(B):
        print("Bag " + str(b))
        dict_models = split_train_test(
            modelsDataList, modelsInfoFrame, target, anchor
        )

        #         print(dict_models["trainModels"])
        #         print(dict_models["testModels"])

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
            30,
            sel_method="MSE",
            display_CV_plot=True,
        )

        coefRaw, y_test_pred, mse = anchor_regression_estimator(
            dict_models, gamma, h_anchors, lambdaSel[2]
        )

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

    dirname = "./../output/data/"
    filename = "./../output/data/HT_aerosols_co2_gamma1000.pkl"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(filename, "wb") as f:
        pickle.dump(
            [alpha_bagging, power_bagging, nb_models_bagging, models], f
        )

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
