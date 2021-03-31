import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from robustDA.process_cmip6 import read_files_cmip6, read_forcing_cmip6
from robustDA.processing import split_train_test
from robustDA.anchor_regression import (
    cross_validation_gamma_lambda,
)


def test_DA(params_climate, params_anchor):

    target = params_climate["target"]
    anchor = params_climate["anchor"]
    gamma_vals = params_anchor["gamma"]
    h_anchors = params_anchor["h_anchors"]

    B = 1
    cv_vals = 10
    grid = (72, 144)
    p = grid[0] * grid[1]
    lambda_vals = np.logspace(-2, 5, cv_vals)

    rmse = np.zeros([B, len(gamma_vals), cv_vals])
    corr = np.zeros([B, len(gamma_vals), cv_vals])
    mi = np.zeros([B, len(gamma_vals), cv_vals])
    coef_raw_opt = np.zeros([B, p])
    y_test_true = [[] for i in range(B)]
    y_test_pred = [[] for i in range(B)]
    y_anchor_test = [[] for i in range(B)]
    ind_gamma_opt = np.zeros([B, 1])
    ind_lambda_opt = np.zeros([B, 1])
    ind_vect_ideal_obj1 = np.zeros([B, 2])
    ind_vect_ideal_obj2 = np.zeros([B, 2])
    gamma_opt = np.zeros([B, 1])
    lambda_opt = np.zeros([B, 1])

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    nbFiles = len(modelsDataList)
    models = sorted(set([modelsDataList[i].model for i in range(nbFiles)]))

    alpha_bagging = np.zeros(len(models))
    power_bagging = np.zeros(len(models))
    nb_models_bagging = np.zeros(len(models))

    if len(h_anchors) == 0:
        filename = "HT_" + target + "_" + anchor + "_K_3"
    else:
        filename = (
            "HT_" + target + "_" + anchor + "_" + "-".join(h_anchors) + "_K_3"
        )

    sys.stdout = open("./../output/logFiles/" + filename + ".log", "w")

    for b in tqdm(range(B)):
        print(
            "\n\n ============== Bag " + str(b) + " =================== \n",
            flush=True,
        )

        dict_models = split_train_test(
            modelsDataList, modelsInfoFrame, target, anchor
        )

        (
            rmse_bag,
            corr_bag,
            mi_bag,
            coef_raw_bag_opt,
            ind_gamma_bag_opt,
            ind_lambda_bag_opt,
            ind_vect_ideal_obj1_bag,
            ind_vect_ideal_obj2_bag,
        ) = cross_validation_gamma_lambda(
            modelsDataList,
            modelsInfoFrame,
            deepcopy(dict_models),
            params_climate,
            gamma_vals,
            h_anchors,
            lambda_vals,
            display_CV_plot=True,
        )

        # Standardized values for the target (true and pred)
        y_test_pred_bag = np.array(
            np.matmul(dict_models["X_test"], coef_raw_bag_opt)
        )

        y_test_true_bag = StandardScaler().fit_transform(dict_models["y_test"])

        rmse[b, :, :] = rmse_bag
        corr[b, :, :] = corr_bag
        mi[b, :, :] = mi_bag
        coef_raw_opt[b, :] = coef_raw_bag_opt
        y_test_true[b].append(y_test_true_bag)
        y_test_pred[b].append(y_test_pred_bag)
        y_anchor_test[b].append(
            dict_models["y_anchor_test"].values.reshape(-1)
        )
        ind_gamma_opt[b] = ind_gamma_bag_opt
        ind_lambda_opt[b] = ind_lambda_bag_opt
        gamma_opt[b] = gamma_vals[ind_gamma_bag_opt]
        lambda_opt[b] = lambda_vals[ind_lambda_bag_opt]
        ind_vect_ideal_obj1[b] = ind_vect_ideal_obj1_bag
        ind_vect_ideal_obj2[b] = ind_vect_ideal_obj2_bag

        alpha_per_bag, power_per_bag, nb_models_per_bag = test_DA_per_bag(
            params_climate, models, dict_models, y_test_pred_bag
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
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    with open(dirname + filename + ".pkl", "wb") as f:
        pickle.dump(
            [
                h_anchors,
                gamma_vals,
                lambda_vals,
                ind_gamma_opt,
                ind_lambda_opt,
                gamma_opt,
                lambda_opt,
                ind_vect_ideal_obj1,
                ind_vect_ideal_obj2,
                coef_raw_opt,
                y_test_true,
                y_test_pred,
                y_anchor_test,
                rmse,
                corr,
                mi,
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

    print(test_models, flush=True)

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
