import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

from copy import deepcopy
# from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

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

    B = 50
    cv_vals = 50
    lambda_vals = np.logspace(0, 9, cv_vals)
    grid = (72, 144)
    p = grid[0] * grid[1]

    rmse_train_lin = np.zeros([B, len(gamma_vals), cv_vals])
    corr_train_lin = np.zeros([B, len(gamma_vals), cv_vals])
    mi_train_lin = np.zeros([B, len(gamma_vals), cv_vals])

    rmse_test_lin = np.zeros([B, len(gamma_vals), cv_vals])
    corr_test_lin = np.zeros([B, len(gamma_vals), cv_vals])
    mi_test_lin = np.zeros([B, len(gamma_vals), cv_vals])

    coef_raw_opt_lin = np.zeros([B, p])
    coef_raw_opt_ridge_lin = np.zeros([B, p])
    y_test_true = [[] for i in range(B)]
    y_test_pred_lin = [[] for i in range(B)]
    y_test_pred_ridge_lin = [[] for i in range(B)]
    y_anchor_test = [[] for i in range(B)]

    ind_gamma_opt_lin = np.zeros([B, 1])
    ind_lambda_opt_lin = np.zeros([B, 1])
    ind_lambda_opt_ridge_lin = np.zeros([B, 1])
    ind_vect_ideal_obj1_lin = np.zeros([B, 2])
    ind_vect_ideal_obj2_lin = np.zeros([B, 2])
    gamma_opt_lin = np.zeros([B, 1])
    lambda_opt_lin = np.zeros([B, 1])
    lambda_opt_ridge_lin = np.zeros([B, 1])

    rmse_train_nonlin = np.zeros([B, len(gamma_vals), cv_vals])
    corr_train_nonlin = np.zeros([B, len(gamma_vals), cv_vals])
    corr2_train_nonlin = np.zeros([B, len(gamma_vals), cv_vals])
    mi_train_nonlin = np.zeros([B, len(gamma_vals), cv_vals])

    rmse_test_nonlin = np.zeros([B, len(gamma_vals), cv_vals])
    corr_test_nonlin = np.zeros([B, len(gamma_vals), cv_vals])
    corr2_test_nonlin = np.zeros([B, len(gamma_vals), cv_vals])
    mi_test_nonlin = np.zeros([B, len(gamma_vals), cv_vals])

    coef_raw_opt_nonlin = np.zeros([B, p])
    coef_raw_opt_ridge_nonlin = np.zeros([B, p])
    y_test_pred_nonlin = [[] for i in range(B)]
    y_test_pred_ridge_nonlin = [[] for i in range(B)]

    ind_gamma_opt_nonlin = np.zeros([B, 1])
    ind_lambda_opt_nonlin = np.zeros([B, 1])
    ind_lambda_opt_ridge_nonlin = np.zeros([B, 1])
    ind_vect_ideal_obj1_nonlin = np.zeros([B, 2])
    ind_vect_ideal_obj2_nonlin = np.zeros([B, 2])
    ind_vect_ideal_obj3_nonlin = np.zeros([B, 2])
    gamma_opt_nonlin = np.zeros([B, 1])
    lambda_opt_nonlin = np.zeros([B, 1])
    lambda_opt_ridge_nonlin = np.zeros([B, 1])

    modelsDataList, modelsInfoFrame = read_files_cmip6(
        params_climate, norm=True
    )

    nbFiles = len(modelsDataList)
    models = sorted(set([modelsDataList[i].model for i in range(nbFiles)]))

    alpha_bagging_lin = np.zeros(len(models))
    power_bagging_lin = np.zeros(len(models))
    nb_models_bagging = np.zeros(len(models))

    alpha_bagging_nonlin = np.zeros(len(models))
    power_bagging_nonlin = np.zeros(len(models))

    if len(h_anchors) == 0:
        filename = "HT_" + target + "_" + anchor + "_B" + str(B) + "_CV3"
    else:
        filename = (
            "HT_"
            + target
            + "_"
            + anchor
            + "_"
            + "-".join(h_anchors)
            + "_B"
            + str(B)
            + "_CV3"
            + "_spearman95_coefRaw"
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
            rmse_train_bag_lin,
            corr_train_bag_lin,
            mi_train_bag_lin,
            coef_raw_bag_opt_lin,
            coef_std_bag_opt_lin,
            coef_raw_bag_opt_ridge_lin,
            coef_std_bag_opt_ridge_lin,
            rmse_test_bag_lin,
            corr_test_bag_lin,
            mi_test_bag_lin,
            ind_gamma_bag_opt_lin,
            ind_lambda_bag_opt_lin,
            ind_lambda_bag_opt_ridge_lin,
            ind_vect_ideal_obj1_bag_lin,
            ind_vect_ideal_obj2_bag_lin,
            rmse_train_bag_nonlin,
            corr_train_bag_nonlin,
            corr2_train_bag_nonlin,
            mi_train_bag_nonlin,
            coef_raw_bag_opt_nonlin,
            coef_std_bag_opt_nonlin,
            coef_raw_bag_opt_ridge_nonlin,
            coef_std_bag_opt_ridge_nonlin,
            rmse_test_bag_nonlin,
            corr_test_bag_nonlin,
            corr2_test_bag_nonlin,
            mi_test_bag_nonlin,
            ind_gamma_bag_opt_nonlin,
            ind_lambda_bag_opt_nonlin,
            ind_lambda_bag_opt_ridge_nonlin,
            ind_vect_ideal_obj1_bag_nonlin,
            ind_vect_ideal_obj2_bag_nonlin,
            ind_vect_ideal_obj3_bag_nonlin,
        ) = cross_validation_gamma_lambda(
            modelsDataList,
            modelsInfoFrame,
            deepcopy(dict_models),
            params_climate,
            gamma_vals,
            lambda_vals,
            h_anchors,
            display_CV_plot=True,
        )

        #         sc_y_train = StandardScaler(with_mean=True, with_std=True)
        #         y_train_std = sc_y_train.fit_transform(dict_models["y_train"].values)

        #         sc_X_test = StandardScaler(with_mean=True, with_std=True)
        #         X_test_std = sc_X_test.fit_transform(dict_models["X_test"].values)

        y_test_true_bag = dict_models["y_test"].values

        y_test_true[b].append(y_test_true_bag)
        y_anchor_test[b].append(
            dict_models["y_anchor_test"].values
        )  # do not change the anchor

        """ Linear anchor """
        #         y_test_pred_std = np.array(
        #             np.matmul(X_test_std, np.transpose(coef_std_bag_opt_lin))
        #         )
        #         y_test_pred_bag = sc_y_train.inverse_transform(y_test_pred_std)

        y_test_pred_bag = np.array(
            np.matmul(
                dict_models["X_test"].values,
                np.transpose(coef_raw_bag_opt_lin),
            )
        )

        #         y_test_pred_std_ridge = np.array(
        #             np.matmul(X_test_std, np.transpose(coef_std_bag_opt_ridge_lin))
        #         )
        #         y_test_pred_bag_ridge = sc_y_train.inverse_transform(y_test_pred_std_ridge)

        y_test_pred_bag_ridge = np.array(
            np.matmul(
                dict_models["X_test"].values,
                np.transpose(coef_raw_bag_opt_ridge_lin),
            )
        )

        rmse_train_lin[b, :, :] = rmse_train_bag_lin
        corr_train_lin[b, :, :] = corr_train_bag_lin
        mi_train_lin[b, :, :] = mi_train_bag_lin

        rmse_test_lin[b, :, :] = rmse_test_bag_lin
        corr_test_lin[b, :, :] = corr_test_bag_lin
        mi_test_lin[b, :, :] = mi_test_bag_lin

        coef_raw_opt_lin[b, :] = coef_raw_bag_opt_lin
        coef_raw_opt_ridge_lin[b, :] = coef_raw_bag_opt_ridge_lin

        y_test_pred_lin[b].append(y_test_pred_bag)
        y_test_pred_ridge_lin[b].append(y_test_pred_bag_ridge)

        ind_gamma_opt_lin[b] = ind_gamma_bag_opt_lin
        ind_lambda_opt_lin[b] = ind_lambda_bag_opt_lin
        ind_lambda_opt_ridge_lin[b] = ind_lambda_bag_opt_ridge_lin
        gamma_opt_lin[b] = gamma_vals[ind_gamma_bag_opt_lin]
        lambda_opt_lin[b] = lambda_vals[ind_lambda_bag_opt_lin]
        lambda_opt_ridge_lin[b] = lambda_vals[ind_lambda_bag_opt_ridge_lin]

        ind_vect_ideal_obj1_lin[b] = ind_vect_ideal_obj1_bag_lin
        ind_vect_ideal_obj2_lin[b] = ind_vect_ideal_obj2_bag_lin

        (
            alpha_per_bag_lin,
            power_per_bag_lin,
            nb_models_per_bag,
        ) = test_DA_per_bag(
            params_climate, models, dict_models, y_test_pred_bag
        )

        alpha_bagging_lin = alpha_bagging_lin + alpha_per_bag_lin
        power_bagging_lin = power_bagging_lin + power_per_bag_lin
        nb_models_bagging = nb_models_bagging + nb_models_per_bag

        """ Nonlinear anchor """
        #         y_test_pred_std = np.array(
        #             np.matmul(X_test_std, np.transpose(coef_std_bag_opt_nonlin))
        #         )
        #         y_test_pred_bag = sc_y_train.inverse_transform(y_test_pred_std)

        y_test_pred_bag = np.array(
            np.matmul(
                dict_models["X_test"].values,
                np.transpose(coef_raw_bag_opt_nonlin),
            )
        )

        #         y_test_pred_std_ridge = np.array(
        #             np.matmul(X_test_std, np.transpose(coef_std_bag_opt_ridge_nonlin))
        #         )
        #         y_test_pred_bag_ridge = sc_y_train.inverse_transform(y_test_pred_std_ridge)

        y_test_pred_bag_ridge = np.array(
            np.matmul(
                dict_models["X_test"].values,
                np.transpose(coef_raw_bag_opt_ridge_nonlin),
            )
        )

        rmse_train_nonlin[b, :, :] = rmse_train_bag_nonlin
        corr_train_nonlin[b, :, :] = corr_train_bag_nonlin
        corr2_train_nonlin[b, :, :] = corr2_train_bag_nonlin
        mi_train_nonlin[b, :, :] = mi_train_bag_nonlin

        rmse_test_nonlin[b, :, :] = rmse_test_bag_nonlin
        corr_test_nonlin[b, :, :] = corr_test_bag_nonlin
        corr2_test_nonlin[b, :, :] = corr2_test_bag_nonlin
        mi_test_nonlin[b, :, :] = mi_test_bag_nonlin

        coef_raw_opt_nonlin[b, :] = coef_raw_bag_opt_nonlin
        coef_raw_opt_ridge_nonlin[b, :] = coef_raw_bag_opt_ridge_nonlin

        y_test_pred_nonlin[b].append(y_test_pred_bag)
        y_test_pred_ridge_nonlin[b].append(y_test_pred_bag_ridge)

        ind_gamma_opt_nonlin[b] = ind_gamma_bag_opt_nonlin
        ind_lambda_opt_nonlin[b] = ind_lambda_bag_opt_nonlin
        ind_lambda_opt_ridge_nonlin[b] = ind_lambda_bag_opt_ridge_nonlin
        gamma_opt_nonlin[b] = gamma_vals[ind_gamma_bag_opt_nonlin]
        lambda_opt_nonlin[b] = lambda_vals[ind_lambda_bag_opt_nonlin]
        lambda_opt_ridge_nonlin[b] = lambda_vals[
            ind_lambda_bag_opt_ridge_nonlin
        ]

        ind_vect_ideal_obj1_nonlin[b] = ind_vect_ideal_obj1_bag_nonlin
        ind_vect_ideal_obj2_nonlin[b] = ind_vect_ideal_obj2_bag_nonlin
        ind_vect_ideal_obj3_nonlin[b] = ind_vect_ideal_obj3_bag_nonlin

        (
            alpha_per_bag_nonlin,
            power_per_bag_nonlin,
            nb_models_per_bag,
        ) = test_DA_per_bag(
            params_climate, models, dict_models, y_test_pred_bag
        )

        alpha_bagging_nonlin = alpha_bagging_nonlin + alpha_per_bag_nonlin
        power_bagging_nonlin = power_bagging_nonlin + power_per_bag_nonlin

    alpha_bagging_lin = np.array(
        [
            alpha_bagging_lin[i] / nb_models_bagging[i]
            if nb_models_bagging[i] != 0
            else 0
            for i in range(len(models))
        ]
    )
    power_bagging_lin = np.array(
        [
            power_bagging_lin[i] / nb_models_bagging[i]
            if nb_models_bagging[i] != 0
            else 0
            for i in range(len(models))
        ]
    )

    alpha_bagging_nonlin = np.array(
        [
            alpha_bagging_nonlin[i] / nb_models_bagging[i]
            if nb_models_bagging[i] != 0
            else 0
            for i in range(len(models))
        ]
    )
    power_bagging_nonlin = np.array(
        [
            power_bagging_nonlin[i] / nb_models_bagging[i]
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
                params_climate,
                params_anchor,
                lambda_vals,
                y_test_true,
                y_anchor_test,
                ind_gamma_opt_lin,
                ind_lambda_opt_lin,
                ind_lambda_opt_ridge_lin,
                gamma_opt_lin,
                lambda_opt_lin,
                lambda_opt_ridge_lin,
                ind_vect_ideal_obj1_lin,
                ind_vect_ideal_obj2_lin,
                coef_raw_opt_lin,
                coef_raw_opt_ridge_lin,
                y_test_pred_lin,
                y_test_pred_ridge_lin,
                rmse_train_lin,
                corr_train_lin,
                mi_train_lin,
                rmse_test_lin,
                corr_test_lin,
                mi_test_lin,
                alpha_bagging_lin,
                power_bagging_lin,
                ind_gamma_opt_nonlin,
                ind_lambda_opt_nonlin,
                ind_lambda_opt_ridge_nonlin,
                gamma_opt_nonlin,
                lambda_opt_nonlin,
                lambda_opt_ridge_nonlin,
                ind_vect_ideal_obj1_nonlin,
                ind_vect_ideal_obj2_nonlin,
                ind_vect_ideal_obj3_nonlin,
                coef_raw_opt_nonlin,
                coef_raw_opt_ridge_nonlin,
                y_test_pred_nonlin,
                y_test_pred_ridge_nonlin,
                rmse_train_nonlin,
                corr_train_nonlin,
                corr2_train_nonlin,
                mi_train_nonlin,
                rmse_test_nonlin,
                corr_test_nonlin,
                corr2_test_nonlin,
                mi_test_nonlin,
                alpha_bagging_nonlin,
                power_bagging_nonlin,
                nb_models_bagging,
                models,
            ],
            f,
        )

    sys.stdout.close()


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
                #                 ''' Linear correlation (Pearson) '''
                #                 ts = np.corrcoef(
                #                     np.transpose(
                #                         y_test_pred[j * nb_years : (j + 1) * nb_years]
                #                     ),
                #                     np.transpose(yf.values.reshape(-1, 1)),
                #                 )[0, 1]

                """ Rank correlation (Spearman) """
                ts = spearmanr(
                    y_test_pred[j * nb_years : (j + 1) * nb_years], yf.values
                ).correlation

                #                 ''' First differences -- rank correlation '''
                #                 diff_pred = np.diff(y_test_pred[j * nb_years : (j + 1) * nb_years])
                #                 diff_true = yf.diff().values[1:]
                #                 ts = spearmanr(diff_pred, diff_true).correlation

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
            ts_null_std = np.std(
                ts_null
            )  # for one control run per model, otherwise also need to compute the variance
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
                    test_model_ts_vals[k]
                    > ts_null_mean
                    + 1.96 * ts_null_std  # 1.96 (95%) or 2.326 (98%)
                    or test_model_ts_vals[k]
                    < ts_null_mean - 1.96 * ts_null_std
                ):
                    test_val = 1  # Reject H0
                else:
                    test_val = 0

                # TO DO: extend for other forcings
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
