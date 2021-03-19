#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from matplotlib import gridspec

from robustDA.utils.helpers import truncate, display_nonlinear_anchors

params = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)


def plotMapCartopy(dataMap, cLim=None, title=None, filename=None):
    fig = plt.figure(figsize=(12, 4))
    logs = np.arange(0, 360, 2.5)
    lats = np.arange(-90, 90, 2.5)

    ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
    ax.coastlines()
    ax.gridlines()

    if cLim is None:
        cLim = max(abs(dataMap.flatten()))

    h = ax.pcolormesh(
        logs,
        lats,
        dataMap,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-cLim,
        vmax=cLim,
    )

    cbar_pos = [0.364, 0, 0.295, 0.03]  # [left,bottom,width,height]
    cbar_axes = fig.add_axes(cbar_pos)
    cbar = fig.colorbar(
        h, cax=cbar_axes, orientation="horizontal", drawedges=False
    )
    cbar.set_label("Temperature", fontsize=12)
    cbar.ax.locator_params(nbins=3)
    cbar.ax.tick_params(labelsize=params["xtick.labelsize"])

    if title is not None:
        fig.suptitle(title, fontsize=params["axes.titlesize"])

    if filename is not None:
        fig.savefig(
            "./../output/figures/tests/" + filename,
            dpi=2000,
            bbox_inches="tight",
        )


def plotMapCartopy_subplots(fig, ax, dataMap, cLim=None, title_subplot=None):
    logs = np.arange(0, 360, 2.5)
    lats = np.arange(-90, 90, 2.5)

    ax.coastlines()
    ax.gridlines()

    if cLim is None:
        cLim = max(abs(dataMap.flatten()))

    h = ax.pcolormesh(
        logs,
        lats,
        dataMap,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-cLim,
        vmax=cLim,
    )

    cbar_pos = [0.18, 0.05, 0.159, 0.03]  # [left,bottom,width,height]
    cbar_axes = fig.add_axes(cbar_pos)
    cbar = fig.colorbar(
        h,
        cax=cbar_axes,
        orientation="horizontal",
    )
    cbar.ax.locator_params(nbins=3)
    cbar.ax.tick_params(labelsize=params["xtick.labelsize"])
    if title_subplot is not None:
        ax.set_title(title_subplot, fontsize=params["axes.titlesize"])


def make_plots(
    dict_models,
    coefRaw,
    y_test_pred,
    A,
    gamma,
    target,
    anchor,
    h_anchors,
    filename=None,
):

    y_test_true = dict_models["y_test"].values

    # Standardize ?
    y_anchor_test = dict_models["y_anchor_test"]

    grid = (72, 144)

    fig = plt.figure(figsize=(18, 3.2))
    spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1.3, 1, 1])

    ax = fig.add_subplot(
        spec[0], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients ($\\gamma$ = " + str(gamma) + ")"
    plotMapCartopy_subplots(
        fig, ax, coefRaw.reshape(grid), cLim=None, title_subplot=title
    )

    ax = fig.add_subplot(spec[1])
    ax.plot(y_test_true, y_test_pred, ".", color="darkseagreen", markersize=5)
    ax.set_xlabel("True target")
    ax.set_ylabel("Predicted target")

    x = np.linspace(
        truncate(min(y_test_true), 2), truncate(max(y_test_true), 2)
    )
    ax.plot(x, x, "k", linestyle="solid")

    rmse = np.sqrt(np.mean((y_test_true - y_test_pred) ** 2))
    r2 = r2_score(y_test_true, y_test_pred)
    ax.set_title(
        "RMSE = "
        + str(np.round(rmse, 2))
        + " --- R2 = "
        + str(np.round(r2, 2)),
    )

    residuals = (y_test_true - y_test_pred).reshape(-1)

    ax = fig.add_subplot(spec[2])
    ax.plot(y_anchor_test, residuals, ".", color="peru", markersize=5)
    ax.set_xlabel("Forcing " + anchor.upper())
    ax.set_ylabel("Residuals " + target.upper())

    if len(h_anchors) == 0:
        corr_pearson = np.round(
            np.corrcoef(np.transpose(y_anchor_test), np.transpose(residuals))[
                0, 1
            ],
            2,
        )
        mi = np.round(mutual_info_regression(y_anchor_test, residuals)[0], 2)
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(corr_pearson)
            + ", $I$ = "
            + str(mi),
        )
    elif len(h_anchors) == 1:
        corr_pearson = np.round(
            np.corrcoef(np.transpose(y_anchor_test), np.transpose(residuals))[
                0, 1
            ],
            2,
        )
        corr_pearson_2 = np.round(
            np.corrcoef(
                np.transpose(y_anchor_test) ** 2, np.transpose(residuals)
            )[0, 1],
            2,
        )
        mi = np.round(mutual_info_regression(y_anchor_test, residuals)[0], 2)
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(corr_pearson)
            + "\n $\\rho ($"
            + display_nonlinear_anchors(h_anchors[0])
            + "$)$ = "
            + str(corr_pearson_2)
            + ", $I$ = "
            + str(mi),
        )
    elif len(h_anchors) == 2:
        corr_pearson = np.round(
            np.corrcoef(np.transpose(y_anchor_test), np.transpose(residuals))[
                0, 1
            ],
            2,
        )
        corr_pearson_2 = np.round(
            np.corrcoef(
                np.transpose(y_anchor_test) ** 2, np.transpose(residuals)
            )[0, 1],
            2,
        )
        corr_pearson_3 = np.round(
            np.corrcoef(
                np.abs(np.transpose(y_anchor_test)), np.transpose(residuals)
            )[0, 1],
            2,
        )
        mi = np.round(mutual_info_regression(y_anchor_test, residuals)[0], 2)
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(corr_pearson)
            + "\n $\\rho ($"
            + display_nonlinear_anchors(h_anchors[0])
            + "$)$ = "
            + str(corr_pearson_2)
            + ", $\\rho ($"
            + display_nonlinear_anchors(h_anchors[1])
            + "$)$ = "
            + str(corr_pearson_3)
            + ", $I$ = "
            + str(mi),
        )

    plt.subplots_adjust(wspace=0.2)

    if filename is not None:
        fig.savefig(
            "./../output/figures/" + filename,
            bbox_inches="tight",
        )


#     plt.close()


def make_plots_HT(
    dict_models,
    coefRaw,
    y_test_true,
    y_test_pred,
    y_anchor_test,
    target,
    anchor,
    h_anchors,
    filename=None,
):

    grid = (72, 144)

    fig = plt.figure(figsize=(18, 3.2))
    spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1.3, 1, 1])

    ax = fig.add_subplot(
        spec[0], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients"
    plotMapCartopy_subplots(
        fig, ax, coefRaw.reshape(grid), cLim=None, title_subplot=title
    )

    ax = fig.add_subplot(spec[1])
    ax.plot(y_test_true, y_test_pred, ".", color="darkseagreen", markersize=5)
    ax.set_xlabel("True target")
    ax.set_ylabel("Predicted target")

    x = np.linspace(
        truncate(min(y_test_true), 2), truncate(max(y_test_true), 2)
    )
    ax.plot(x, x, "k", linestyle="solid")

    rmse = np.sqrt(np.mean((y_test_true - y_test_pred) ** 2))
    r2 = r2_score(y_test_true, y_test_pred)
    ax.set_title(
        "RMSE = "
        + str(np.round(rmse, 2))
        + " --- R2 = "
        + str(np.round(r2, 2)),
    )

    residuals = (y_test_true - y_test_pred).reshape(-1)

    ax = fig.add_subplot(spec[2])
    ax.plot(y_anchor_test, residuals, ".", color="peru", markersize=5)
    ax.set_xlabel("Forcing " + anchor.upper())
    ax.set_ylabel("Residuals " + target.upper())

    if len(h_anchors) == 0:
        corr_pearson = np.round(
            np.corrcoef(np.transpose(y_anchor_test), np.transpose(residuals))[
                0, 1
            ],
            2,
        )
        mi = np.round(mutual_info_regression(y_anchor_test, residuals)[0], 2)
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(corr_pearson)
            + ", $I$ = "
            + str(mi),
        )
    elif len(h_anchors) == 1:
        corr_pearson = np.round(
            np.corrcoef(np.transpose(y_anchor_test), np.transpose(residuals))[
                0, 1
            ],
            2,
        )
        corr_pearson_2 = np.round(
            np.corrcoef(
                np.transpose(y_anchor_test) ** 2, np.transpose(residuals)
            )[0, 1],
            2,
        )
        mi = np.round(mutual_info_regression(y_anchor_test, residuals)[0], 2)
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(corr_pearson)
            + "\n $\\rho ($"
            + display_nonlinear_anchors(h_anchors[0])
            + "$)$ = "
            + str(corr_pearson_2)
            + ", $I$ = "
            + str(mi),
        )
    elif len(h_anchors) == 2:
        corr_pearson = np.round(
            np.corrcoef(np.transpose(y_anchor_test), np.transpose(residuals))[
                0, 1
            ],
            2,
        )
        corr_pearson_2 = np.round(
            np.corrcoef(
                np.transpose(y_anchor_test) ** 2, np.transpose(residuals)
            )[0, 1],
            2,
        )
        corr_pearson_3 = np.round(
            np.corrcoef(
                np.abs(np.transpose(y_anchor_test)), np.transpose(residuals)
            )[0, 1],
            2,
        )
        mi = np.round(mutual_info_regression(y_anchor_test, residuals)[0], 2)
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(corr_pearson)
            + "\n $\\rho ($"
            + display_nonlinear_anchors(h_anchors[0])
            + "$)$ = "
            + str(corr_pearson_2)
            + ", $\\rho ($"
            + display_nonlinear_anchors(h_anchors[1])
            + "$)$ = "
            + str(corr_pearson_3)
            + ", $I$ = "
            + str(mi),
        )

    plt.subplots_adjust(wspace=0.2)

    if filename is not None:
        fig.savefig(
            "./../output/figures/" + filename,
            bbox_inches="tight",
        )


#     plt.close()


def plot_CV_multipleMSE(mse_df, lambdasSelAll, filename, folds):
    nbStd = np.array([5, 10])
    clr = ["r", "b"]

    fig = plt.figure(figsize=(7, 4))
    # suptitle = "Cross validation (anchor regression γ = "
    # + str(gamma) + "): " + variables[0].upper() + " -- "
    # + target.upper() + " forcing [" + ', '.join(scenarios) + "] (" + \
    #             str(startDate) + " - " + str(endDate) + ") \n"
    # fig.suptitle(suptitle, fontsize = 18)

    for i in range(mse_df.shape[1] - 1):
        plt.plot(mse_df.index, mse_df.iloc[:, i], label="_nolegend_")
    #         ax1.plot(mse_df.index, mse_df.iloc[:, i], label=folds[i])

    plt.plot(mse_df.index, mse_df.iloc[:, i + 1], "k.-")

    plt.axvline(
        lambdasSelAll[0],
        ls="--",
        color="k",
        label="$\\lambda_{opt}$ = " + str(np.round(lambdasSelAll[0], 2)),
    )

    for j in range(len(lambdasSelAll) - 1):
        plt.axvline(
            lambdasSelAll[j + 1],
            color=clr[j],
            label="$\\lambda_{"
            + str(nbStd[j])
            + " MSE} $ = "
            + str(np.round(lambdasSelAll[j + 1], 2)),
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
    plt.xlabel("$\\lambda$", fontsize=14)
    plt.legend(fontsize=10)

    if not os.path.isdir("./../output/figures/"):
        os.makedirs("./../output/figures/")
    fig.savefig("./../output/figures/" + filename, bbox_inches="tight")

    plt.close()


def plot_CV_sem(mse_df, lambdasSelAll, sem_CV, filename, folds):
    nbStd = np.array([1, 2, 3])
    clr = ["r", "b", "k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    # suptitle = "Cross validation (anchor regression γ = "
    # + str(gamma) + "): " + variables[0].upper() + " -- "
    # + target.upper() + " forcing [" + ', '.join(scenarios) + "] (" + \
    #             str(startDate) + " - " + str(endDate) + ") \n"
    # fig.suptitle(suptitle, fontsize = 18)

    for i in range(mse_df.shape[1] - 1):
        ax1.plot(mse_df.index, mse_df.iloc[:, i], label="_nolegend_")
    #         ax1.plot(mse_df.index, mse_df.iloc[:, i], label=folds[i])

    ax1.plot(mse_df.index, mse_df.iloc[:, i + 1], "k.-")

    ax1.axvline(
        lambdasSelAll[0],
        ls="--",
        color="k",
        label="$\\lambda_{opt}$ = " + str(np.round(lambdasSelAll[0], 2)),
    )

    for j in range(len(lambdasSelAll) - 1):
        ax1.axvline(
            lambdasSelAll[j + 1],
            color=clr[j],
            label="$\\lambda_{"
            + str(nbStd[j])
            + " SEM} $ = "
            + str(np.round(lambdasSelAll[j + 1], 2)),
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=14)
    ax1.set_xlabel("$\\lambda$", fontsize=14)
    ax1.legend(fontsize=10)

    ax2.plot(mse_df.index, mse_df["MSE - TOTAL"], "k.-")
    ax2.errorbar(
        mse_df.index,
        mse_df["MSE - TOTAL"],
        yerr=sem_CV,
        fmt="k.-",
        ecolor="r",
        capsize=2,
        capthick=2,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("$\\lambda$", fontsize=14)

    if not os.path.isdir("./../output/figures/"):
        os.makedirs("./../output/figures/")
    fig.savefig("./../output/figures/" + filename, bbox_inches="tight")

    plt.close()


def plot_CV_pareto(mse_df, lambdasSelAll, filename, folds):
    nbStd = np.array([5, 10])
    clr = ["r", "b"]

    fig = plt.figure(figsize=(7, 4))
    # suptitle = "Cross validation (anchor regression γ = "
    # + str(gamma) + "): " + variables[0].upper() + " -- "
    # + target.upper() + " forcing [" + ', '.join(scenarios) + "] (" + \
    #             str(startDate) + " - " + str(endDate) + ") \n"
    # fig.suptitle(suptitle, fontsize = 18)

    for i in range(mse_df.shape[1] - 1):
        plt.plot(mse_df.index, mse_df.iloc[:, i], label="_nolegend_")
    #         ax1.plot(mse_df.index, mse_df.iloc[:, i], label=folds[i])

    plt.plot(mse_df.index, mse_df.iloc[:, i + 1], "k.-")

    plt.axvline(
        lambdasSelAll[0],
        ls="--",
        color="k",
        label="$\\lambda_{opt}$ = " + str(np.round(lambdasSelAll[0], 2)),
    )

    for j in range(len(lambdasSelAll) - 1):
        plt.axvline(
            lambdasSelAll[j + 1],
            color=clr[j],
            label="$\\lambda_{"
            + str(nbStd[j])
            + " MSE} $ = "
            + str(np.round(lambdasSelAll[j + 1], 2)),
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
    plt.xlabel("$\\lambda$", fontsize=14)
    plt.legend(fontsize=10)

    if not os.path.isdir("./../output/figures/"):
        os.makedirs("./../output/figures/")
    fig.savefig("./../output/figures/" + filename, bbox_inches="tight")

    plt.close()


def plot_Pareto_fronts(
    mse, corr, mi, gamma_vals, lambda_vals, ind_opt_gamma, ind_opt_lambda
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.4))

    for i in range(mse.shape[0]):
        ax1.plot(
            mse[i, :],
            corr[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
    ax1.plot(
        mse[ind_opt_gamma, ind_opt_lambda],
        corr[ind_opt_gamma, ind_opt_lambda],
        "rs",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Mean squared error (MSE)", fontsize=14)
    ax1.set_ylabel(
        "Correlation of residuals with the anchor ($\\rho$)", fontsize=14
    )

    for i in range(mse.shape[0]):
        ax2.plot(
            mse[i, :],
            mi[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
    ax2.plot(
        mse[ind_opt_gamma, ind_opt_lambda],
        mi[ind_opt_gamma, ind_opt_lambda],
        "rs",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Mean squared error (MSE)", fontsize=14)
    ax2.set_ylabel(
        "Mutual information of residuals with the anchor ($I$)", fontsize=14
    )

    for i in range(mse.shape[0]):
        ax3.plot(
            mi[i, :],
            corr[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
    ax3.plot(
        mi[ind_opt_gamma, ind_opt_lambda],
        corr[ind_opt_gamma, ind_opt_lambda],
        "rs",
    )
    ax3.set_xlabel(
        "Mutual information of residuals with the anchor ($I$)", fontsize=14
    )
    ax3.set_ylabel(
        "Correlation of residuals with the anchor ($\\rho$)", fontsize=14
    )
    ax3.legend(bbox_to_anchor=(1.03, 1.023), loc="upper left", fontsize=12)
