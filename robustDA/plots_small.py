#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.transforms as mtransforms

from pandas.core.common import flatten
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from statistics import mode
from matplotlib import gridspec, transforms

from robustDA.utils.helpers import truncate, display_nonlinear_anchors

params = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.minor.size": 0,
    "xtick.minor.width": 0,
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

    cbar_pos = [0.364, 0.05, 0.295, 0.03]  # [left,bottom,width,height]
    cbar_axes = fig.add_axes(cbar_pos)
    cbar = fig.colorbar(
        h, cax=cbar_axes, orientation="horizontal", drawedges=False
    )
    # cbar.set_label("Temperature", fontsize=12)
    cbar.ax.locator_params(nbins=3)
    cbar.ax.tick_params(labelsize=params["xtick.labelsize"] + 10)
    cbar.outline.set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=params["axes.titlesize"])

    if filename is not None:
        fig.savefig(
            "./../output/figures/" + filename,
            dpi=2000,
            bbox_inches="tight",
        )


def plotMapCartopy_subplots(
    fig, ax, dataMap, cLim=None, dx=None, dy=None, title_subplot=None
):
    logs = np.arange(0, 360, 2.5)
    lats = np.arange(-90, 90, 2.5)

    ax_pos = ax.get_position()

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

    #     cbar_pos = [
    #         0.183,
    #         0.69,
    #         0.16,
    #         0.007,
    #     ]  # [left,bottom,width,height] [0.18, 0.05, 0.159, 0.03]

    # cbar_pos = [ax_pos.x0 + dx, ax_pos.y0 + dy, 0.2, 0.006]
    cbar_pos = [ax_pos.x0 + dx, ax_pos.y0 + dy, 0.19, 0.005] # for the small figures
    cbar_axes = fig.add_axes(cbar_pos)
    cbar = fig.colorbar(
        h,
        cax=cbar_axes,
        orientation="horizontal",
    )
    cbar.ax.locator_params(nbins=3)
    cbar.ax.tick_params(labelsize=params["xtick.labelsize"])
    cbar.outline.set_visible(False)

    if title_subplot is not None:
        ax.set_title(title_subplot, fontsize=params["axes.titlesize"])
        
        
def plotMapCartopy_subplots_maps(
    fig, ax, dataMap, cLim=None, dx=None, dy=None, title_subplot=None
):
    logs = np.arange(0, 360, 2.5)
    lats = np.arange(-90, 90, 2.5)

    ax_pos = ax.get_position()

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

    #     cbar_pos = [
    #         0.183,
    #         0.69,
    #         0.16,
    #         0.007,
    #     ]  # [left,bottom,width,height] [0.18, 0.05, 0.159, 0.03]

    cbar_pos = [ax_pos.x0 + dx, ax_pos.y0 + dy, 0.2, 0.008]
    cbar_axes = fig.add_axes(cbar_pos)
    cbar = fig.colorbar(
        h,
        cax=cbar_axes,
        orientation="horizontal",
    )
    cbar.ax.locator_params(nbins=3)
    cbar.ax.tick_params(labelsize=params["xtick.labelsize"])
    cbar.outline.set_visible(False)

    if title_subplot is not None:
        ax.set_title(title_subplot, fontsize=params["axes.titlesize"])



def make_plots(
    dict_models,
    coefRaw,
    y_test_true,
    y_test_pred,
    A,
    gamma,
    target,
    anchor,
    h_anchors,
    filename=None,
):

    #     y_test_true = dict_models["y_test"].values

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

    plt.subplots_adjust(wspace=0.25)

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
    ax.plot(
        y_test_true,
        y_test_pred,
        ".",
        color="darkseagreen",
        markersize=3,
        rasterized=True,
    )
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
    ax.plot(
        y_anchor_test,
        residuals,
        ".",
        color="peru",
        markersize=3,
        rasterized=True,
    )
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
            format="pdf",
        )

    plt.subplots_adjust(wspace=0.25)


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


def plot_CV_pareto_MSE(mse_df, lambdasSelAll, filename, folds):
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

    
def plot_maps(
    coefRaw,
    coefRawRidge,
    y_test_true,
    y_test_pred,
    y_test_pred_ridge,
    y_anchor_test,
    target,
    anchor,
    h_anchors,
    rmse,
    rmse_PA,
    corr,
    mi,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    filename=None,
):

    grid = (72, 144)

    rmse_bagging = np.mean(rmse, axis=0)
    rmse_PA_bagging = np.mean(rmse_PA, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    mi_bagging = np.mean(mi, axis=0)

    #     """ Evaluation measures (ridge) """
    #     rmse_av = 0
    #     r2_av = 0
    #     corr_av = 0
    #     corr2_av = 0
    #     corr3_av = 0
    #     mi_av = 0
    #     for i in range(len(y_test_true)):
    #         res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
    #             y_test_pred[i]
    #         ).reshape(-1, 1)
    #         rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
    #         r2_av = r2_av + r2_score(
    #             np.array(y_test_true[i]).reshape(-1, 1),
    #             np.array(y_test_pred[i]).reshape(-1, 1),
    #         )
    #         corr_av = (
    #             corr_av
    #             + np.corrcoef(
    #                 np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
    #             )[0, 1]
    #         )
    #         mi_av = (
    #             mi_av
    #             + mutual_info_regression(
    #                 np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
    #             )[0]
    #         )
    #         if len(h_anchors) == 1:
    #             corr2_av = corr2_av + np.corrcoef(
    #                 np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
    #                 res.reshape(1, -1),
    #             )[0, 1]
    #         elif len(h_anchors) == 2:
    #             corr3_av = corr3_av + np.corrcoef(
    #                 np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
    #                 res.reshape(1, -1),
    #             )[0, 1]
    #     rmse_av = rmse_av / (len(y_test_true))
    #     r2_av = r2_av / len(y_test_true)
    #     corr_av = corr_av / len(y_test_true)
    #     corr2_av = corr2_av / len(y_test_true)
    #     corr3_av = corr3_av / len(y_test_true)
    #     mi_av = mi_av / len(y_test_true)

    """ Evaluation measures (anchor) """
    rmse_av = 0
    r2_av = 0
    corr_av = 0
    corr2_av = 0
    corr3_av = 0
    mi_av = 0
    for i in range(len(y_test_true)):
        res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
            y_test_pred[i]
        ).reshape(-1, 1)
        rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
        r2_av = r2_av + r2_score(
            np.array(y_test_true[i]).reshape(-1, 1),
            np.array(y_test_pred[i]).reshape(-1, 1),
        )
        corr_av = (
            corr_av
            + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
            )[0, 1]
        )
        mi_av = (
            mi_av
            + mutual_info_regression(
                np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
            )[0]
        )
        if len(h_anchors) == 1:
            corr2_av = corr2_av + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
                res.reshape(1, -1),
            )[0, 1]
        elif len(h_anchors) == 2:
            corr3_av = corr3_av + np.corrcoef(
                np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
                res.reshape(1, -1),
            )[0, 1]
    rmse_av = rmse_av / (len(y_test_true))
    r2_av = r2_av / len(y_test_true)
    corr_av = corr_av / len(y_test_true)
    corr2_av = corr2_av / len(y_test_true)
    corr3_av = corr3_av / len(y_test_true)
    mi_av = mi_av / len(y_test_true)

    """ ############################ """
    """       Plotting started       """
    """ ############################ """
    yt = np.array(list(flatten(y_test_true))).reshape(-1, 1)
    yp = np.array(list(flatten(y_test_pred))).reshape(-1, 1)
    #     yp_ridge = np.array(list(flatten(y_test_pred_ridge))).reshape(-1, 1)
    ya = np.array(list(flatten(y_anchor_test))).reshape(-1, 1)
    residuals = (yt - yp).reshape(-1)
    #     residuals_ridge = (yt - yp_ridge).reshape(-1)

    fig = plt.figure(figsize=(15, 12))
    spec = gridspec.GridSpec(nrows=6, ncols=10)

    #     """ --- RIDGE --- """
    #     """ Plot map """
    #     ax = fig.add_subplot(
    #         spec[0:2, 0:4], projection=ccrs.Robinson(central_longitude=180)
    #     )
    #     title = "Raw coefficients (Ridge regression)"
    #     plotMapCartopy_subplots(
    #         fig, ax, coefRawRidge.reshape(grid), cLim=None, title_subplot=title
    #     )

    #     """ Prediction results """
    #     ax = fig.add_subplot(spec[0:2, 4:7])
    #     ax.plot(yt, yp_ridge, ".", color="darkseagreen", markersize=3, rasterized=True)
    #     x = np.linspace(truncate(min(yt), 2), truncate(max(yt), 2))
    #     ax.plot(x, x, "k", linestyle="solid")
    #     ax.set_xlabel("True target " + target[:3].upper())
    #     ax.set_ylabel("Predicted target " + target[:3].upper())
    #     ax.set_title(
    #         "Prediction: RMSE = " + str(np.round(rmse_av, 2)) + ", R2 = " + str(np.round(r2_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    #     """ Residuals """
    #     ax = fig.add_subplot(spec[0:2, 7:10])
    #     ax.plot(ya, residuals_ridge, ".", color="peru", markersize=3, rasterized=True)
    #     ax.set_xlabel(
    #         "Anchor $A$ (" + anchor[:3].upper() + " forcing)",
    #         fontsize=params["axes.labelsize"],
    #     )
    #     ax.set_ylabel(
    #         "Residuals " + target[:3].upper(), fontsize=params["axes.labelsize"]
    #     )
    #     if len(h_anchors) == 0:
    #         ax.set_title(
    #             "Correlation and independence: "
    #             + "\n $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    #     elif len(h_anchors) == 1:
    #         ax.set_title(
    #             "Correlation and independence: "
    #             + "\n $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + ", $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[0])
    #             + "$)$ = "
    #             + str(np.round(corr2_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    #     elif len(h_anchors) == 2:
    #         ax.set_title(
    #             anchor.upper()
    #             + " anchor:  $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + "\n $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[0])
    #             + "$)$ = "
    #             + str(np.round(corr2_av, 2))
    #             + ", $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[1])
    #             + "$)$ = "
    #             + str(np.round(corr3_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    """ --- Ridge regression --- """
    """ Plot map """
    ax = fig.add_subplot(
        spec[0:3, 0:5], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients (ridge regression)"
    
    if target == "GHG" and anchor == "aerosols":
        clim_val = 0.01
    elif target == "aerosols" and anchor == "co2":
        clim_val = 0.008
    else:
        clim_val = 0.01
    
    plotMapCartopy_subplots_maps(
        fig,
        ax,
        coefRawRidge.reshape(grid),
        cLim=clim_val, 
        dx=0.03,
        dy=0.05,
        title_subplot=title,
    )

    """ --- Anchor regression --- """
    """ Plot map """
    ax = fig.add_subplot(
        spec[0:3, 5:10], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients (anchor regression)"
    plotMapCartopy_subplots_maps(
        fig,
        ax,
        coefRaw.reshape(grid),
        cLim=clim_val,
        dx=0.06,
        dy=0.05,
        title_subplot=title,
    )

    """ Prediction results """
    ax = fig.add_subplot(spec[3:5, 0:5])
    ax.plot(yt, yp, ".", color="darkseagreen", markersize=3, rasterized=True)
    x = np.linspace(truncate(min(yt), 2), truncate(max(yt), 2))
    ax.plot(x, x, "k", linestyle="solid")
    ax.set_xlabel("True target $Y$ (" + target[:3].upper() + ")")
    ax.set_ylabel(
        r"Predicted target $\widehat{Y}$ (" + target[:3].upper() + ")"
    )
    ax.set_title(
        "Prediction: $RMSE$ = "
        + str(np.round(rmse_av, 2))
        + ", $R^2$ = "
        + str(np.round(r2_av, 2)),
        fontsize=params["axes.titlesize"],
    )

    """ Residuals """
    ax = fig.add_subplot(spec[3:5, 5:10])
    ax.plot(ya, residuals, ".", color="peru", markersize=3, rasterized=True)
    if anchor[:3] == "co2":
        ax.set_xlabel(
            "Anchor $A$ (CO$_2$)",
            fontsize=params["axes.labelsize"],
        )
    else:
        ax.set_xlabel(
            "Anchor $A$ (" + anchor[:3].upper() + ")",
            fontsize=params["axes.labelsize"],
        )

    ax.set_ylabel(
        "Residuals (" + target[:3].upper() + ")",
        fontsize=params["axes.labelsize"],
    )
#    if len(h_anchors) == 0:
    ax.set_title(
        r"Correlation \& independence: "
        + "$\\rho (A)$ = "
        + str(np.round(corr_av, 2))
        + ", $I$ = "
        + str(np.round(mi_av, 2)),
        fontsize=params["axes.titlesize"],
    )

    # elif len(h_anchors) == 1:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    # elif len(h_anchors) == 2:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + "$\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[1])
    #         + "$)$ = "
    #         + str(np.round(corr3_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )
    plt.subplots_adjust(wspace=5, hspace=0.4)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, bbox_inches="tight")
        
        
def plot_residuals(
    y_test_true,
    y_test_pred,
    y_test_pred_ridge,
    y_anchor_test,
    target,
    anchor,
    h_anchors,
    rmse,
    rmse_PA,
    corr,
    mi,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    filename=None,
):

    grid = (72, 144)

    rmse_bagging = np.mean(rmse, axis=0)
    rmse_PA_bagging = np.mean(rmse_PA, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    mi_bagging = np.mean(mi, axis=0)

    """ Evaluation measures (anchor) """
    rmse_av = 0
    r2_av = 0
    corr_av = 0
    corr2_av = 0
    corr3_av = 0
    mi_av = 0
    
    for i in range(len(y_test_true)):
        res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
            y_test_pred[i]
        ).reshape(-1, 1)
        rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
        r2_av = r2_av + r2_score(
            np.array(y_test_true[i]).reshape(-1, 1),
            np.array(y_test_pred[i]).reshape(-1, 1),
        )
        corr_av = (
            corr_av
            + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
            )[0, 1]
        )
        mi_av = (
            mi_av
            + mutual_info_regression(
                np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
            )[0]
        )
        if len(h_anchors) == 1:
            corr2_av = corr2_av + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
                res.reshape(1, -1),
            )[0, 1]
        elif len(h_anchors) == 2:
            corr3_av = corr3_av + np.corrcoef(
                np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
                res.reshape(1, -1),
            )[0, 1]
    rmse_av = rmse_av / (len(y_test_true))
    r2_av = r2_av / len(y_test_true)
    corr_av = corr_av / len(y_test_true)
    corr2_av = corr2_av / len(y_test_true)
    corr3_av = corr3_av / len(y_test_true)
    mi_av = mi_av / len(y_test_true)

    """ ############################ """
    """       Plotting started       """
    """ ############################ """
    yt = np.array(list(flatten(y_test_true))).reshape(-1, 1)
    yp = np.array(list(flatten(y_test_pred))).reshape(-1, 1)
    yp_ridge = np.array(list(flatten(y_test_pred_ridge))).reshape(-1, 1)
    ya = np.array(list(flatten(y_anchor_test))).reshape(-1, 1)
    residuals = (yt - yp).reshape(-1)
    residuals_ridge = (yt - yp_ridge).reshape(-1)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 3))

    """ Residuals """
    ax1.plot(ya, residuals, ".", color="peru", markersize=3, rasterized=True)
    if anchor[:3] == "co2":
        ax1.set_xlabel(
            "Anchor $A$ (CO$_2$)",
            fontsize=params["axes.labelsize"],
        )
    else:
        ax1.set_xlabel(
            "Anchor $A$ (" + anchor[:3].upper() + ")",
            fontsize=params["axes.labelsize"],
        )

    ax1.set_ylabel(
        "Residuals (" + target[:3].upper() + ")",
        fontsize=params["axes.labelsize"],
    )
#    if len(h_anchors) == 0:
    ax1.set_title(
        r"Correlation \& dependence: "
        + "$\\rho (A)$ = "
        + str(np.round(corr_av, 2))
        + ", $I$ = "
        + str(np.round(mi_av, 2)),
        fontsize=params["axes.titlesize"],
    )

    # elif len(h_anchors) == 1:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    # elif len(h_anchors) == 2:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + "$\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[1])
    #         + "$)$ = "
    #         + str(np.round(corr3_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )
    plt.subplots_adjust(wspace=5, hspace=0.4)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, bbox_inches="tight")
        
        
def plot_residuals_ridge(
    y_test_true,
    y_test_pred,
    y_test_pred_ridge,
    y_anchor_test,
    target,
    anchor,
    h_anchors,
    rmse,
    rmse_PA,
    corr,
    mi,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    filename=None,
):

    grid = (72, 144)

    rmse_bagging = np.mean(rmse, axis=0)
    rmse_PA_bagging = np.mean(rmse_PA, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    mi_bagging = np.mean(mi, axis=0)


    """ Evaluation measures (anchor) """
    rmse_av = 0
    r2_av = 0
    corr_av = 0
    corr2_av = 0
    corr3_av = 0
    mi_av = 0
    
    for i in range(len(y_test_true)):
        res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
            y_test_pred_ridge[i]
        ).reshape(-1, 1)
        rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
        r2_av = r2_av + r2_score(
            np.array(y_test_true[i]).reshape(-1, 1),
            np.array(y_test_pred_ridge[i]).reshape(-1, 1),
        )
        corr_av = (
            corr_av
            + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
            )[0, 1]
        )
        mi_av = (
            mi_av
            + mutual_info_regression(
                np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
            )[0]
        )
        if len(h_anchors) == 1:
            corr2_av = corr2_av + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
                res.reshape(1, -1),
            )[0, 1]
        elif len(h_anchors) == 2:
            corr3_av = corr3_av + np.corrcoef(
                np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
                res.reshape(1, -1),
            )[0, 1]
    rmse_av = rmse_av / (len(y_test_true))
    r2_av = r2_av / len(y_test_true)
    corr_av = corr_av / len(y_test_true)
    corr2_av = corr2_av / len(y_test_true)
    corr3_av = corr3_av / len(y_test_true)
    mi_av = mi_av / len(y_test_true)

    """ ############################ """
    """       Plotting started       """
    """ ############################ """
    yt = np.array(list(flatten(y_test_true))).reshape(-1, 1)
    yp = np.array(list(flatten(y_test_pred))).reshape(-1, 1)
    yp_ridge = np.array(list(flatten(y_test_pred_ridge))).reshape(-1, 1)
    ya = np.array(list(flatten(y_anchor_test))).reshape(-1, 1)
    residuals = (yt - yp).reshape(-1)
    residuals_ridge = (yt - yp_ridge).reshape(-1)

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 3))

    """ Residuals """
    ax1.plot(ya, residuals_ridge, ".", color="peru", markersize=3, rasterized=True)
    if anchor[:3] == "co2":
        ax1.set_xlabel(
            "Anchor $A$ (CO$_2$)",
            fontsize=params["axes.labelsize"],
        )
    else:
        ax1.set_xlabel(
            "Anchor $A$ (" + anchor[:3].upper() + ")",
            fontsize=params["axes.labelsize"],
        )

    ax1.set_ylabel(
        "Residuals (" + target[:3].upper() + ")",
        fontsize=params["axes.labelsize"],
    )
#    if len(h_anchors) == 0:
    ax1.set_title(
        r"Correlation \& independence: "
        + "$\\rho (A)$ = "
        + str(np.round(corr_av, 2))
        + ", $I$ = "
        + str(np.round(mi_av, 2)),
        fontsize=params["axes.titlesize"],
    )

    # elif len(h_anchors) == 1:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    # elif len(h_anchors) == 2:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + "$\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[1])
    #         + "$)$ = "
    #         + str(np.round(corr3_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )
    plt.subplots_adjust(wspace=5, hspace=0.4)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, bbox_inches="tight")
    

def plot_Pareto(
    rmse,
    rmse_PA,
    corr,
    mi,
    h_anchors,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    formatFig,
    filename=None,
):

    rmse_bagging = np.mean(rmse, axis=0)
    rmse_PA_bagging = np.mean(rmse_PA, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    mi_bagging = np.mean(mi, axis=0)

    i1 = int(mode(list(ind_gamma_opt.reshape(-1))))
    i2 = int(mode(list(ind_lambda_opt.reshape(-1))))
    i2_ridge = int(mode(list(ind_lambda_opt_ridge.reshape(-1))))
    iv1 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj1[:, 0]))),
            int(mode(list(ind_vect_ideal_obj1[:, 1]))),
        ]
    )
    iv2 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj2[:, 0]))),
            int(mode(list(ind_vect_ideal_obj2[:, 1]))),
        ]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5))

    #     _, _, vect_ideal, vect_Nadir = choose_gamma_lambda_pareto(
    #         rmse, mi, maxX=False, maxY=False,)

    for i in range(rmse_bagging.shape[0]):
        
        gamma_exponent = np.log10(gamma_vals[i])
 
        if gamma_exponent.is_integer():
            if gamma_exponent == 0 or gamma_exponent == 1: # if gamma=1 or gamma=10
                (line,) = ax1.plot(
                    rmse_bagging[i, :],
                    rmse_PA_bagging[i, :],
                    ".-",
                    label="$\\gamma = {}$ ".format(gamma_vals[i]),
                )
            elif gamma_exponent > 1:
                (line,) = ax1.plot(
                    rmse_bagging[i, :],
                    rmse_PA_bagging[i, :],
                    ".-",
                    label="$\\gamma = 10^{}$ ".format(int(gamma_exponent)),
                )
        else:
            (line,) = ax1.plot(
                rmse_bagging[i, :],
                rmse_PA_bagging[i, :],
                ".-",
                label="$\\gamma = {}$ ".format(gamma_vals[i]),
            )
            
        ax1.plot(
            rmse_bagging[i, 0],
            rmse_PA_bagging[i, 0],
            color=line.get_color(),
            marker="o",
            markersize=8,
        )
    ax1.plot(
        rmse_bagging[i1, i2],
        rmse_PA_bagging[i1, i2],
        "ks",
    )
    ideal_vector, = ax1.plot(
        rmse_bagging[
            iv1[0].astype(int),
            iv1[1].astype(int),
        ],
        rmse_PA_bagging[
            iv2[0].astype(int),
            iv2[1].astype(int),
        ],
        "k^",
        markersize=7,
    )
    ridge_optimal, = ax1.plot(
        rmse_bagging[0, i2_ridge],
        rmse_PA_bagging[0, i2_ridge],
        "ks",
        markersize=7,
    )
    anchor_optimal, = ax1.plot(
        rmse_bagging[i1, i2],
        rmse_PA_bagging[i1, i2],
        "ko",
        markersize=7,
    )

    #     ax1.set_xscale("log")
    #     ticks = [0.5, 0.6, 0.8, 1, 1.5, 2]
    #     ax1.set_xticks(ticks)
    #     ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel(
        "Root mean squared error ($RMSE$)  (" u"\N{DEGREE SIGN}C)",
        fontsize=params["axes.labelsize"],
    )
    ax1.set_ylabel(
        "$RMSE_{\Pi_{\mathbf{A}}}$ (error explained by $A$)",
        fontsize=params["axes.labelsize"],
    )
    
    legend1 = ax1.legend(fontsize=params["axes.labelsize"] - 2, loc = "upper right")
    legend2 = ax1.legend([ideal_vector,ridge_optimal, anchor_optimal],['Ideal vector','Ridge (optimal)', 'Anchor (optimal)'], loc='upper left', fontsize=params["axes.labelsize"] - 2)
    ax1.add_artist(legend1)
    
    # ax1.legend(fontsize=params["axes.labelsize"] - 4)

    if len(h_anchors) == 0: # plot correlation
        for i in range(rmse_bagging.shape[0]):
            (line,) = ax2.plot(
                rmse_bagging[i, :],
                corr_bagging[i, :],
                ".-",
                label="$\\gamma = $ " + str(gamma_vals[i]),
            )
            ax2.plot(
                rmse_bagging[i, 0],
                corr_bagging[i, 0],
                color=line.get_color(),
                marker="o",
                markersize=8,
            )
        ax2.plot(
            rmse_bagging[i1, i2],
            corr_bagging[i1, i2], # corr for linear anchor and mi for nonlinear anchor
            "ko",
            markersize=7,
        )
        ax2.plot(
            rmse_bagging[0, i2_ridge],
            corr_bagging[0, i2_ridge],
            "ks",
            markersize=7,
        )
        #     ax.set_yscale("log")
        #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
        #     ax.set_xticks(ticks)
        #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xlabel(
            "Root mean squared error ($RMSE$) (" u"\N{DEGREE SIGN}C)",
            fontsize=params["axes.labelsize"],
        )
        ax2.set_ylabel(
            "Linear correlation (residuals, $A$)",
            fontsize=params["axes.labelsize"],
        )
    else: # plot mutual information
        for i in range(rmse_bagging.shape[0]):
            (line,) = ax2.plot(
                rmse_bagging[i, :],
                mi_bagging[i, :],
                ".-",
                label="$\\gamma = $ " + str(gamma_vals[i]),
            )
            ax2.plot(
                rmse_bagging[i, 0],
                mi_bagging[i, 0],
                color=line.get_color(),
                marker="o",
                markersize=8,
            )
        ax2.plot(
            rmse_bagging[i1, i2],
            mi_bagging[i1, i2], 
            "ko",
            markersize=7,
        )
        ax2.plot(
            rmse_bagging[0, i2_ridge],
            mi_bagging[0, i2_ridge],
            "ks",
            markersize=7,
        )
        #     ax.set_yscale("log")
        #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
        #     ax.set_xticks(ticks)
        #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xlabel(
            "Root mean squared error ($RMSE$) (" u"\N{DEGREE SIGN}C)",
            fontsize=params["axes.labelsize"],
        )
        ax2.set_ylabel(
            "Mutual information (residuals, $A$)",
            fontsize=params["axes.labelsize"],
        )

    plt.subplots_adjust(wspace=0.25)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, format = formatFig, bbox_inches="tight")


def plot_Pareto_3(
    rmse,
    corr,
    mi,
    h_anchors,
    gamma_vals,
    lambda_vals,
    ind_opt_gamma,
    ind_opt_lambda,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    filename=None,
):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    #     _, _, vect_ideal, vect_Nadir = choose_gamma_lambda_pareto(
    #         rmse, mi, maxX=False, maxY=False,)

    for i in range(rmse.shape[0]):
        (line,) = ax1.plot(
            rmse[i, :],
            corr[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax1.plot(
            rmse[i, 0],
            corr[i, 0],
            color=line.get_color(),
            marker="o",
        )
    ax1.plot(
        rmse[ind_opt_gamma, ind_opt_lambda],
        corr[ind_opt_gamma, ind_opt_lambda],
        "ks",
    )
    #     ax1.set_xscale("log")
    ax1.set_xlabel("Root mean squared error (RMSE)", fontsize=17)
    ax1.set_ylabel("Correlation of residuals and anchor", fontsize=17)
    ax1.legend(fontsize=14)

    for i in range(rmse.shape[0]):
        (line,) = ax2.plot(
            rmse[i, :],
            mi[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax2.plot(
            rmse[i, 0],
            mi[i, 0],
            color=line.get_color(),
            marker="o",
        )
    ax2.plot(
        rmse[ind_opt_gamma, ind_opt_lambda],
        mi[ind_opt_gamma, ind_opt_lambda],
        "ks",
    )
    #     ax2.set_xscale("log")
    ax2.set_xlabel("Root mean squared error (RMSE)", fontsize=17)
    ax2.set_ylabel("Mutual information of residuals and anchor", fontsize=17)

    for i in range(rmse.shape[0]):
        (line,) = ax3.plot(
            corr[i, :],
            mi[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax3.plot(
            corr[i, 0],
            mi[i, 0],
            color=line.get_color(),
            marker="o",
        )
    ax3.plot(
        corr[ind_opt_gamma, ind_opt_lambda],
        mi[ind_opt_gamma, ind_opt_lambda],
        "ks",
    )
    ax3.set_xlabel("Correlation of residuals and anchor", fontsize=17)
    ax3.set_ylabel("Mutual information of residuals and anchor", fontsize=17)

    # plot ideal vector
    if len(h_anchors) == 0:
        ax1.plot(
            rmse[
                ind_vect_ideal_obj1[0].astype(int),
                ind_vect_ideal_obj1[1].astype(int),
            ],
            corr[
                ind_vect_ideal_obj2[0].astype(int),
                ind_vect_ideal_obj2[1].astype(int),
            ],
            "k^",
        )
    else:
        ax2.plot(
            rmse[
                ind_vect_ideal_obj1[0].astype(int),
                ind_vect_ideal_obj1[1].astype(int),
            ],
            mi[
                ind_vect_ideal_obj2[0].astype(int), ind_vect_ideal_obj2[1]
            ].astype(int),
            "k^",
        )

    plt.subplots_adjust(wspace=0.25)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, bbox_inches="tight")


def plot_all(
    coefRaw,
    coefRawRidge,
    y_test_true,
    y_test_pred,
    y_test_pred_ridge,
    y_anchor_test,
    target,
    anchor,
    h_anchors,
    rmse,
    corr,
    mi,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    alpha_bagging,
    power_bagging,
    filename=None,
):

    grid = (72, 144)

    #     axis_labels = ["A", "B", "C", "D", "E", "F", "G"]

    rmse_bagging = np.mean(rmse, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    mi_bagging = np.mean(mi, axis=0)

    #     """ Evaluation measures (ridge) """
    #     rmse_av = 0
    #     r2_av = 0
    #     corr_av = 0
    #     corr2_av = 0
    #     corr3_av = 0
    #     mi_av = 0
    #     for i in range(len(y_test_true)):
    #         res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
    #             y_test_pred[i]
    #         ).reshape(-1, 1)
    #         rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
    #         r2_av = r2_av + r2_score(
    #             np.array(y_test_true[i]).reshape(-1, 1),
    #             np.array(y_test_pred[i]).reshape(-1, 1),
    #         )
    #         corr_av = (
    #             corr_av
    #             + np.corrcoef(
    #                 np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
    #             )[0, 1]
    #         )
    #         mi_av = (
    #             mi_av
    #             + mutual_info_regression(
    #                 np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
    #             )[0]
    #         )
    #         if len(h_anchors) == 1:
    #             corr2_av = corr2_av + np.corrcoef(
    #                 np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
    #                 res.reshape(1, -1),
    #             )[0, 1]
    #         elif len(h_anchors) == 2:
    #             corr3_av = corr3_av + np.corrcoef(
    #                 np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
    #                 res.reshape(1, -1),
    #             )[0, 1]
    #     rmse_av = rmse_av / (len(y_test_true))
    #     r2_av = r2_av / len(y_test_true)
    #     corr_av = corr_av / len(y_test_true)
    #     corr2_av = corr2_av / len(y_test_true)
    #     corr3_av = corr3_av / len(y_test_true)
    #     mi_av = mi_av / len(y_test_true)

    """ Evaluation measures (anchor) """
    rmse_av = 0
    r2_av = 0
    corr_av = 0
    corr2_av = 0
    corr3_av = 0
    mi_av = 0
    for i in range(len(y_test_true)):
        res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
            y_test_pred[i]
        ).reshape(-1, 1)
        rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
        r2_av = r2_av + r2_score(
            np.array(y_test_true[i]).reshape(-1, 1),
            np.array(y_test_pred[i]).reshape(-1, 1),
        )
        corr_av = (
            corr_av
            + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
            )[0, 1]
        )
        mi_av = (
            mi_av
            + mutual_info_regression(
                np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
            )[0]
        )
        if len(h_anchors) == 1:
            corr2_av = corr2_av + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
                res.reshape(1, -1),
            )[0, 1]
        elif len(h_anchors) == 2:
            corr3_av = corr3_av + np.corrcoef(
                np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
                res.reshape(1, -1),
            )[0, 1]
    rmse_av = rmse_av / (len(y_test_true))
    r2_av = r2_av / len(y_test_true)
    corr_av = corr_av / len(y_test_true)
    corr2_av = corr2_av / len(y_test_true)
    corr3_av = corr3_av / len(y_test_true)
    mi_av = mi_av / len(y_test_true)

    """ ############################ """
    """       Plotting started       """
    """ ############################ """
    yt = np.array(list(flatten(y_test_true))).reshape(-1, 1)
    yp = np.array(list(flatten(y_test_pred))).reshape(-1, 1)
    #     yp_ridge = np.array(list(flatten(y_test_pred_ridge))).reshape(-1, 1)
    ya = np.array(list(flatten(y_anchor_test))).reshape(-1, 1)
    residuals = (yt - yp).reshape(-1)
    #     residuals_ridge = (yt - yp_ridge).reshape(-1)

    fig = plt.figure(figsize=(6.7, 5.3))
    spec = gridspec.GridSpec(nrows=7, ncols=10)

    #     """ --- RIDGE --- """
    #     """ Plot map """
    #     ax = fig.add_subplot(
    #         spec[0:2, 0:4], projection=ccrs.Robinson(central_longitude=180)
    #     )
    #     title = "Raw coefficients (Ridge regression)"
    #     plotMapCartopy_subplots(
    #         fig, ax, coefRawRidge.reshape(grid), cLim=None, title_subplot=title
    #     )

    #     """ Prediction results """
    #     ax = fig.add_subplot(spec[0:2, 4:7])
    #     ax.plot(yt, yp_ridge, ".", color="darkseagreen", markersize=3, rasterized=True)
    #     x = np.linspace(truncate(min(yt), 2), truncate(max(yt), 2))
    #     ax.plot(x, x, "k", linestyle="solid")
    #     ax.set_xlabel("True target " + target[:3].upper())
    #     ax.set_ylabel("Predicted target " + target[:3].upper())
    #     ax.set_title(
    #         "Prediction: RMSE = " + str(np.round(rmse_av, 2)) + ", R2 = " + str(np.round(r2_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    #     """ Residuals """
    #     ax = fig.add_subplot(spec[0:2, 7:10])
    #     ax.plot(ya, residuals_ridge, ".", color="peru", markersize=3, rasterized=True)
    #     ax.set_xlabel(
    #         "Anchor $A$ (" + anchor[:3].upper() + " forcing)",
    #         fontsize=params["axes.labelsize"],
    #     )
    #     ax.set_ylabel(
    #         "Residuals " + target[:3].upper(), fontsize=params["axes.labelsize"]
    #     )
    #     if len(h_anchors) == 0:
    #         ax.set_title(
    #             "Correlation and independence: "
    #             + "\n $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    #     elif len(h_anchors) == 1:
    #         ax.set_title(
    #             "Correlation and independence: "
    #             + "\n $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + ", $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[0])
    #             + "$)$ = "
    #             + str(np.round(corr2_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    #     elif len(h_anchors) == 2:
    #         ax.set_title(
    #             anchor.upper()
    #             + " anchor:  $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + "\n $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[0])
    #             + "$)$ = "
    #             + str(np.round(corr2_av, 2))
    #             + ", $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[1])
    #             + "$)$ = "
    #             + str(np.round(corr3_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    """ --- Anchor regression --- """
    """ Plot map """
    ax = fig.add_subplot(
        spec[0:2, 0:4], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients"
    plotMapCartopy_subplots(
        fig, ax, coefRaw.reshape(grid), cLim=None, title_subplot=title
    )
    label_panel(ax, "A")

    """ Prediction results """
    ax = fig.add_subplot(spec[0:2, 4:7])
    ax.plot(yt, yp, ".", color="darkseagreen", markersize=3, rasterized=True)
    x = np.linspace(truncate(min(yt), 2), truncate(max(yt), 2))
    ax.plot(x, x, "k", linestyle="solid")
    ax.set_xlabel("True target $Y$ (" + target[:3].upper() + ")")
    ax.set_ylabel(
        r"Predicted target $\widehat{Y}$ (" + target[:3].upper() + ")"
    )
    ax.set_title(
        "Prediction: RMSE = "
        + str(np.round(rmse_av, 2))
        + ", R2 = "
        + str(np.round(r2_av, 2)),
        fontsize=params["axes.titlesize"],
    )
    label_panel(ax, "B")

    """ Residuals """
    ax = fig.add_subplot(spec[0:2, 7:10])
    ax.plot(ya, residuals, ".", color="peru", markersize=3, rasterized=True)
    ax.set_xlabel(
        "Anchor $A$ (" + anchor[:3].upper() + ")",
        fontsize=params["axes.labelsize"],
    )
    ax.set_ylabel(
        "Residuals (" + target[:3].upper() + ")",
        fontsize=params["axes.labelsize"],
    )
    if len(h_anchors) == 0:
        ax.set_title(
            "Correlation and independence: "
            + "\n $\\rho (A)$ = "
            + str(np.round(corr_av, 2))
            + ", $I$ = "
            + str(np.round(mi_av, 2)),
            fontsize=params["axes.titlesize"],
        )

    elif len(h_anchors) == 1:
        ax.set_title(
            "Correlation and independence: "
            + "\n $\\rho (A)$ = "
            + str(np.round(corr_av, 2))
            + ", $\\rho ($"
            + display_nonlinear_anchors(h_anchors[0])
            + "$)$ = "
            + str(np.round(corr2_av, 2))
            + ", $I$ = "
            + str(np.round(mi_av, 2)),
            fontsize=params["axes.titlesize"],
        )

    elif len(h_anchors) == 2:
        ax.set_title(
            anchor.upper()
            + " anchor:  $\\rho (A)$ = "
            + str(np.round(corr_av, 2))
            + "\n $\\rho ($"
            + display_nonlinear_anchors(h_anchors[0])
            + "$)$ = "
            + str(np.round(corr2_av, 2))
            + ", $\\rho ($"
            + display_nonlinear_anchors(h_anchors[1])
            + "$)$ = "
            + str(np.round(corr3_av, 2))
            + ", $I$ = "
            + str(np.round(mi_av, 2)),
            fontsize=params["axes.titlesize"],
        )
    label_panel(ax, "C")

    """ Plot RMSE vs correlation vs MI """
    i1 = int(mode(list(ind_gamma_opt.reshape(-1))))
    i2 = int(mode(list(ind_lambda_opt.reshape(-1))))
    i2_ridge = int(mode(list(ind_lambda_opt_ridge.reshape(-1))))
    iv1 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj1[:, 0]))),
            int(mode(list(ind_vect_ideal_obj1[:, 1]))),
        ]
    )
    iv2 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj2[:, 0]))),
            int(mode(list(ind_vect_ideal_obj2[:, 1]))),
        ]
    )

    ax = fig.add_subplot(spec[2:5, 0:5])

    for i in range(rmse_bagging.shape[0]):
        (line,) = ax.plot(
            rmse_bagging[i, :],
            corr_bagging[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax.plot(
            rmse_bagging[i, 0],
            corr_bagging[i, 0],
            color=line.get_color(),
            marker="o",
            markersize=9,
        )
    ax.plot(rmse_bagging[i1, i2], corr_bagging[i1, i2], "ks", markersize=8)
    ax.plot(
        rmse_bagging[0, i2_ridge],
        corr_bagging[0, i2_ridge],
        "ko",
        markersize=9,
    )

    #     ax.set_xscale("log")
    #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
    #     ax.set_xticks(ticks)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(
        "Root mean squared error (RMSE) (" u"\N{DEGREE SIGN}C)",
        fontsize=params["axes.labelsize"],
    )
    ax.set_ylabel(
        "Rank correlation (residuals, anchor)",
        fontsize=params["axes.labelsize"],
    )
    ax.legend(fontsize=params["axes.labelsize"] - 2)

    # plot ideal vector
    if len(h_anchors) == 0:
        ax.plot(
            rmse_bagging[
                iv1[0].astype(int),
                iv1[1].astype(int),
            ],
            corr_bagging[
                iv2[0].astype(int),
                iv2[1].astype(int),
            ],
            "k^",
            markersize=8,
        )
    label_panel(ax, "D")

    ax = fig.add_subplot(spec[2:5, 5:10])
    for i in range(rmse_bagging.shape[0]):
        (line,) = ax.plot(
            rmse_bagging[i, :],
            mi_bagging[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax.plot(
            rmse_bagging[i, 0],
            mi_bagging[i, 0],
            color=line.get_color(),
            marker="o",
            markersize=9,
        )
    ax.plot(
        rmse_bagging[i1, i2],
        mi_bagging[i1, i2],
        "ks",
        markersize=8,
    )
    ax.plot(
        rmse_bagging[0, i2_ridge],
        mi_bagging[0, i2_ridge],
        "ko",
        markersize=9,
    )
    ax.set_yscale("log")
    #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
    #     ax.set_xticks(ticks)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(
        "Root mean squared error (RMSE) (" u"\N{DEGREE SIGN}C)",
        fontsize=params["axes.labelsize"],
    )
    ax.set_ylabel(
        "Mutual information (residuals, anchor)",
        fontsize=params["axes.labelsize"],
    )

    # plot ideal vector
    if len(h_anchors) == 1:
        ax.plot(
            rmse_bagging[
                iv1[0].astype(int),
                iv1[1].astype(int),
            ],
            mi_bagging[iv2[0].astype(int), iv2[1]].astype(int),
            "k^",
            markersize=8,
        )
    label_panel(ax, "E")

    """ Plot HT results """
    bins = np.linspace(0, 1, 20)
    ax = fig.add_subplot(spec[5:7, 0:5])
    ax.hist(
        alpha_bagging.reshape(-1),
        bins,
        fill=True,
        alpha=0.8,
        linewidth=0,
        color="steelblue",
        label="Type I error",
    )
    ax.legend(fontsize=params["axes.labelsize"])
    label_panel(ax, "F")

    ax = fig.add_subplot(spec[5:7, 5:10])
    ax.hist(
        power_bagging.reshape(-1),
        bins,
        fill=True,
        alpha=0.8,
        linewidth=0,
        color="brown",
        label="Power of test",
    )
    ax.legend(fontsize=params["axes.labelsize"])
    label_panel(ax, "G")

    #     axes = [fig.add_subplot(gs[i]) for i in range(3)]

    #     for ax, label in zip(axes, axis_labels):
    #     bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    #     fig.text(bbox.x0, bbox.y1, axis_label, fontsize=12, fontweight="bold", va="top", ha="left",
    #              transform=None)

    #     label_panel(ax1,'A')
    #     label_panel(ax2,'B')
    #     label_panel(ax3,'C')
    #     label_panel(ax4,'D')

    plt.subplots_adjust(wspace=2, hspace=1)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig(
            "./../output/figures/" + filename,
            bbox_inches="tight",
            format="pdf",
        )


""" Main plotting function """


def plot_all_v2(
    coefRaw,
    coefRawRidge,
    y_test_true,
    y_test_pred,
    y_test_pred_ridge,
    y_anchor_test,
    target,
    anchor,
    h_anchors,
    rmse,
    rmse_PA,
    corr,
    mi,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    alpha_bagging,
    power_bagging,
    formatFig,
    filename=None,
):

    grid = (72, 144)

    rmse_bagging = np.mean(rmse, axis=0)
    rmse_PA_bagging = np.mean(rmse_PA, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    mi_bagging = np.mean(mi, axis=0)

    #     """ Evaluation measures (ridge) """
    #     rmse_av = 0
    #     r2_av = 0
    #     corr_av = 0
    #     corr2_av = 0
    #     corr3_av = 0
    #     mi_av = 0
    #     for i in range(len(y_test_true)):
    #         res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
    #             y_test_pred[i]
    #         ).reshape(-1, 1)
    #         rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
    #         r2_av = r2_av + r2_score(
    #             np.array(y_test_true[i]).reshape(-1, 1),
    #             np.array(y_test_pred[i]).reshape(-1, 1),
    #         )
    #         corr_av = (
    #             corr_av
    #             + np.corrcoef(
    #                 np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
    #             )[0, 1]
    #         )
    #         mi_av = (
    #             mi_av
    #             + mutual_info_regression(
    #                 np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
    #             )[0]
    #         )
    #         if len(h_anchors) == 1:
    #             corr2_av = corr2_av + np.corrcoef(
    #                 np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
    #                 res.reshape(1, -1),
    #             )[0, 1]
    #         elif len(h_anchors) == 2:
    #             corr3_av = corr3_av + np.corrcoef(
    #                 np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
    #                 res.reshape(1, -1),
    #             )[0, 1]
    #     rmse_av = rmse_av / (len(y_test_true))
    #     r2_av = r2_av / len(y_test_true)
    #     corr_av = corr_av / len(y_test_true)
    #     corr2_av = corr2_av / len(y_test_true)
    #     corr3_av = corr3_av / len(y_test_true)
    #     mi_av = mi_av / len(y_test_true)

    """ Evaluation measures (anchor) """
    rmse_av = 0
    r2_av = 0
    corr_av = 0
    corr2_av = 0
    corr3_av = 0
    mi_av = 0
    for i in range(len(y_test_true)):
        res = np.array(y_test_true[i]).reshape(-1, 1) - np.array(
            y_test_pred[i]
        ).reshape(-1, 1)
        rmse_av = rmse_av + np.sqrt(np.mean(res ** 2))
        r2_av = r2_av + r2_score(
            np.array(y_test_true[i]).reshape(-1, 1),
            np.array(y_test_pred[i]).reshape(-1, 1),
        )
        corr_av = (
            corr_av
            + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1), res.reshape(1, -1)
            )[0, 1]
        )
        mi_av = (
            mi_av
            + mutual_info_regression(
                np.array(y_anchor_test[i]).reshape(-1, 1), res.reshape(-1)
            )[0]
        )
        if len(h_anchors) == 1:
            corr2_av = corr2_av + np.corrcoef(
                np.array(y_anchor_test[i]).reshape(1, -1) ** 2,
                res.reshape(1, -1),
            )[0, 1]
        elif len(h_anchors) == 2:
            corr3_av = corr3_av + np.corrcoef(
                np.abs(np.array(y_anchor_test[i]).reshape(1, -1)),
                res.reshape(1, -1),
            )[0, 1]
    rmse_av = rmse_av / (len(y_test_true))
    r2_av = r2_av / len(y_test_true)
    corr_av = corr_av / len(y_test_true)
    corr2_av = corr2_av / len(y_test_true)
    corr3_av = corr3_av / len(y_test_true)
    mi_av = mi_av / len(y_test_true)

    """ ############################ """
    """       Plotting started       """
    """ ############################ """
    yt = np.array(list(flatten(y_test_true))).reshape(-1, 1)
    yp = np.array(list(flatten(y_test_pred))).reshape(-1, 1)
    #     yp_ridge = np.array(list(flatten(y_test_pred_ridge))).reshape(-1, 1)
    ya = np.array(list(flatten(y_anchor_test))).reshape(-1, 1)
    residuals = (yt - yp).reshape(-1)
    #     residuals_ridge = (yt - yp_ridge).reshape(-1)

    fig = plt.figure(figsize=(13.5, 17))
    spec = gridspec.GridSpec(nrows=10, ncols=10)

    #     """ --- RIDGE --- """
    #     """ Plot map """
    #     ax = fig.add_subplot(
    #         spec[0:2, 0:4], projection=ccrs.Robinson(central_longitude=180)
    #     )
    #     title = "Raw coefficients (Ridge regression)"
    #     plotMapCartopy_subplots(
    #         fig, ax, coefRawRidge.reshape(grid), cLim=None, title_subplot=title
    #     )

    #     """ Prediction results """
    #     ax = fig.add_subplot(spec[0:2, 4:7])
    #     ax.plot(yt, yp_ridge, ".", color="darkseagreen", markersize=3, rasterized=True)
    #     x = np.linspace(truncate(min(yt), 2), truncate(max(yt), 2))
    #     ax.plot(x, x, "k", linestyle="solid")
    #     ax.set_xlabel("True target " + target[:3].upper())
    #     ax.set_ylabel("Predicted target " + target[:3].upper())
    #     ax.set_title(
    #         "Prediction: RMSE = " + str(np.round(rmse_av, 2)) + ", R2 = " + str(np.round(r2_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    #     """ Residuals """
    #     ax = fig.add_subplot(spec[0:2, 7:10])
    #     ax.plot(ya, residuals_ridge, ".", color="peru", markersize=3, rasterized=True)
    #     ax.set_xlabel(
    #         "Anchor $A$ (" + anchor[:3].upper() + " forcing)",
    #         fontsize=params["axes.labelsize"],
    #     )
    #     ax.set_ylabel(
    #         "Residuals " + target[:3].upper(), fontsize=params["axes.labelsize"]
    #     )
    #     if len(h_anchors) == 0:
    #         ax.set_title(
    #             "Correlation and independence: "
    #             + "\n $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    #     elif len(h_anchors) == 1:
    #         ax.set_title(
    #             "Correlation and independence: "
    #             + "\n $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + ", $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[0])
    #             + "$)$ = "
    #             + str(np.round(corr2_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    #     elif len(h_anchors) == 2:
    #         ax.set_title(
    #             anchor.upper()
    #             + " anchor:  $\\rho (A)$ = "
    #             + str(np.round(corr_av, 2))
    #             + "\n $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[0])
    #             + "$)$ = "
    #             + str(np.round(corr2_av, 2))
    #             + ", $\\rho ($"
    #             + display_nonlinear_anchors(h_anchors[1])
    #             + "$)$ = "
    #             + str(np.round(corr3_av, 2))
    #             + ", $I$ = "
    #             + str(np.round(mi_av, 2)),
    #             fontsize=params["axes.titlesize"],
    #         )

    """ --- Ridge regression --- """
    """ Plot map """
    ax = fig.add_subplot(
        spec[0:3, 0:5], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients (ridge regression)"
    
    if target == "GHG" and anchor == "aerosols":
        clim_val = 0.01
    elif target == "aerosols" and anchor == "co2":
        clim_val = 0.008
    else:
        clim_val = 0.01
    
    plotMapCartopy_subplots(
        fig,
        ax,
        coefRawRidge.reshape(grid),
        cLim=clim_val, 
        dx=0.030,
        dy=0.04,
        title_subplot=title,
    )
    label_panel(ax, "A")

    """ --- Anchor regression --- """
    """ Plot map """
    ax = fig.add_subplot(
        spec[0:3, 5:10], projection=ccrs.Robinson(central_longitude=180)
    )
    title = "Raw coefficients (anchor regression)"
    plotMapCartopy_subplots(
        fig,
        ax,
        coefRaw.reshape(grid),
        cLim=clim_val,
        dx=0.06,
        dy=0.04,
        title_subplot=title,
    )
    label_panel(ax, "B")

    """ Prediction results """
    ax = fig.add_subplot(spec[3:5, 0:5])
    ax.plot(yt, yp, ".", color="darkseagreen", markersize=3, rasterized=True)
    x = np.linspace(truncate(min(yt), 2), truncate(max(yt), 2))
    ax.plot(x, x, "k", linestyle="solid")
    ax.set_xlabel("True target $Y$ (" + target[:3].upper() + ")")
    ax.set_ylabel(
        r"Predicted target $\widehat{Y}$"
    )
    ax.set_title(
        "Prediction: $RMSE$ = "
        + str(np.round(rmse_av, 2))
        + ", $R^2$ = "
        + str(np.round(r2_av, 2)),
        fontsize=params["axes.titlesize"],
    )
    label_panel(ax, "C")

    """ Residuals """
    ax = fig.add_subplot(spec[3:5, 5:10])
    ax.plot(ya, residuals, ".", color="peru", markersize=3, rasterized=True)
    if anchor[:3] == "co2":
        ax.set_xlabel(
            "Anchor $A$ (CO$_2$)",
            fontsize=params["axes.labelsize"],
        )
    else:
        ax.set_xlabel(
            "Anchor $A$ (" + anchor[:3].upper() + ")",
            fontsize=params["axes.labelsize"],
        )

    ax.set_ylabel(
        "Residuals (" + target[:3].upper() + ")",
        fontsize=params["axes.labelsize"],
    )
#    if len(h_anchors) == 0:
    ax.set_title(
        r"Correlation \& dependence: "
        + "$\\rho (A)$ = "
        + str(np.round(corr_av, 2))
        + ", $I$ = "
        + str(np.round(mi_av, 2)),
        fontsize=params["axes.titlesize"],
    )

    # elif len(h_anchors) == 1:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )

    # elif len(h_anchors) == 2:
    #     ax.set_title(
    #         r"Correlation \& independence: "
    #         + "$\\rho (A)$ = "
    #         + str(np.round(corr_av, 2))
    #         + "$\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[0])
    #         + "$)$ = "
    #         + str(np.round(corr2_av, 2))
    #         + ", \n $\\rho ($"
    #         + display_nonlinear_anchors(h_anchors[1])
    #         + "$)$ = "
    #         + str(np.round(corr3_av, 2))
    #         + ", $I$ = "
    #         + str(np.round(mi_av, 2)),
    #         fontsize=params["axes.titlesize"],
    #     )
    label_panel(ax, "D")

    """ Plot RMSE vs correlation vs MI """
    i1 = int(mode(list(ind_gamma_opt.reshape(-1))))
    i2 = int(mode(list(ind_lambda_opt.reshape(-1))))
    i2_ridge = int(mode(list(ind_lambda_opt_ridge.reshape(-1))))
    iv1 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj1[:, 0]))),
            int(mode(list(ind_vect_ideal_obj1[:, 1]))),
        ]
    )
    iv2 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj2[:, 0]))),
            int(mode(list(ind_vect_ideal_obj2[:, 1]))),
        ]
    )

    ax = fig.add_subplot(spec[5:8, 0:5])

    for i in range(rmse_bagging.shape[0]):

        gamma_exponent = np.log10(gamma_vals[i])
 
        if gamma_exponent.is_integer():
            if gamma_exponent == 0 or gamma_exponent == 1: # if gamma=1 or gamma=10
                (line,) = ax.plot(
                    rmse_bagging[i, :],
                    rmse_PA_bagging[i, :],
                    ".-",
                    label="$\\gamma = {}$ ".format(gamma_vals[i]),
                )
            elif gamma_exponent > 1:
                (line,) = ax.plot(
                    rmse_bagging[i, :],
                    rmse_PA_bagging[i, :],
                    ".-",
                    label="$\\gamma = 10^{}$ ".format(int(gamma_exponent)),
                )
        else:
            (line,) = ax.plot(
                rmse_bagging[i, :],
                rmse_PA_bagging[i, :],
                ".-",
                label="$\\gamma = {}$ ".format(gamma_vals[i]),
            )
                
        ax.plot(
            rmse_bagging[i, 0],
            rmse_PA_bagging[i, 0],
            color=line.get_color(),
            marker="o",
            markersize=8,
        )
    # plot ideal vector
    #     if len(h_anchors) == 0:
    
    ''' First, find the most common optimal pair (gamma, lambda)
    across the different bootstraps and plot the mean RMSE/RMSE_PA for that pair.
    It might not correspond to the min RMSE/RMSE_PA.
    This is done in the same way (using mode) for the ideal vector, 
    the optimal ridge and the optimal anchor solutions.'''
    ideal_vector, = ax.plot(
        rmse_bagging[
            iv1[0].astype(int),
            iv1[1].astype(int),
        ],
        rmse_PA_bagging[
            iv2[0].astype(int),
            iv2[1].astype(int),
        ],
        "k^",
        markersize=7,
        # label="Ideal vector",
    )
    ridge_optimal, = ax.plot(
        rmse_bagging[0, i2_ridge],
        rmse_PA_bagging[0, i2_ridge],
        "ks",
        markersize=7,
        # label="Ridge (optimal)",
    )
    anchor_optimal, = ax.plot(
        rmse_bagging[i1, i2],
        rmse_PA_bagging[i1, i2],
        "ko",
        markersize=7,
        # label="Anchor (optimal)",
    )

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
    #     ax.set_xticks(ticks)
    #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(
        "Root mean squared error ($RMSE$) (" u"\N{DEGREE SIGN}C)",
        fontsize=params["axes.labelsize"],
    )
    ax.set_ylabel(
        "$RMSE_{\Pi_{\mathbf{A}}}$ (error explained by $A$)",
        fontsize=params["axes.labelsize"],
    )

    legend1 = ax.legend(fontsize=params["axes.labelsize"] - 2, loc = "upper right")
    legend2 = ax.legend([ideal_vector,ridge_optimal, anchor_optimal],['Ideal vector','Ridge (optimal)', 'Anchor (optimal)'], loc='upper left', fontsize=params["axes.labelsize"] - 2)
    ax.add_artist(legend1)

    label_panel(ax, "E")

    ax = fig.add_subplot(spec[5:8, 5:10])
    if len(h_anchors) == 0: # plot correlation
        for i in range(rmse_bagging.shape[0]):
            (line,) = ax.plot(
                rmse_bagging[i, :],
                corr_bagging[i, :],
                ".-",
                label="$\\gamma = $ " + str(gamma_vals[i]),
            )
            ax.plot(
                rmse_bagging[i, 0],
                corr_bagging[i, 0],
                color=line.get_color(),
                marker="o",
                markersize=8,
            )
        ax.plot(
            rmse_bagging[i1, i2],
            corr_bagging[i1, i2], # corr for linear anchor and mi for nonlinear anchor
            "ko",
            markersize=7,
        )
        ax.plot(
            rmse_bagging[0, i2_ridge],
            corr_bagging[0, i2_ridge],
            "ks",
            markersize=7,
        )
        #     ax.set_yscale("log")
        #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
        #     ax.set_xticks(ticks)
        #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(
            "Root mean squared error ($RMSE$) (" u"\N{DEGREE SIGN}C)",
            fontsize=params["axes.labelsize"],
        )
        ax.set_ylabel(
            "Linear correlation (residuals, $A$)",
            fontsize=params["axes.labelsize"],
        )
    else: # plot mutual information
        for i in range(rmse_bagging.shape[0]):
            (line,) = ax.plot(
                rmse_bagging[i, :],
                mi_bagging[i, :],
                ".-",
                label="$\\gamma = $ " + str(gamma_vals[i]),
            )
            ax.plot(
                rmse_bagging[i, 0],
                mi_bagging[i, 0],
                color=line.get_color(),
                marker="o",
                markersize=8,
            )
        ax.plot(
            rmse_bagging[i1, i2],
            mi_bagging[i1, i2], 
            "ko",
            markersize=7,
        )
        ax.plot(
            rmse_bagging[0, i2_ridge],
            mi_bagging[0, i2_ridge],
            "ks",
            markersize=7,
        )
        #     ax.set_yscale("log")
        #     ticks = [0.6, 0.8, 1, 2, 3, 4, 5, 6, 7]
        #     ax.set_xticks(ticks)
        #     ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(
            "Root mean squared error ($RMSE$) (" u"\N{DEGREE SIGN}C)",
            fontsize=params["axes.labelsize"],
        )
        ax.set_ylabel(
            "Mutual information (residuals, $A$)",
            fontsize=params["axes.labelsize"],
        )

    # plot ideal vector
    #     if len(h_anchors) == 1:
    # ax.plot(
    #     rmse_bagging[
    #         iv1[0].astype(int),
    #         iv1[1].astype(int),
    #     ],
    #     corr_bagging[iv2[0].astype(int), iv2[1]].astype(int),
    #     "k^", color = "blue",
    #     markersize=7,
    # )
    label_panel(ax, "F")

    """ Plot HT results """
    bins = np.linspace(0, 1, 20)
    ax = fig.add_subplot(spec[8:10, 0:5])
    ax.hist(
        alpha_bagging.reshape(-1),
        bins,
        fill=True,
        alpha=0.8,
        linewidth=0,
        color="steelblue",
        label="Type I error",
    )
    ax.legend(fontsize=params["axes.labelsize"])
    label_panel(ax, "G")

    ax = fig.add_subplot(spec[8:10, 5:10])
    ax.hist(
        power_bagging.reshape(-1),
        bins,
        fill=True,
        alpha=0.8,
        linewidth=0,
        color="brown",
        label="Power of test",
    )
    ax.legend(fontsize=params["axes.labelsize"])
    label_panel(ax, "H")

    #     axes = [fig.add_subplot(gs[i]) for i in range(3)]

    #     for ax, label in zip(axes, axis_labels):
    #     bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    #     fig.text(bbox.x0, bbox.y1, axis_label, fontsize=12, fontweight="bold", va="top", ha="left",
    #              transform=None)

    #     label_panel(ax1,'A')
    #     label_panel(ax2,'B')
    #     label_panel(ax3,'C')
    #     label_panel(ax4,'D')

    plt.subplots_adjust(wspace=5, hspace=2)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig(
            "./../output/figures/" + filename,
            bbox_inches="tight",
            format=formatFig,
        )


def plot_Pareto_corr(
    rmse,
    corr,
    corr2,
    h_anchors,
    gamma_vals,
    lambda_vals,
    ind_gamma_opt,
    ind_lambda_opt,
    ind_lambda_opt_ridge,
    ind_vect_ideal_obj1,
    ind_vect_ideal_obj2,
    filename=None,
):

    rmse_bagging = np.mean(rmse, axis=0)
    corr_bagging = np.mean(corr, axis=0)
    corr2_bagging = np.mean(corr2, axis=0)

    i1 = int(mode(list(ind_gamma_opt.reshape(-1))))
    i2 = int(mode(list(ind_lambda_opt.reshape(-1))))
    i2_ridge = int(mode(list(ind_lambda_opt_ridge.reshape(-1))))
    iv1 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj1[:, 0]))),
            int(mode(list(ind_vect_ideal_obj1[:, 1]))),
        ]
    )
    iv2 = np.array(
        [
            int(mode(list(ind_vect_ideal_obj2[:, 0]))),
            int(mode(list(ind_vect_ideal_obj2[:, 1]))),
        ]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    #     _, _, vect_ideal, vect_Nadir = choose_gamma_lambda_pareto(
    #         rmse, mi, maxX=False, maxY=False,)

    for i in range(rmse_bagging.shape[0]):
        (line,) = ax1.plot(
            rmse_bagging[i, :],
            corr2_bagging[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax1.plot(
            rmse_bagging[i, 0],
            corr2_bagging[i, 0],
            color=line.get_color(),
            marker="o",
            markersize=10,
        )
    ax1.plot(
        rmse_bagging[i1, i2],
        corr2_bagging[i1, i2],
        "ks",
    )
    ax1.plot(
        rmse_bagging[
            iv1[0].astype(int),
            iv1[1].astype(int),
        ],
        corr2_bagging[
            iv2[0].astype(int),
            iv2[1].astype(int),
        ],
        "k^",
        markersize=8,
        label="Ideal vector",
    )
    ax1.plot(
        rmse_bagging[0, i2_ridge],
        corr2_bagging[0, i2_ridge],
        "ko",
        markersize=9,
        label="Ridge (optimal)",
    )
    ax1.plot(
        rmse_bagging[i1, i2],
        corr2_bagging[i1, i2],
        "ks",
        markersize=8,
        label="Anchor (optimal)",
    )

    #     ax1.set_xscale("log")
    #     ticks = [0.5, 0.6, 0.8, 1, 1.5, 2]
    #     ax1.set_xticks(ticks)
    #     ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel(
        "Root mean squared error (RMSE)  (" u"\N{DEGREE SIGN}C)",
        fontsize=params["axes.labelsize"],
    )
    ax1.set_ylabel(
        "Correlation of residuals with $A^2$",
        fontsize=params["axes.labelsize"],
    )
    ax1.legend(fontsize=params["axes.labelsize"] - 4)

    for i in range(rmse_bagging.shape[0]):
        (line,) = ax2.plot(
            corr_bagging[i, :],
            corr2_bagging[i, :],
            ".-",
            label="$\\gamma = $ " + str(gamma_vals[i]),
        )
        ax2.plot(
            corr_bagging[i, 0],
            corr2_bagging[i, 0],
            color=line.get_color(),
            marker="o",
            markersize=10,
        )
    ax2.plot(
        corr_bagging[i1, i2],
        corr2_bagging[i1, i2],
        "ks",
    )
    # plot ideal vector
    ax2.plot(
        corr_bagging[
            iv1[0].astype(int),
            iv1[1].astype(int),
        ],
        corr2_bagging[iv2[0].astype(int), iv2[1]].astype(int),
        "k^",
        markersize=8,
    )
    ax2.plot(
        corr_bagging[0, i2_ridge],
        corr2_bagging[0, i2_ridge],
        "ko",
        markersize=9,
    )
    ax2.plot(corr_bagging[i1, i2], corr2_bagging[i1, i2], "ks", markersize=8)

    #     ax2.set_xscale("log")
    #     ticks = [0.5, 0.6, 0.8, 1, 1.5, 2]
    #     ax2.set_xticks(ticks)
    #     ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel(
        "Linear correlation (residuals, anchor)",
        fontsize=params["axes.labelsize"],
    )
    ax2.set_ylabel(
        "Correlation of residuals with $A^2$",
        fontsize=params["axes.labelsize"],
    )

    plt.subplots_adjust(wspace=0.5)

    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, bbox_inches="tight")


def label_panel(
    ax,
    letter,
    *,
    offset_left=0.6,
    offset_up=0.2,
    prefix="",
    postfix="",
    **font_kwds
):
    kwds = dict(fontsize=19, fontweight="heavy")
    kwds.update(font_kwds)
    # this mad looking bit of code says that we should put the code offset a certain distance in
    # inches (using the fig.dpi_scale_trans transformation) from the top left of the frame
    # (which is (0, 1) in ax.transAxes transformation space)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(
        -offset_left, offset_up, fig.dpi_scale_trans
    )
    ax.text(0, 1, prefix + letter + postfix, transform=trans, **kwds)


def plot_motiv_example():
    
    params["axes.titlesize"] = 23
    params["axes.labelsize"] = 23
    plt.rcParams.update(params)
    
    overview_NOIV = pd.read_csv(
        "./../data/local/motivatingExample/00_overview_NOIV.txt",
        sep=";",
        header=0,
    )
    overview_trainIV = pd.read_csv(
        "./../data/local/motivatingExample/00_overview_trainIV.txt",
        sep=";",
        header=0,
    )
    overview_testIV = pd.read_csv(
        "./../data/local/motivatingExample/00_overview_testIV.txt",
        sep=";",
        header=0,
    )

    pred_cntl = pd.read_csv(
        "./../data/local/motivatingExample/01_pred_cntl.txt", sep=";", header=0
    )
    pred_coldsun = pd.read_csv(
        "./../data/local/motivatingExample/01_pred_coldsun.txt",
        sep=";",
        header=0,
    )
    pred_warmsun = pd.read_csv(
        "./../data/local/motivatingExample/01_pred_warmsun.txt",
        sep=";",
        header=0,
    )

    fig, axes = plt.subplots(3, 3, figsize=(18.5, 18.5))
    #     fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
    #                               constrained_layout=True)

    ax_let_left = -0.11
    ax_let_up = 1.05
    
    """ a. No Interventions """
    ax = axes[0, 0]
    ax.plot(
        overview_NOIV[overview_NOIV["scen"] == "cntl"]["year"],
        overview_NOIV[overview_NOIV["scen"] == "cntl"]["T_glm"],
        color="darkorange",
        label="Control runs",
    )

    start_years = np.unique(
        overview_NOIV[overview_NOIV["scen"] == "1perCO2"]["start.year"]
    )
    for i in range(len(start_years)):
        tmp = overview_NOIV[
            (overview_NOIV["scen"] == "1perCO2")
            & (overview_NOIV["start.year"] == start_years[i])
        ]
        if i == 0:
            ax.plot(
                tmp["year"],
                tmp["T_glm"],
                color="dimgrey",
                label=r"1\%/yr CO2 increase",
            )
        else:
            ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    ax.set_ylim([-5, 10])
    ax.set_yticks([-5, 0, 5, 10])
    ax.legend(fontsize=params["axes.labelsize"] - 2)
    ax.set_xlabel("Year", fontsize=params["axes.labelsize"])
    ax.set_ylabel(
        "Global Mean Temp Anomaly (°C)",
        fontsize=params["axes.labelsize"],
    )
    ax.set_title("No Interventions")
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(a)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    """ b. TRAINING DATA """
    ax = axes[0, 1]
    ax.plot(
        overview_trainIV[overview_trainIV["scen"] == "cntl"]["year"],
        overview_trainIV[overview_trainIV["scen"] == "cntl"]["T_glm"],
        color="darkorange",
    )
    ax.plot(
        overview_trainIV[overview_trainIV["scen"] == "plus6W"]["year"],
        overview_trainIV[overview_trainIV["scen"] == "plus6W"]["T_glm"],
        color="red",
        label="warm sun ($+6$Wm$^{-2}$)",
    )
    ax.plot(
        overview_trainIV[overview_trainIV["scen"] == "minus6W"]["year"],
        overview_trainIV[overview_trainIV["scen"] == "minus6W"]["T_glm"],
        color="lightblue",
        label="cool sun ($-6$Wm$^{-2}$)",
    )

    start_years = np.unique(
        overview_trainIV[overview_trainIV["scen"] == "1perCO2"]["start.year"]
    )
    for i in range(len(start_years)):
        tmp = overview_trainIV[
            (overview_trainIV["scen"] == "1perCO2")
            & (overview_trainIV["start.year"] == start_years[i])
        ]
        ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    start_years = np.unique(
        overview_trainIV[overview_trainIV["scen"] == "plus6W.1perCO2"][
            "start.year"
        ]
    )
    for i in range(len(start_years)):
        tmp = overview_trainIV[
            (overview_trainIV["scen"] == "plus6W.1perCO2")
            & (overview_trainIV["start.year"] == start_years[i])
        ]
        ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    start_years = np.unique(
        overview_trainIV[overview_trainIV["scen"] == "minus6W.1perCO2"][
            "start.year"
        ]
    )
    for i in range(len(start_years)):
        tmp = overview_trainIV[
            (overview_trainIV["scen"] == "minus6W.1perCO2")
            & (overview_trainIV["start.year"] == start_years[i])
        ]
        ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    ax.set_ylim([-5, 10])
    ax.set_yticks([-5, 0, 5, 10])
    ax.legend(fontsize=params["axes.labelsize"] - 2)
    ax.set_xlabel("Year", fontsize=params["axes.labelsize"])
    ax.set_ylabel(
        "Global Mean Temp Anomaly (°C)",
        fontsize=params["axes.labelsize"],
    )
    ax.set_title("Small Interventions (Training data)")
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(b)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    """ c. TESTING DATA """
    ax = axes[0, 2]
    # only plot 2000 years for control runs
    ax.plot(
        overview_testIV[overview_testIV["scen"] == "cntl"]["year"].iloc[:2000],
        overview_testIV[overview_testIV["scen"] == "cntl"]["T_glm"].iloc[
            :2000
        ],
        color="darkorange",
    )
    ax.plot(
        overview_testIV[overview_testIV["scen"] == "plus25W"]["year"],
        overview_testIV[overview_testIV["scen"] == "plus25W"]["T_glm"],
        color="darkred",
        label="hot sun ($+25$Wm$^{-2}$)",
    )
    ax.plot(
        overview_testIV[overview_testIV["scen"] == "minus25W"]["year"],
        overview_testIV[overview_testIV["scen"] == "minus25W"]["T_glm"],
        color="darkblue",
        label="cold sun ($-25$Wm$^{-2}$)",
    )

    start_years = np.unique(
        overview_testIV[overview_testIV["scen"] == "1perCO2"]["start.year"]
    )
    for i in range(len(start_years)):
        tmp = overview_testIV[
            (overview_testIV["scen"] == "1perCO2")
            & (overview_testIV["start.year"] == start_years[i])
        ]
        ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    start_years = np.unique(
        overview_testIV[overview_testIV["scen"] == "plus25W.1perCO2"][
            "start.year"
        ]
    )
    for i in range(len(start_years)):
        tmp = overview_testIV[
            (overview_testIV["scen"] == "plus25W.1perCO2")
            & (overview_testIV["start.year"] == start_years[i])
        ]
        ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    start_years = np.unique(
        overview_testIV[overview_testIV["scen"] == "minus25W.1perCO2"][
            "start.year"
        ]
    )
    for i in range(len(start_years)):
        tmp = overview_testIV[
            (overview_testIV["scen"] == "minus25W.1perCO2")
            & (overview_testIV["start.year"] == start_years[i])
        ]
        ax.plot(tmp["year"], tmp["T_glm"], color="dimgrey")

    #     ax.set_xlim([500, 2500])
    ax.set_ylim([-5, 10])
    ax.set_yticks([-5, 0, 5, 10])
    ax.legend(fontsize=params["axes.labelsize"] - 2)
    ax.set_xlabel("Year", fontsize=params["axes.labelsize"])
    ax.set_ylabel(
        "Global Mean Temp Anomaly (°C)",
        fontsize=params["axes.labelsize"],
    )
    ax.set_title("Strong Interventions (Test data)")
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(c)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    """ ii. Plot predictions and residuals """
    flim = [-5, 15]

    """NO INTERVENTIONS"""
    ax = axes[1, 0]
    ax.plot(
        pred_cntl["y_cntl"], pred_cntl["yhat_cntl_A1"], ".", color="darkorange"
    )
    x = np.linspace(-5, 15, 100)
    ax.plot(x, x, "darkgrey", linestyle="solid")
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_xlabel("True CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_ylabel("Predicted CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_title("No Interventions")
    rmse = np.round(
        np.sqrt(
            np.mean((pred_cntl["y_cntl"] - pred_cntl["yhat_cntl_A1"]) ** 2)
        ),
        2,
    )
    corr = np.round(
        np.corrcoef(pred_cntl["y_cntl"], pred_cntl["yhat_cntl_A1"])[0, 1], 2
    )
    ax.text(
        5,
        13.5,
        "$\\gamma = 1$ (RMSE = "
        + str(rmse)
        + ", $\\rho$ = "
        + str(corr)
        + ")",
        ha="center",
        va="center",
        rotation=0,
        size=20,
        bbox=dict(boxstyle="round, pad=0.4", fc="white", ec="lightgrey", lw=1),
    )
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(d)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    ax = axes[2, 0]
    ax.plot(
        pred_cntl["y_cntl"],
        pred_cntl["yhat_cntl_A100"],
        ".",
        color="darkorange",
    )
    x = np.linspace(-5, 15, 100)
    ax.plot(x, x, "darkgrey", linestyle="solid")
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_xlabel("True CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_ylabel("Predicted CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_title("No Interventions")
    rmse = np.round(
        np.sqrt(
            np.mean((pred_cntl["y_cntl"] - pred_cntl["yhat_cntl_A100"]) ** 2)
        ),
        2,
    )
    corr = np.round(
        np.corrcoef(pred_cntl["y_cntl"], pred_cntl["yhat_cntl_A100"])[0, 1], 2
    )
    ax.text(
        5,
        13.5,
        "$\\gamma = 100$ (RMSE = "
        + str(rmse)
        + ", $\\rho$ = "
        + str(corr)
        + ")",
        ha="center",
        va="center",
        rotation=0,
        size=20,
        bbox=dict(boxstyle="round, pad=0.4", fc="white", ec="lightgrey", lw=1),
    )
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(g)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    """STRONG INTERVENTIONS"""

    ax = axes[1, 1]
    ax.plot(
        pred_cntl["y_cntl"], pred_cntl["yhat_cntl_A1"], ".", color="darkorange"
    )
    ax.plot(
        pred_warmsun["y_warmsun"],
        pred_warmsun["yhat_warmsun_A1"],
        ".",
        color="darkred",
    )
    ax.plot(
        pred_coldsun["y_coldsun"],
        pred_coldsun["yhat_coldsun_A1"],
        ".",
        color="darkblue",
    )
    x = np.linspace(-5, 15, 100)
    ax.plot(x, x, "darkgrey", linestyle="solid")
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_xlabel("True CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_ylabel("Predicted CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_title("Strong Interventions (Prediction)")
    y_true = pd.concat(
        [
            pred_cntl["y_cntl"],
            pred_warmsun["y_warmsun"],
            pred_coldsun["y_coldsun"],
        ]
    )
    yhat_pred = pd.concat(
        [
            pred_cntl["yhat_cntl_A1"],
            pred_warmsun["yhat_warmsun_A1"],
            pred_coldsun["yhat_coldsun_A1"],
        ]
    )
    rmse = np.round(np.sqrt(np.mean((y_true - yhat_pred) ** 2)), 2)
    corr = np.round(np.corrcoef(y_true, yhat_pred)[0, 1], 2)
    ax.text(
        5,
        13.5,
        "$\\gamma = 1$ (RMSE = "
        + str(rmse)
        + ", $\\rho$ = "
        + str(corr)
        + ")",
        ha="center",
        va="center",
        rotation=0,
        size=20,
        bbox=dict(boxstyle="round, pad=0.4", fc="white", ec="lightgrey", lw=1),
    )
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(e)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    ax = axes[2, 1]
    ax.plot(
        pred_cntl["y_cntl"],
        pred_cntl["yhat_cntl_A100"],
        ".",
        color="darkorange",
    )
    ax.plot(
        pred_warmsun["y_warmsun"],
        pred_warmsun["yhat_warmsun_A100"],
        ".",
        color="darkred",
    )
    ax.plot(
        pred_coldsun["y_coldsun"],
        pred_coldsun["yhat_coldsun_A100"],
        ".",
        color="darkblue",
    )
    x = np.linspace(-5, 15, 100)
    ax.plot(x, x, "darkgrey", linestyle="solid")
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_yticks([-5, 0, 5, 10, 15])
    ax.set_xlabel("True CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_ylabel("Predicted CO$_2$ forcing (Wm$^{-2}$)")
    ax.set_title("Strong Interventions (Prediction)")
    y_true = pd.concat(
        [
            pred_cntl["y_cntl"],
            pred_warmsun["y_warmsun"],
            pred_coldsun["y_coldsun"],
        ]
    )
    yhat_pred = pd.concat(
        [
            pred_cntl["yhat_cntl_A100"],
            pred_warmsun["yhat_warmsun_A100"],
            pred_coldsun["yhat_coldsun_A100"],
        ]
    )
    rmse = np.round(np.sqrt(np.mean((y_true - yhat_pred) ** 2)), 2)
    corr = np.round(np.corrcoef(y_true, yhat_pred)[0, 1], 2)
    ax.text(
        5,
        13.5,
        "$\\gamma = 100$ (RMSE = "
        + str(rmse)
        + ", $\\rho$ = "
        + str(corr)
        + ")",
        ha="center",
        va="center",
        rotation=0,
        size=20,
        bbox=dict(boxstyle="round, pad=0.4", fc="white", ec="lightgrey", lw=1),
    )
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(h)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    """STRONG INTERVENTIONS, residual correlations"""
    ax = axes[1, 2]
    ax.plot(
        np.repeat(0, 641),
        pred_cntl["y_cntl"] - pred_cntl["yhat_cntl_A1"],
        ".",
        color="darkorange",
    )
    ax.plot(
        np.repeat(25, 1423),
        pred_warmsun["y_warmsun"] - pred_warmsun["yhat_warmsun_A1"],
        ".",
        color="darkred",
    )
    ax.plot(
        np.repeat(-25, 1413),
        pred_coldsun["y_coldsun"] - pred_coldsun["yhat_coldsun_A1"],
        ".",
        color="darkblue",
    )
    ax.set_xlim([-36, 35])
    ax.set_ylim([-6, 6])
    ax.set_xlabel("Solar anchor (Wm$^{-2}$)")
    ax.set_ylabel("Prediction residuals")
    ax.set_title(
        "Strong Interventions \n (Residual correlation w/ Anchor)",
    )
    res = pd.concat(
        [
            pred_cntl["y_cntl"] - pred_cntl["yhat_cntl_A1"],
            pred_warmsun["y_warmsun"] - pred_warmsun["yhat_warmsun_A1"],
            pred_coldsun["y_coldsun"] - pred_coldsun["yhat_coldsun_A1"],
        ]
    )
    anchor = pd.concat(
        [
            pd.Series(np.repeat(0, 641)),
            pd.Series(np.repeat(25, 1423)),
            pd.Series(np.repeat(-25, 1413)),
        ]
    )
    corr = np.round(np.corrcoef(res, anchor)[0, 1], 2)
    ax.text(
        0,
        5.1,
        "$\\gamma = 1$ ($\\rho$ = " + str(corr) + ")",
        ha="center",
        va="center",
        rotation=0,
        size=20,
        bbox=dict(boxstyle="round, pad=0.4", fc="white", ec="lightgrey", lw=1),
    )
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(f)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    ax = axes[2, 2]
    ax.plot(
        np.repeat(0, 641),
        pred_cntl["y_cntl"] - pred_cntl["yhat_cntl_A100"],
        ".",
        color="darkorange",
    )
    ax.plot(
        np.repeat(25, 1423),
        pred_warmsun["y_warmsun"] - pred_warmsun["yhat_warmsun_A100"],
        ".",
        color="darkred",
    )
    ax.plot(
        np.repeat(-25, 1413),
        pred_coldsun["y_coldsun"] - pred_coldsun["yhat_coldsun_A100"],
        ".",
        color="darkblue",
    )
    ax.set_xlim([-36, 35])
    ax.set_ylim([-6, 6])
    ax.set_xlabel("Solar anchor (Wm$^{-2}$)")
    ax.set_ylabel("Prediction residuals")
    ax.set_title("Strong Interventions \n (Residual correlation w/ Anchor)")
    res = pd.concat(
        [
            pred_cntl["y_cntl"] - pred_cntl["yhat_cntl_A100"],
            pred_warmsun["y_warmsun"] - pred_warmsun["yhat_warmsun_A100"],
            pred_coldsun["y_coldsun"] - pred_coldsun["yhat_coldsun_A100"],
        ]
    )
    anchor = pd.concat(
        [
            pd.Series(np.repeat(0, 641)),
            pd.Series(np.repeat(25, 1423)),
            pd.Series(np.repeat(-25, 1413)),
        ]
    )
    corr = np.round(np.corrcoef(res, anchor)[0, 1], 2)
    ax.text(
        0,
        5.1,
        "$\\gamma = 100$ ($\\rho$ = " + str(corr) + ")",
        ha="center",
        va="center",
        rotation=0,
        size=20,
        bbox=dict(boxstyle="round, pad=0.4", fc="white", ec="lightgrey", lw=1),
    )
    trans = mtransforms.ScaledTranslation(
        -20 / 72, 7 / 72, fig.dpi_scale_trans
    )
    ax.text(
        ax_let_left,
        ax_let_up,
        "(i)",
        transform=ax.transAxes + trans,
        fontsize=params["axes.titlesize"],
        va="bottom",
        fontfamily="serif",
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    filename = "motivating_example.pdf"
    if filename:
        if not os.path.isdir("./../output/figures/"):
            os.makedirs("./../output/figures/")
        fig.savefig("./../output/figures/" + filename, format = "pdf", bbox_inches="tight")

    plt.show()
