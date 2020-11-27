import numpy as np
import pandas as pd
import random

from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from robustDA.process_cmip6 import (
    read_forcing_cmip6,
    compute_forced_response_cmip6,
)
from robustDA.utils.helpers import partition


""" Split the data into training and testing """


def split_train_test(
    modelsDataList, modelsInfoFrame, target, anchor, percTrain=0.5
):

    forcings = [
        "total",
        "total_anthropogenic",
        "total_natural",
        "GHG",
        "co2",
        "aerosols",
        "volcanic",
        "solar",
    ]
    forcedResponses = ["hist-aer", "hist-GHG", "hist-nat"]
    nbFiles = len(modelsDataList)

    """
    Check if the target is a forcing or a forced response
    If target is a forcing, use all models,
    as the forcing is available regardless of model.
    If target is a forced response,
    use only models that have the correponding forced response.
    Use modelFull for computing the forced response, not just model.
    """
    if target in forcings:  # sorted not needed
        models = sorted(set([modelsDataList[i].model for i in range(nbFiles)]))

        """ Split the models """
        trainModels = np.sort(
            list(random.sample(models, int(np.round(percTrain * len(models)))))
        )

        trainFiles = [
            modelsDataList[i].filename
            for i in range(nbFiles)
            if modelsDataList[i].model in trainModels
        ]
        trainDataList = [
            modelsDataList[i].data
            for i in range(nbFiles)
            if modelsDataList[i].model in trainModels
        ]

        testModels = np.sort(list(set(models) - set(trainModels)))

        testFiles = [
            modelsDataList[i].filename
            for i in range(nbFiles)
            if modelsDataList[i].model in testModels
        ]
        testDataList = [
            modelsDataList[i].data
            for i in range(nbFiles)
            if modelsDataList[i].model in testModels
        ]

    elif target in forcedResponses:
        """Read which runs are available
        for the specified forced response
        """
        if target == "hist-aer":
            forcedResponse_df = pd.read_csv(
                "./../data/local/forcedResponse/"
                "CMIP6_forcedResponse_hist-aer.csv",
                index_col="Year",
            )
        elif target == "hist-GHG":
            forcedResponse_df = pd.read_csv(
                "./../data/local/forcedResponse/"
                "CMIP6_forcedResponse_hist-GHG.csv",
                index_col="Year",
            )

        modelsFull_forcedResponse = list(forcedResponse_df.columns)
        models = sorted(
            set(
                [
                    modelsDataList[i].model
                    for i in range(nbFiles)
                    if modelsDataList[i].modelFull in modelsFull_forcedResponse
                ]
            )
        )

        """ Split the models """
        trainModels = np.sort(
            list(random.sample(models, int(np.round(percTrain * len(models)))))
        )
        trainFiles = [
            modelsDataList[i].filename
            for i in range(nbFiles)
            if modelsDataList[i].model in trainModels
            and modelsDataList[i].modelFull in modelsFull_forcedResponse
        ]
        trainDataList = [
            modelsDataList[i].data
            for i in range(nbFiles)
            if modelsDataList[i].model in trainModels
            and modelsDataList[i].modelFull in modelsFull_forcedResponse
        ]

        testModels = np.sort(list(set(models) - set(trainModels)))
        testFiles = [
            modelsDataList[i].filename
            for i in range(nbFiles)
            if modelsDataList[i].model in testModels
            and modelsDataList[i].modelFull in modelsFull_forcedResponse
        ]
        testDataList = [
            modelsDataList[i].data
            for i in range(nbFiles)
            if modelsDataList[i].model in testModels
            and modelsDataList[i].modelFull in modelsFull_forcedResponse
        ]

    """ TRAINING DATA """
    X_train_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_anchor_train_df = pd.DataFrame([])

    for i in range(len(trainFiles)):
        """ Build data X """
        X_tmp = pd.DataFrame(trainDataList[i])
        X_train_df = pd.concat([X_train_df, X_tmp], axis=0)

        ind_tmp = int(
            modelsInfoFrame[
                modelsInfoFrame["filename"] == trainFiles[i]
            ].index.values
        )
        startDate, endDate = (
            modelsDataList[ind_tmp].data.index[0],
            modelsDataList[ind_tmp].data.index[-1],
        )

        """ Build the target y """
        if target in forcings:
            y_tmp = read_forcing_cmip6(
                modelsDataList[ind_tmp].scenario, target, startDate, endDate
            )
        elif target in forcedResponses:
            if modelsDataList[ind_tmp].scenario != "piControl":
                y_tmp = forcedResponse_df[modelsDataList[ind_tmp].modelFull]
            elif modelsDataList[ind_tmp].scenario == "piControl":
                y_tmp = pd.Series(
                    np.zeros(endDate - startDate + 1),
                    index=np.linspace(
                        startDate, endDate, endDate - startDate + 1
                    ),
                )
                y_tmp.index.name = "Year"

        y_train_df = pd.concat([y_train_df, y_tmp], axis=0)

        """ Build the anchor """
        if anchor in forcings:
            tmp_anchor = read_forcing_cmip6(
                modelsDataList[ind_tmp].scenario, anchor, startDate, endDate
            )
            y_anchor_train_df = pd.concat([y_anchor_train_df, tmp_anchor])

        elif anchor in forcedResponses:
            if anchor == "hist-GHG":
                forcedResponse_df = pd.read_csv(
                    "./../data/local/forcedResponse/"
                    "CMIP6_forcedResponse_hist-GHG.csv",
                    index_col="Year",
                )

                y_hist_GHG = []
                for i in range(len(trainFiles)):
                    ind_tmp = int(
                        modelsInfoFrame[
                            modelsInfoFrame["filename"] == trainFiles[i]
                        ].index.values
                    )
                    y_tmp = forcedResponse_df[
                        modelsDataList[ind_tmp].modelFull
                    ]
                    y_hist_GHG = np.concatenate([y_hist_GHG, y_tmp], axis=0)

                y_hist_GHG = y_hist_GHG.reshape(-1, 1)
                sc_y_hist_GHG = StandardScaler(with_mean=True, with_std=True)
                y_hist_GHG_std = sc_y_hist_GHG.fit_transform(y_hist_GHG)
                y_anchor_train_df = y_hist_GHG_std

            elif anchor == "hist-aer":
                forcedResponse_df = pd.read_csv(
                    "./../data/local/forcedResponse/"
                    "CMIP6_forcedResponse_hist-aer.csv",
                    index_col="Year",
                )

                y_hist_aer = []
                for i in range(len(trainFiles)):
                    ind_tmp = int(
                        modelsInfoFrame[
                            modelsInfoFrame["filename"] == trainFiles[i]
                        ].index.values
                    )
                    y_tmp = forcedResponse_df[
                        modelsDataList[ind_tmp].modelFull
                    ]
                    y_hist_aer = np.concatenate([y_hist_aer, y_tmp], axis=0)

                y_hist_aer = y_hist_aer.reshape(-1, 1)
                sc_y_hist_aer = StandardScaler(with_mean=True, with_std=True)
                y_hist_aer_std = sc_y_hist_aer.fit_transform(y_hist_aer)
                y_anchor_train_df = y_hist_aer_std

    """ TESTING DATA """
    X_test_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    y_anchor_test_df = pd.DataFrame([])

    for i in range(len(testFiles)):
        X_tmp = pd.DataFrame(testDataList[i])
        X_test_df = pd.concat([X_test_df, X_tmp], axis=0)

        ind_tmp = int(
            modelsInfoFrame[
                modelsInfoFrame["filename"] == testFiles[i]
            ].index.values
        )
        startDate, endDate = (
            modelsDataList[ind_tmp].data.index[0],
            modelsDataList[ind_tmp].data.index[-1],
        )

        if target in forcings:
            y_tmp = read_forcing_cmip6(
                modelsDataList[ind_tmp].scenario, target, startDate, endDate
            )
        elif target in forcedResponses:
            if modelsDataList[ind_tmp].scenario != "piControl":
                y_tmp = forcedResponse_df[modelsDataList[ind_tmp].modelFull]
            elif modelsDataList[ind_tmp].scenario == "piControl":
                y_tmp = pd.Series(
                    np.zeros(endDate - startDate + 1),
                    index=np.linspace(
                        startDate, endDate, endDate - startDate + 1
                    ),
                )
                y_tmp.index.name = "Year"

        y_test_df = pd.concat([y_test_df, y_tmp], axis=0)

        """ Build the anchor """
        if anchor in forcings:
            tmp_anchor = read_forcing_cmip6(
                modelsDataList[ind_tmp].scenario, anchor, startDate, endDate
            )
            y_anchor_test_df = pd.concat([y_anchor_test_df, tmp_anchor])

        elif anchor in forcedResponses:
            if anchor == "hist-GHG":
                forcedResponse_df = pd.read_csv(
                    "./../data/local/forcedResponse/"
                    "CMIP6_forcedResponse_hist-GHG.csv",
                    index_col="Year",
                )

                y_hist_GHG_test = []
                for i in range(len(testFiles)):
                    ind_tmp = int(
                        modelsInfoFrame[
                            modelsInfoFrame["filename"] == testFiles[i]
                        ].index.values
                    )
                    y_tmp = forcedResponse_df[
                        modelsDataList[ind_tmp].modelFull
                    ]
                    y_hist_GHG_test = np.concatenate(
                        [y_hist_GHG_test, y_tmp], axis=0
                    )

                y_hist_GHG_test = y_hist_GHG_test.reshape(-1, 1)
                sc_y_hist_GHG_test = StandardScaler(
                    with_mean=True, with_std=True
                )
                y_hist_GHG_test_std = sc_y_hist_GHG_test.fit_transform(
                    y_hist_GHG_test
                )
                y_anchor_test_df = y_hist_GHG_test_std

                forcedResponse_df = pd.read_csv(
                    "./../data/local/forcedResponse/"
                    "CMIP6_forcedResponse_hist-aer.csv",
                    index_col="Year",
                )

                y_hist_aer_test = []
                for i in range(len(testFiles)):
                    ind_tmp = int(
                        modelsInfoFrame[
                            modelsInfoFrame["filename"] == testFiles[i]
                        ].index.values
                    )
                    y_tmp = forcedResponse_df[
                        modelsDataList[ind_tmp].modelFull
                    ]
                    y_hist_aer_test = np.concatenate(
                        [y_hist_aer_test, y_tmp], axis=0
                    )

                y_hist_aer_test = y_hist_aer_test.reshape(-1, 1)
                sc_y_hist_aer_test = StandardScaler(
                    with_mean=True, with_std=True
                )
                y_hist_aer_test_std = sc_y_hist_aer_test.fit_transform(
                    y_hist_aer_test
                )
                y_anchor_test_df = y_hist_aer_test_std

    dict_models = {
        "trainModels": trainModels,
        "testModels": testModels,
        "trainFiles": trainFiles,
        "testFiles": testFiles,
        "X_train": X_train_df,
        "y_train": y_train_df,
        "y_anchor_train": y_anchor_train_df,
        "X_test": X_test_df,
        "y_test": y_test_df,
        "y_anchor_test": y_anchor_test_df,
    }

    return dict_models


def split_folds_CV(
    modelsDataList,
    modelsInfoFrame,
    dict_models,
    target,
    anchor,
    nbFoldsCV=5,
    displayModels=False,
):
    forcings = [
        "total",
        "total_anthropogenic",
        "total_natural",
        "GHG",
        "co2",
        "aerosols",
        "volcanic",
        "solar",
    ]
    forcedResponses = ["hist-aer", "hist-GHG", "hist-nat"]

    dict_copy = deepcopy(dict_models)
    trainFolds = partition(dict_copy["trainModels"], nbFoldsCV)

    #     forcedResponse_df = computeForcedResponse_CMIP6(
    #         variables, temporalRes, scenario, startDate, endDate, norm
    #     )
    #     modelsFull_forcedResponse = forcedResponse_df

    foldsData = []
    foldsTarget = []
    foldsAnchor = []

    for k in range(nbFoldsCV):
        X_df = pd.DataFrame()
        y_df = pd.DataFrame()
        y_anchor_df = pd.DataFrame()

        modelsFold = np.sort(trainFolds[k])
        if displayModels:
            print("Fold " + str(k) + " : " + str(modelsFold))

        if target in forcings:
            filesFold = [
                modelsDataList[i].filename
                for i in range(len(modelsDataList))
                if modelsDataList[i].model in modelsFold
            ]
            dataFoldList = [
                modelsDataList[i].data
                for i in range(len(modelsDataList))
                if modelsDataList[i].model in modelsFold
            ]
        elif target in forcedResponses:
            filesFold = [
                modelsDataList[i].filename
                for i in range(len(modelsDataList))
                if modelsDataList[i].model in modelsFold
                and modelsDataList[i].modelFull in modelsFull_forcedResponse
            ]
            dataFoldList = [
                modelsDataList[i].data
                for i in range(len(modelsDataList))
                if modelsDataList[i].model in modelsFold
                and modelsDataList[i].modelFull in modelsFull_forcedResponse
            ]

        for i in range(len(filesFold)):
            X_tmp = pd.DataFrame(dataFoldList[i])
            X_df = pd.concat([X_df, X_tmp], axis=0)

            ind_tmp = int(
                modelsInfoFrame[
                    modelsInfoFrame["filename"] == filesFold[i]
                ].index.values
            )
            startDate, endDate = (
                modelsDataList[ind_tmp].data.index[0],
                modelsDataList[ind_tmp].data.index[-1],
            )
            if target in forcings:
                y_tmp = read_forcing_cmip6(
                    modelsDataList[ind_tmp].scenario,
                    target,
                    startDate,
                    endDate,
                )
            elif target in forcedResponses:
                if modelsDataList[ind_tmp].scenario != "piControl":
                    y_tmp = forcedResponse_df[
                        modelsDataList[ind_tmp].modelFull
                    ]
                elif modelsDataList[ind_tmp].scenario == "piControl":
                    y_tmp = pd.Series(
                        np.zeros(endDate - startDate + 1),
                        index=np.linspace(
                            startDate, endDate, endDate - startDate + 1
                        ),
                    )
                    y_tmp.index.name = "Year"

            y_df = pd.concat([y_df, y_tmp], axis=0)

            """ Build the anchor """
            if anchor in forcings:
                tmp_anchor = read_forcing_cmip6(
                    modelsDataList[ind_tmp].scenario,
                    anchor,
                    startDate,
                    endDate,
                )
                y_anchor_df = pd.concat([y_anchor_df, tmp_anchor])

        foldsData.append(X_df)
        foldsTarget.append(y_df)
        foldsAnchor.append(y_anchor_df)

    dict_folds = {
        "foldsData": foldsData,
        "foldsTarget": foldsTarget,
        "foldsAnchor": foldsAnchor,
        "trainFolds": trainFolds,
    }

    return dict_folds
