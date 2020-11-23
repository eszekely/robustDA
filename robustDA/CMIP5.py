import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import netCDF4
import cartopy.crs as ccrs


class ClimateModelData:
    def __init__(self, filename, data):
        self.filename = filename

        attr = filename.split("_")

        self.var = attr[0]
        self.temporalRes = attr[1]
        self.modelFull = attr[2]
        self.model = attr[2][:3]
        self.scenario = attr[3]
        self.run = attr[4]
        self.spatialRes = attr[5].split(".")[0]
        self.data = data


def printInfoNetCDF(netCDFdataset):
    print(netCDFdataset.file_format)
    print("Dimensions: ", end="")
    print(list(netCDFdataset.dimensions.keys()))
    print("Variables: ", end="")
    print(list(netCDFdataset.variables.keys()))


def readFiles(variables, temporalRes, scenarios, startDate, endDate, norm):
    modelsDataList = []

    for var in variables:
        if var == "tas":
            dirFiles = "./../data/mount/_DATA/CMIP5/2D/tas/"
            files = sorted(os.listdir(dirFiles))
        elif var == "ta10":
            dirFiles = "./../data/mount/ta_10/"
            files = sorted(os.listdir(dirFiles))
            files.remove(
                "ta10_ann_CMCC-CESM_rcp85_r1i1p1_g025.nc"
            )  # some years are repeating
        elif var == "pr":
            dirFiles = "./../data/mount/_DATA/CMIP5/2D/pr/"
            files = sorted(os.listdir(dirFiles))

        for filename in files:
            if (
                (var in filename)
                and (temporalRes in filename)
                and ("r1i1p1_g025.nc" in filename)
            ):  # only full maps ("g025.nc")
                for scen in scenarios:
                    if scen in filename:
                        temp_nc = netCDF4.Dataset(dirFiles + filename)
                        temp_ncdata = np.array(temp_nc.variables[var])
                        temp_ncdata = temp_ncdata.reshape(
                            temp_ncdata.shape[0],
                            temp_ncdata.shape[1] * temp_ncdata.shape[2],
                        )

                        if scen != "piControl":
                            dates = pd.to_datetime(
                                netCDF4.num2date(
                                    temp_nc.variables["year"][:],
                                    temp_nc.variables["year"].units,
                                )
                            ).year
                            temp_ncdata_df = pd.DataFrame(
                                temp_ncdata, index=dates
                            )
                            temp_ncdata_df_selDates = temp_ncdata_df.loc[
                                str(startDate) : str(endDate)
                            ]

                            if norm == True:
                                # Remove mean of 1870-1920 for each model independently
                                meanmap = np.mean(
                                    temp_ncdata_df.loc["1870":"1920"], axis=0
                                )
                                temp_ncdata_df_selDates = (
                                    temp_ncdata_df_selDates - meanmap
                                )

                            modelsDataList.append(
                                ClimateModelData(
                                    filename, temp_ncdata_df_selDates
                                )
                            )

                        elif scen == "piControl":
                            #                             dates = pd.core.indexes.numeric.Int64Index(np.linspace(startDate, \
                            #                                          min(endDate, temp_ncdata.shape[0] + startDate - 1), \
                            #                                          min(endDate, temp_ncdata.shape[0] + startDate - 1) - startDate + 1))
                            #                             temp_ncdata_df = pd.DataFrame(temp_ncdata[: min(endDate, temp_ncdata.shape[0] + startDate - 1) - \
                            #                                            startDate + 1,], index = dates)
                            #                             temp_ncdata_df_selDates = temp_ncdata_df
                            if temp_ncdata.shape[0] >= 231:
                                dates = pd.core.indexes.numeric.Int64Index(
                                    np.linspace(
                                        startDate,
                                        endDate,
                                        endDate - startDate + 1,
                                    )
                                )
                                temp_ncdata_df = pd.DataFrame(
                                    temp_ncdata[
                                        : endDate - startDate + 1,
                                    ],
                                    index=dates,
                                )
                                temp_ncdata_df_selDates = temp_ncdata_df
                                #                             else:
                                #                                 print(filename)

                                if norm == True:
                                    # Remove mean for each model independently
                                    meanmap = np.mean(
                                        temp_ncdata_df_selDates, axis=0
                                    )
                                    temp_ncdata_df_selDates = (
                                        temp_ncdata_df_selDates - meanmap
                                    )

                                modelsDataList.append(
                                    ClimateModelData(
                                        filename, temp_ncdata_df_selDates
                                    )
                                )

    modelsInfoFrame = pd.DataFrame(
        index=range(len(modelsDataList)),
        columns=[
            "filename",
            "var",
            "temporalRes",
            "modelFull",
            "model",
            "scenario",
            "spatialRes",
        ],
    )

    for i in range(len(modelsDataList)):
        modelsInfoFrame.iloc[i, 0] = modelsDataList[i].filename
        modelsInfoFrame.iloc[i, 1] = modelsDataList[i].var
        modelsInfoFrame.iloc[i, 2] = modelsDataList[i].temporalRes
        modelsInfoFrame.iloc[i, 3] = modelsDataList[i].modelFull
        modelsInfoFrame.iloc[i, 4] = modelsDataList[i].model
        modelsInfoFrame.iloc[i, 5] = modelsDataList[i].scenario
        modelsInfoFrame.iloc[i, 6] = modelsDataList[i].spatialRes

    return modelsDataList, modelsInfoFrame


def readForcing(scenario, forcing, startDate, endDate):
    dirFiles = "./../data/mount/_DATA/reg_glob/FORCING_CMIP5/global/ALLDATA_30May2010/"

    # read RCP
    if scenario != "piControl":
        if scenario == "rcp45":
            radforcing_df = pd.read_excel(
                dirFiles + "RCP45_MIDYEAR_RADFORCING.xls",
                sheet_name=3,
                skiprows=59,
                header=0,
                index_col=0,
            )
        elif scenario == "rcp85":
            radforcing_df = pd.read_excel(
                dirFiles + "RCP85_MIDYEAR_RADFORCING.xls",
                sheet_name=3,
                skiprows=59,
                header=0,
                index_col=0,
            )
        elif scenario == "historicalNat":
            radforcing_df = pd.read_excel(
                dirFiles + "RCP85_MIDYEAR_RADFORCING.xls",
                sheet_name=3,
                skiprows=59,
                header=0,
                index_col=0,
            )

        radforcing_df.index.name = "YEAR"
        radforcing_df = radforcing_df.loc[startDate:endDate]

        if forcing == "all":
            radforcing_df = radforcing_df["TOTAL_INCLVOLCANIC_RF"]
        elif forcing == "anthro":
            radforcing_df = radforcing_df["TOTAL_ANTHRO_RF"]
        elif forcing == "volcanic":
            radforcing_df = radforcing_df["VOLCANIC_ANNUAL_RF"]
        elif forcing == "solar":
            radforcing_df = radforcing_df["SOLAR_RF"]
        elif forcing == "natural":
            radforcing_df = (
                radforcing_df["SOLAR_RF"] + radforcing_df["VOLCANIC_ANNUAL_RF"]
            )
        elif forcing == "GHG":
            radforcing_df = radforcing_df["GHG_RF"]
        elif forcing == "aerosols":
            radforcing_df = radforcing_df["TOTAER_DIR_RF"]

    # build control run radiative forcing (only zeros)
    elif scenario == "piControl":
        radforcing_df = pd.Series(
            np.zeros(endDate - startDate + 1),
            index=np.linspace(startDate, endDate, endDate - startDate + 1),
        )
        radforcing_df.index.name = "YEAR"

    return radforcing_df


def partition(list_in, n):
    random.shuffle(list_in)
    return [list(list_in[i::n]) for i in range(n)]


# Split the data into training and testing
def splitDataTrainTest(
    modelList,
    modelsInfoFrame,
    forcing,
    percTrain=0.5,
    nbFoldsCV=3,
    displayModels=True,
):

    filenames = [modelList[i].filename for i in range(len(modelList))]
    models = list(set([modelList[i].model for i in range(len(modelList))]))

    trainModels = np.sort(
        list(random.sample(models, int(np.round(percTrain * len(models)))))
    )
    trainFiles = [
        modelList[i].filename
        for i in range(len(modelList))
        if modelList[i].model in trainModels
    ]
    trainDataList = [
        modelList[i].data
        for i in range(len(modelList))
        if modelList[i].model in trainModels
    ]

    testModels = list(set(models) - set(trainModels))
    testFiles = [
        modelList[i].filename
        for i in range(len(modelList))
        if modelList[i].model in testModels
    ]
    testDataList = [
        modelList[i].data
        for i in range(len(modelList))
        if modelList[i].model in testModels
    ]

    # TRAINING DATA (as a list of folds where each fold contains multiple full models)
    trainFolds = partition(trainModels, nbFoldsCV)

    trainData_listFolds = []
    trainForcing_listFolds = []

    for k in range(nbFoldsCV):
        X_train_df = pd.DataFrame()
        y_train_df = pd.DataFrame()

        modelsFold = np.sort(trainFolds[k])
        if displayModels == True:
            print("Fold " + str(k) + " : [" + ", ".join(modelsFold) + "]")

        filesFold = [
            modelList[i].filename
            for i in range(len(modelList))
            if modelList[i].model in trainFolds[k]
        ]
        dataFoldList = [
            modelList[i].data
            for i in range(len(modelList))
            if modelList[i].model in trainFolds[k]
        ]

        for i in range(len(filesFold)):
            X_tmp = pd.DataFrame(dataFoldList[i])
            X_train_df = pd.concat([X_train_df, X_tmp], axis=0)

            ind_tmp = int(
                modelsInfoFrame[
                    modelsInfoFrame["filename"] == filesFold[i]
                ].index.values
            )
            #             print(filesFold[i] + " --- " + modelList[ind_tmp].filename)
            y_tmp = readForcing(
                modelList[ind_tmp].scenario,
                forcing,
                modelList[ind_tmp].data.index[0],
                modelList[ind_tmp].data.index[-1],
            )
            y_train_df = pd.concat([y_train_df, y_tmp], axis=0)

        trainData_listFolds.append(X_train_df)
        trainForcing_listFolds.append(y_train_df)

    # TESTING DATA
    X_test_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    if displayModels == True:
        print("Testing : [" + ", ".join(testModels) + "]")

    for i in range(len(testFiles)):
        X_tmp = pd.DataFrame(testDataList[i])
        X_test_df = pd.concat([X_test_df, X_tmp], axis=0)

        ind_tmp = int(
            modelsInfoFrame[
                modelsInfoFrame["filename"] == testFiles[i]
            ].index.values
        )
        #         print(testFiles[i] + " --- " + modelList[ind_tmp].filename)
        y_tmp = readForcing(
            modelList[ind_tmp].scenario,
            forcing,
            modelList[ind_tmp].data.index[0],
            modelList[ind_tmp].data.index[-1],
        )
        y_test_df = pd.concat([y_test_df, y_tmp], axis=0)

    dict_models = {
        "trainData_listFolds": trainData_listFolds,
        "trainForcing_listFolds": trainForcing_listFolds,
        "X_test": X_test_df,
        "y_test": y_test_df,
        "trainFiles": trainFiles,
        "testFiles": testFiles,
    }

    return dict_models


def plotMapCartopy(dataMap, cLim=None, title=None, filename=None, fontSize=16):
    fig = plt.figure(figsize=(12, 4))
    logs = np.arange(0, 360, 2.5)
    lats = np.arange(-90, 90, 2.5)

    ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
    ax.coastlines()
    ax.gridlines()

    if cLim == None:
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
    cbar.set_label("Temperature coefficients", fontsize=12)
    cbar.ax.tick_params(labelsize=fontSize - 4)

    if title != None:
        fig.suptitle(title, fontsize=fontSize)

    if filename != None:
        fig.savefig(filename, bbox_inches="tight")


def plotMapCartopy_subplots(
    ax, dataMap, cLim=None, title_subplot=None, fontSize=20
):
    logs = np.arange(0, 360, 2.5)
    lats = np.arange(-90, 90, 2.5)

    ax.coastlines()
    ax.gridlines()

    if cLim == None:
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

    cbar = plt.colorbar(
        h, orientation="horizontal", shrink=0.8, pad=0.1, aspect=30
    )
    cbar.ax.tick_params(labelsize=fontSize - 4)

    if title_subplot != None:
        plt.title(title_subplot, fontsize=fontSize)
