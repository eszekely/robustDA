import netCDF4
import numpy as np
import os
import pandas as pd

dirFiles = "/net/ch4/data/cmip6-Next_Generation/tas/ann/g025/"
# dirFiles = "./../data/mount/CMIP6/"


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


def print_info_netcdf(netcdf_dataset):
    print(netcdf_dataset.file_format)
    print("Dimensions: ", end="")
    print(list(netcdf_dataset.dimensions.keys()))
    print("Variables: ", end="")
    print(list(netcdf_dataset.variables.keys()))


def read_files_cmip6(params_climate, norm=True):

    temporalRes = params_climate["temporalRes"]
    variables = params_climate["variables"]
    scenarios = params_climate["scenarios"]
    startDate = params_climate["startDate"]
    endDate = params_climate["endDate"]

    modelsDataList = []

    for var in variables:
        if var == "tas":
            files = sorted(os.listdir(dirFiles))
        elif var == "pr":
            #             dirFiles = "./../data/mount/_DATA/CMIP5/2D/pr/"
            files = sorted(os.listdir(dirFiles))

        for filename in files:
            if (
                var in filename
                and temporalRes in filename
                and "r1i1p1f1" in filename
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
                                    temp_nc.variables["time"][:],
                                    temp_nc.variables["time"].units,
                                    only_use_cftime_datetimes=False,
                                )
                            ).year
                            """ Keep files only if dates are unique
                                (there are no repeating years)
                            """

                            if len(dates) == len(set(dates)):
                                temp_ncdata_df = pd.DataFrame(
                                    temp_ncdata, index=dates
                                )
                                temp_ncdata_df_selDates = temp_ncdata_df.loc[
                                    str(startDate) : str(endDate)
                                ]

                                if norm:
                                    """Remove mean of 1850-1900
                                    for each model run independently
                                    """
                                    meanmap = np.mean(
                                        temp_ncdata_df.loc["1850":"1900"],
                                        axis=0,
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
                            if temp_ncdata.shape[0] >= (endDate - startDate):
                                dates = pd.core.indexes.numeric.Int64Index(
                                    np.linspace(
                                        startDate,
                                        endDate,
                                        endDate - startDate + 1,
                                    )
                                )
                                temp_ncdata_df = pd.DataFrame(
                                    temp_ncdata[
                                        -(endDate - startDate + 1) :,
                                    ],
                                    index=dates,
                                )
                                temp_ncdata_df_selDates = temp_ncdata_df

                                """ Remove mean for each model independently"""
                                if norm:
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

    """ Remove models that only have piControl runs """
    nbFiles = len(modelsDataList)
    models = sorted(set([modelsDataList[i].model for i in range(nbFiles)]))
    modelsRemove = set()

    for i in range(len(models)):
        scenarios = [
            modelsDataList[j].scenario
            for j in range(nbFiles)
            if modelsDataList[j].model == models[i]
        ]
        if set(scenarios) == set(["piControl"]):
            modelsRemove.add(models[i])

    modelsDataList_final = [
        modelsDataList[i]
        for i in range(len(modelsDataList))
        if modelsDataList[i].model not in modelsRemove
    ]
    ind = [
        i
        for i in range(modelsInfoFrame.shape[0])
        if modelsInfoFrame.loc[i]["model"] in modelsRemove
    ]
    modelsInfoFrame = modelsInfoFrame.drop(ind)
    modelsInfoFrame = modelsInfoFrame.reset_index(drop=True)

    return modelsDataList_final, modelsInfoFrame


def read_forcing_cmip6(scenario, forcing, startDate, endDate):
    dirFiles = "./../data/local/radForcing/"

    # read RCP/SSP
    if scenario != "piControl":
        if "ssp245" in scenario:
            radforcing_df = pd.read_csv(
                dirFiles + "ERF_ssp245_1750-2500.csv", index_col=0
            )
        elif (
            "ssp585" in scenario
            or "historical" in scenario
            or "hist-GHG" in scenario
            or "hist-aer" in scenario
            or "hist-nat" in scenario
        ):
            radforcing_df = pd.read_csv(
                dirFiles + "ERF_ssp585_1750-2500.csv", index_col=0
            )

        radforcing_df.index.name = "Year"
        radforcing_df = radforcing_df.loc[startDate:endDate]

        if forcing == "total":
            radforcing_df = radforcing_df["total"]
        elif forcing == "total_anthropogenic":
            radforcing_df = radforcing_df["total_anthropogenic"]
        elif forcing == "volcanic":
            radforcing_df = radforcing_df["volcanic"]
        elif forcing == "solar":
            radforcing_df = radforcing_df["solar"]
        elif forcing == "total_natural":
            radforcing_df = radforcing_df["total_natural"]
        elif forcing == "GHG":
            radforcing_df = (
                radforcing_df["co2"]
                + radforcing_df["ch4"]
                + radforcing_df["n2o"]
            )
        elif forcing == "co2":
            radforcing_df = radforcing_df["co2"]
        elif forcing == "aerosols":
            radforcing_df = (
                radforcing_df["aerosol-cloud_interactions"]
                + radforcing_df["aerosol-radiation_interactions"]
            )

    # build control run radiative forcing (only zeros)
    elif scenario == "piControl":
        radforcing_df = pd.Series(
            np.zeros(endDate - startDate + 1),
            index=np.linspace(startDate, endDate, endDate - startDate + 1),
        )
        radforcing_df.index.name = "Year"

    return radforcing_df


""" Reads all runs, not just r1i1p1f1
    in order to do the averaging over all the runs
"""


def compute_forced_response_cmip6(
    variables, temporalRes, scenario, startDate, endDate, norm
):
    modelsDataList = []

    files = sorted(os.listdir(dirFiles))

    for var in variables:
        for filename in files:
            if (
                var in filename
                and temporalRes in filename
                and scenario in filename
            ):  # only full maps ("g025.nc")
                temp_nc = netCDF4.Dataset(dirFiles + filename)
                temp_ncdata = np.array(temp_nc.variables[var])
                temp_ncdata = temp_ncdata.reshape(
                    temp_ncdata.shape[0],
                    temp_ncdata.shape[1] * temp_ncdata.shape[2],
                )

                """ keep only files that go at least
                    1850-2014 (165 time steps) or
                    1850-2020 (171 time steps)
                """
                if temp_ncdata.shape[0] >= 165:
                    dates = pd.to_datetime(
                        netCDF4.num2date(
                            temp_nc.variables["time"][:],
                            temp_nc.variables["time"].units,
                        )
                    ).year
                    """ keep files only if dates are unique
                        (there are no repeating years)
                    """
                    if len(dates) == len(set(dates)):
                        temp_ncdata_df = pd.DataFrame(temp_ncdata, index=dates)
                        temp_ncdata_df_selDates = temp_ncdata_df.loc[
                            str(startDate) : str(endDate)
                        ]

                        if norm:
                            """Remove mean of 1850-1900
                            for each model independently
                            """
                            meanmap = np.mean(
                                temp_ncdata_df.loc["1850":"1900"], axis=0
                            )
                            temp_ncdata_df_selDates = (
                                temp_ncdata_df_selDates - meanmap
                            )

                        modelsDataList.append(
                            ClimateModelData(filename, temp_ncdata_df_selDates)
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

    uniqueModels = list(modelsInfoFrame.modelFull.unique())
    forcedResponse = np.zeros(
        shape=(len(uniqueModels), modelsDataList[0].data.shape[0])
    )

    for i in range(len(uniqueModels)):
        tempModelAverage = np.zeros(shape=modelsDataList[0].data.shape)
        indicesModels = []
        indicesModels = modelsInfoFrame.loc[
            modelsInfoFrame["modelFull"] == uniqueModels[i]
        ].index
        for j in indicesModels:
            tempModelAverage = tempModelAverage + modelsDataList[j].data.values
        tempModelAverage2 = tempModelAverage / len(indicesModels)
        forcedResponse[i, :] = np.mean(tempModelAverage2, axis=1).reshape(-1)

    forcedResponse_df = pd.DataFrame(
        np.transpose(forcedResponse),
        columns=uniqueModels,
        index=list(range(1850, 2015)),
    )
    forcedResponse_df.index.name = "Year"

    if scenario == "hist-aer":
        forcedResponse_df.to_csv(
            "./../data/local/forcedResponse/CMIP6_forcedResponse_hist-aer.csv"
        )
    elif scenario == "hist-GHG":
        forcedResponse_df.to_csv(
            "./../data/local/forcedResponse/CMIP6_forcedResponse_hist-GHG.csv"
        )

    return forcedResponse_df
