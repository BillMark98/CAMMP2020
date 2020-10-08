import pandas as pd
import os
import re

import fileIO
FILENUM_INF = 100000
def calculateMeanVarMedian(df, columns = ["MSD_x", "MSD_y", "MSD_z", "MSD_xy", "MSD_xyz"], dropXY = True, addXYZ = True):
    """
    calculate the useful statistics, mean, variance, median, rename the column accordingly,
    dropXY indicate whether to drop the column MSD_x and MSD_y

    --------
    Parameters:
    --------

    df: dataframe

    columns: list like
    specify what statistics will be calculated for

    dropXY: bool

    return: dataframe containing the statistics

    -----
    Assume
    ------
    df has column called "time" and "trajectory"
    """
    
    # drop "MSD_x" and "MSD_y" if needed
    if (dropXY):
        if ("MSD_x" in columns):
            columns.remove("MSD_x")
        if ("MSD_y" in columns):
            columns.remove("MSD_y")
        df.drop(["MSD_x","MSD_y"], axis = 1, errors = "ignore", inplace = True)

    df_mean = df.groupby(["time"], as_index = False).mean().drop(["trajectory"], axis = 1)
    mean_dict = {key : key + "_mean" for key in columns}
    df_mean.rename(columns = mean_dict, inplace = True)

    df_median = df.groupby(["time"], as_index = False).median().drop(["time","trajectory"], axis = 1)
    median_dict = {key : key + "_median" for key in columns}
    df_median.rename(columns = median_dict, inplace = True)

    df_var = df.groupby(["time"], as_index = False).var().drop(["time", "trajectory"], axis = 1)
    var_dict = {key : key + "_var" for key in columns}
    df_var.rename(columns = var_dict, inplace = True)

    return pd.concat([df_mean,df_median, df_var], axis = 1)

def mergeMSD(df_fs, df_ps, regime_change = 80, cut_off = 0.2):
    """
    Merge two dataframe of equal molecule and concentration at two different time scales
    """
    df1 = df_fs[df_fs.time < regime_change]
    df2 = df_ps[(df_ps.time > regime_change) &  (df_ps.time < df_ps.time.max() * cut_off)]
    df_final = pd.concat([df1,df2])
    # df_final["MSD_xyz_mean"] = df_final["MSD_xy_mean"] + df_final["MSD_z_mean"]
    # df_final
    return df_final

def generateAllMeanVarMedian(dirLists, extension = ".csv", regime_change = 80, cut_off = 0.2, fileNum = FILENUM_INF,regime_change = 80, cut_off = 0.2):
    """
    generate all mean,median, var files

    -----
    Assume
    -----
    dir has first fs, then ps
    """
    df_ps = None
    df_fs = None
    df_merge = None
    for count in len(dirLists):
        if (count % 2 == 0):
            df_fs = fileIO.createPDF(dirLists[count], extension = extension, fileNum = fileNum)
        else:
            df_ps = fileIO.createPDF(dirLists[count], extension = extension, fileNum = fileNum)
            df_merge = mergeMSD(df_fs, df_ps, regime_change = regime_change, cut_off = cut_off)