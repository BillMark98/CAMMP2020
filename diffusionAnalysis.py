import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.stats import shapiro
# from scipy.stats import chisquare
# from scipy.stats import t
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
import fileIO
import statisticsAnalysis as stAna

def generateAlpha(dirLists, columns = ["MSD_xy_mean"], threshold2_low = 0.3,threshold2_high = 0.8, \
    threshold1_low = 1.6, threshold3_low = 0.8, region2start = -1, region3start = -1, time_unit = "ps", msd_unit = "nm", outputUnit = "si", \
        potentKnown = True, confidence = 0.95, secondDerivThreshold = 0.6):
    """
    given the directory lists, generate the alpha dataframe

    | foldername | molecule | concentration | cholesterol_concentration | confidence | alpha | alpha_interval| 

    -----
    To do:
    -----

    make more columns possible
    """
    for col in columns:
        dfFinal = pd.DataFrame(columns = ["foldername","molecule","concentration","cholesterol_concentration","confidence","alpha","alpha_errorbar"])
        for dir in dirLists:
            fileLists = fileIO.getFiles(dir)
            folderName = dir.split("/")[-1]
            for file in fileLists:
                molecule = file.split("/")[-1].split(".")[0]
                moleculeConcentration = fileIO.getMoleculeConentration(folderName)
                cholesterol_concentration = int(moleculeConcentration[molecule])
                if (molecule != "CHOL"):
                    cholesterol_concentration = 100.0 - cholesterol_concentration
                # treat the first column as the index
                df = pd.read_csv(file, index_col= 0) 
                regressionDict = stAna.msdFittingPlot(df, columns = [col], threshold2_low= threshold2_low, threshold2_high = threshold2_high,\
                    threshold1_low = threshold1_low, threshold3_low = threshold3_low, region2start = region2start, region3start = region3start, \
                        time_unit = time_unit, msd_unit = msd_unit, outputUnit = outputUnit, potentKnown = potentKnown, confidence = confidence, secondDerivThreshold = secondDerivThreshold)
                alpha = regressionDict['alpha']
                alpha_interval = regressionDict['alpha_interval']
                alpha_errorbar = alpha_interval[1] - alpha
                df_current = pd.DataFrame([{
                    'foldername': folderName,
                    'molecule': molecule,
                    'concentration': int(moleculeConcentration[molecule]),
                    'cholesterol_concentration': cholesterol_concentration,
                    'confidence': confidence,
                    'alpha': alpha,
                    'alpha_errorbar': alpha_errorbar
                }])
                dfFinal = dfFinal.append(df_current, ignore_index = True)
    return dfFinal

def generateDiffusionConst(dirLists, dim,  columns = ["MSD_xy_mean"], threshold2_low = 0.3,threshold2_high = 0.8, \
    threshold1_low = 1.6, threshold3_low = 0.8, region2start = -1, region3start = -1, time_unit = "ps", msd_unit = "nm", outputUnit = "si", \
        potentKnown = True, confidence = 0.95, secondDerivThreshold = 0.6, useRegression = True):
    """
    given the directory lists, generate the alpha dataframe

    | foldername | molecule | concentration | cholesterol_concentration | unit | confidence | diffusion_coeff | diffusion_errorbar| 

    -----
    To do:
    -----

    make more columns possible
    support more units for time and msd

    """
    if (outputUnit == "si"):
        unitEntry = "si"
    else:
        unitEntry = time_unit + "_" + msd_unit
    for col in columns:
        dfFinal = pd.DataFrame(columns = ["foldername","molecule","concentration","cholesterol_concentration","unit", "confidence","diffusion_coeff","diffusion_errorbar"])
        for dir in dirLists:
            fileLists = fileIO.getFiles(dir)
            folderName = dir.split("/")[-1]
            moleculeConcentration = fileIO.getMoleculeConentration(folderName)
            cholesterol_concentration = 0
            for file in fileLists:
                molecule = file.split("/")[-1].split(".")[0]
                cholesterol_concentration = int(moleculeConcentration[molecule])
                if (molecule != "CHOL"):
                    cholesterol_concentration = 100.0 - cholesterol_concentration
                # treat the first column as the index
                df = pd.read_csv(file, index_col= 0) 
                regressionDict = stAna.msdFittingPlot(df, columns = [col], threshold2_low= threshold2_low, threshold2_high = threshold2_high,\
                    threshold1_low = threshold1_low, threshold3_low = threshold3_low, region2start = region2start, region3start = region3start, \
                        time_unit = time_unit, msd_unit = msd_unit, outputUnit = outputUnit, potentKnown = potentKnown, confidence = confidence, secondDerivThreshold = secondDerivThreshold)
                if (useRegression) :
                    Dd = regressionDict['b3']
                    Dd_interval = regressionDict['b3_interval']
                else:
                    Dd = regressionDict['Dd']
                    Dd_interval = regressionDict['Dd_interval']

                D = Dd / dim
                D_errorbar_high = (Dd_interval[1] - Dd) / dim
                D_errorbar_low = (Dd - Dd_interval[0])/dim
                df_current = pd.DataFrame([{
                    'foldername': folderName,
                    'molecule': molecule,
                    'concentration': int(moleculeConcentration[molecule]),
                    'cholesterol_concentration': cholesterol_concentration,
                    "unit": unitEntry,
                    'confidence': confidence,
                    'diffusion_coeff': D,
                    'diffusion_errorbar': [D_errorbar_low, D_errorbar_high]
                }])
                dfFinal = dfFinal.append(df_current, ignore_index = True)
            # mixture
            if (len(moleculeConcentration) > 1):
                # there is a mixture
                df_temp = dfFinal[dfFinal["foldername"] == folderName]
                D_mix = 0
                Derrorbar_mix_low = 0 # errorbar low
                Derrorbar_mix_high = 0 # errorbar high
                for i in range(len(df_temp)):
                    D_mix += df_temp.iloc[i]["concentration"] * df_temp.iloc[i]["diffusion_coeff"]
                    Derrorbar_mix_low += df_temp.iloc[i]["concentration"] * df_temp.iloc[i]["diffusion_errorbar"][0]
                    Derrorbar_mix_high += df_temp.iloc[i]["concentration"] * df_temp.iloc[i]["diffusion_errorbar"][1]
                D_mix /= 100
                Derrorbar_mix_low /= 100
                Derrorbar_mix_high /= 100
                df_current = pd.DataFrame([{
                    'foldername' : folderName,
                    'molecule': "MIX",
                    "concentration": np.nan,
                    'cholesterol_concentration': cholesterol_concentration,
                    "unit": unitEntry,
                    'confidence': confidence,
                    'diffusion_coeff': D_mix,
                    'diffusion_errorbar': [Derrorbar_mix_low,Derrorbar_mix_high]
                }])
                dfFinal = dfFinal.append(df_current, ignore_index = True)
    return dfFinal


def plotAlpha(dfAlpha):
    groups = dfAlpha.groupby('molecule')
    x = []
    y = []
    yerror = []
    for name, group in groups:
        plt.plot(group.cholesterol_concentration, group.alpha, marker = 'o', linestyle = '', ms = 12, label = name)
        # x.append(group.cholesterol_concentration)
        # y.append(group.alpha)
        # yerror.append(group.alpha_errorbar)
    x = dfAlpha["cholesterol_concentration"]
    y = dfAlpha["alpha"]
    yerror = dfAlpha["alpha_errorbar"]
    plt.legend()
    print("x")
    print(x)
    print("y")
    print(y)
    print("yerror")
    print(yerror)
    # plt.plot(x,y)
    plt.errorbar(x,y,yerr = [yerror,yerror] , fmt = ' ',ms = 1,ecolor= 'magenta')
    plt.xlabel("cholesterol_concentration")
    plt.xticks([0,25,50])
    plt.ylabel("alpha")        

def plotDiffusionCoeff(dfDiffusionCoeff):
    groups = dfDiffusionCoeff.groupby('molecule')
    x = []
    y = []
    yerror = []

    for name, group in groups:
        plt.plot(group.cholesterol_concentration, group.diffusion_coeff, marker = 'o', linestyle = '', ms = 12, label = name)
        # x.append(group.cholesterol_concentration)
        # y.append(group.alpha)
        # yerror.append(group.alpha_errorbar)
    x = dfDiffusionCoeff["cholesterol_concentration"]
    y = dfDiffusionCoeff["diffusion_coeff"]
    yerror = dfDiffusionCoeff["diffusion_errorbar"]
    # print("x:")
    # print(x)
    # print("yerror")
    # print(yerror)
    yerrorLow = [None] * len(x)
    yerrorHigh = [None] * len(x)
    for i in range(len(x)):
        yerrorLow[i] = yerror[i][0]
        yerrorHigh[i] = yerror[i][1]
    print("yerrorLow")
    print(yerrorLow)
    print("yerrorHigh")
    print(yerrorHigh)
    plt.legend()
    print("x")
    print(x)
    print("y")
    print(y)
    print("yerror")
    print(yerror)
    # plt.plot(x,y)
    plt.errorbar(x,y,yerr = [yerrorLow,yerrorHigh] , fmt = ' ',ms = 1,ecolor= 'magenta')
    plt.xlabel("cholesterol_concentration")
    plt.xticks([0,25,50])
    plt.ylabel("diffusion_coeff")   