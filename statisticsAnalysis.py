import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import chisquare
import fileIO

DEFAULT_NUM = 10000
EPSILON = 1e-20
def almostEqual(a,b,epsilon = EPSILON):
    return np.abs(a-b) < epsilon

def makeDivisible(a, d = 2):
    if a % d != 0:
        return d - a % d + a
    else:
        return a
def chooseIndex(indices, num, lowPercentile = 0, upPercentile = 1):
    """
    choose indices from [lowPercentile * len, upPercentile * len]:

    ------
    Parameters:
    -------

    indices: array like 

    num: int

    lowPercentile: float

    upPercentile: float
    
    """

    lenIndices = len(indices)
    lowerBound = int(np.floor(lenIndices * lowPercentile))
    upperBound = int(np.floor(lenIndices * upPercentile))
    # upperBound will not be reached so no need to add 1,
    # think for example upPercentile = 1, then upPercentile == lenIndices, which should not be reached as an index of the list
    # but lowerBound = 0 could be reached.
    totalLen = upperBound - lowerBound
    if (totalLen < 0):
        raise Exception("totalLen < 0!")
    if ( num >= totalLen):
        return list(range(lowerBound, upperBound))
    else:
        indexPrim = np.random.choice(range(lowerBound, upperBound), num , replace= False)
        indexPrim.sort()
        return indexPrim

def rvNormalityTest(rvSeries, alpha = "0.05"):
    """
    Test if a given series of random variables is normal distributed based on shapiro wilk 

    ------
    Parameter   
    -------

    rvSeries:
    array like structure, series of random variables

    alpha: float
    the alpha value for the hypothesis test

    """
    stat, p = shapiro(rvSeries)
    if p > alpha:
        return True
    else:
        return False

def rvTest(rvSeries, alpha = "0.05", distribution = "normal"):
    """
    Test if a given series of random variables is distributed based according to the scheme specified distribution

    ------
    Parameter   
    -------

    rvSeries:
    array like structure, series of random variables

    alpha: float
    the alpha value for the hypothesis test

    """

    if (distribution == "normal"):
        stat, p = shapiro(rvSeries)
    elif (distribution == "chisquare"):
        stat, p = chisquare(rvSeries)

    if p > alpha:
        return True
    else:
        return False        

def dfNormalityTest(df, num = DEFAULT_NUM, lowPercentile = 0.02, upPercentile = 0.1, alpha = 0.05, normalityTestCol = ["MSD_xy"]):
    """
    Test if a dataframe from the msd analysis, in particular the "MSD_xy" 

    -------
    Parameter
    -------

    df: pandas.DataFrame

    num: int

    alpha: float

    """

    ## select only the MSD_xy and time
    ### maybe could insert at the beginning

    ### To do
    ### add concentration, molecule, info
    timeStep = df["samplingrate"].unique()
    dataFrameCol = ["concentration", "samplingrate", "molecule", "lowPercentile","upPercentile"] + normalityTestCol
    testResultFinalDf = pd.DataFrame(columns = dataFrameCol)
    concentrationLists = df["concentration"].unique()
    for samplingRate in timeStep:
        df_samplingLevel = df[df["samplingrate"] == samplingRate]
        "df_samp"
        for concentration in concentrationLists:
            moleculeConcentrationDict = fileIO.getMoleculeConentration(concentration)
            columnConcentration = ["time"] + normalityTestCol + ["concentration", "molecule"]
            df_concentrationLevel = df_samplingLevel[columnConcentration]
            df_concentrationLevel = df_concentrationLevel[df_concentrationLevel["concentration"] == concentration]        
            for molecule in moleculeConcentrationDict.keys():
                df_moleculeLevel = df_concentrationLevel[df_concentrationLevel["molecule"] == molecule]
                testResultDict = {}
                # initialize testResultDict
                for elem in normalityTestCol:
                    testResultDict[elem] = 0
                
                columns = ["time"] + normalityTestCol
                df_prune = df_moleculeLevel[columns]
                timeLag = df_prune["time"].unique()
                Indices = chooseIndex(list(range(len(timeLag))), num, lowPercentile = lowPercentile, upPercentile= upPercentile)
                # perhaps there should be more molecules, but this dataframe only include one
                # like POPC75_CHOL25, and only CHOL is present
                if (len(Indices) < 1):
                    continue
                for index in Indices:   
                    tempDF = df_prune[almostEqual(df_prune["time"], timeLag[index])]
                    for elem in normalityTestCol:
                        rvSeries = tempDF[elem]
                        isNormal = rvNormalityTest(rvSeries, alpha)
                        if (isNormal):
                            testResultDict[elem] += 1
                # print("the testResultDict absolute: ")
                # print(testResultDict)
                totalNum = len(Indices)
                for key in testResultDict.keys():
                    testResultDict[key] /= totalNum
                # print("------------------------------")
                # print("after modification:")
                # print("the testResultDict absolute: ")
                # print(testResultDict)
                # print("-----------------------------")
                ## print result:
                print ("running normality test with time range :")
                print("({0:.4f}, {1:.4f})".format(timeLag[Indices[0]], timeLag[Indices[-1]]))
                print("total number of samplings: ")
                print(totalNum)
                print("alpha: " + str('{:.5f}'.format(alpha)))
                print("rate of passing:")
                # ["concentration", "samplingrate", "molecule", "lowPercentile","upPercentile"] + normalityTestCol
                df_testResult = pd.DataFrame({
                    "concentration": [concentration],
                    "samplingrate": [samplingRate],
                    "molecule": [molecule],
                    "lowPercentile": [lowPercentile],
                    "upPercentile":[upPercentile],
                }
                )                    
                for key, value in testResultDict.items():
                    print("%-10s %10.5f"%(key, value))
                    df_testResult[key] = value
                testResultFinalDf = testResultFinalDf.append(df_testResult)
    return testResultFinalDf                    
def dfTest(df, num = DEFAULT_NUM, lowPercentile = 0.02, upPercentile = 0.1, alpha = 0.05, normalityTestCol = ["MSD_xy"], distribution = "normal"):
    """
    Test if a dataframe from the msd analysis, in particular the "MSD_xy" 

    -------
    Parameter
    -------

    df: pandas.DataFrame

    num: int

    alpha: float

    """

    ## select only the MSD_xy and time
    ### maybe could insert at the beginning

    ### To do
    ### add concentration, molecule, info
    timeStep = df["samplingrate"].unique()
    dataFrameCol = ["concentration", "samplingrate", "molecule", "lowPercentile","upPercentile"] + normalityTestCol
    testResultFinalDf = pd.DataFrame(columns = dataFrameCol)
    concentrationLists = df["concentration"].unique()
    for samplingRate in timeStep:
        df_samplingLevel = df[df["samplingrate"] == samplingRate]
        "df_samp"
        for concentration in concentrationLists:
            moleculeConcentrationDict = fileIO.getMoleculeConentration(concentration)
            columnConcentration = ["time"] + normalityTestCol + ["concentration", "molecule"]
            df_concentrationLevel = df_samplingLevel[columnConcentration]
            df_concentrationLevel = df_concentrationLevel[df_concentrationLevel["concentration"] == concentration]        
            for molecule in moleculeConcentrationDict.keys():
                df_moleculeLevel = df_concentrationLevel[df_concentrationLevel["molecule"] == molecule]
                testResultDict = {}
                # initialize testResultDict
                for elem in normalityTestCol:
                    testResultDict[elem] = 0
                
                columns = ["time"] + normalityTestCol
                df_prune = df_moleculeLevel[columns]
                timeLag = df_prune["time"].unique()
                Indices = chooseIndex(list(range(len(timeLag))), num, lowPercentile = lowPercentile, upPercentile= upPercentile)
                # perhaps there should be more molecules, but this dataframe only include one
                # like POPC75_CHOL25, and only CHOL is present
                if (len(Indices) < 1):
                    continue
                for index in Indices:   
                    tempDF = df_prune[almostEqual(df_prune["time"], timeLag[index])]
                    for elem in normalityTestCol:
                        rvSeries = tempDF[elem]
                        distriTrue = rvTest(rvSeries, alpha, distribution= distribution)
                        if (distriTrue):
                            testResultDict[elem] += 1
                # print("the testResultDict absolute: ")
                # print(testResultDict)
                totalNum = len(Indices)
                for key in testResultDict.keys():
                    testResultDict[key] /= totalNum
                # print("------------------------------")
                # print("after modification:")
                # print("the testResultDict absolute: ")
                # print(testResultDict)
                # print("-----------------------------")
                ## print result:
                print ("running test with time range :")
                print("({0:.4f}, {1:.4f})".format(timeLag[Indices[0]], timeLag[Indices[-1]]))
                print("total number of samplings: ")
                print(totalNum)
                print("alpha: " + str('{:.5f}'.format(alpha)))
                print("rate of passing:")
                # ["concentration", "samplingrate", "molecule", "lowPercentile","upPercentile"] + normalityTestCol
                df_testResult = pd.DataFrame({
                    "concentration": [concentration],
                    "samplingrate": [samplingRate],
                    "molecule": [molecule],
                    "lowPercentile": [lowPercentile],
                    "upPercentile":[upPercentile],
                }
                )                    
                for key, value in testResultDict.items():
                    print("%-10s %10.5f"%(key, value))
                    df_testResult[key] = value
                testResultFinalDf = testResultFinalDf.append(df_testResult)
    return testResultFinalDf         

def histPlot(df, timePoints, num, bins = 50, normalityTestCol = ["MSD_xy"]):
    """
    plot the histogram of df

    -------
    Parameters:
    -------

    timePoints: array-like structure
    the time points from which to choose points to plot

    num: int
    number of time points to choose

    """
    
    ## make sure the timePoints are unique
    timePoints = timePoints.unique()
    concentrationLists = df["concentration"].unique()
    

    columns = ["time"] + normalityTestCol
    df_prune = df[columns]    
    timePointsNum = len(timePoints)

    num = makeDivisible(num)
    numberOfPoints = min(timePointsNum, num)
    timeLists = np.random.choice(timePoints, size = numberOfPoints, replace = False)
    timeLists.sort()
    # df_prune = df_prune[df_prune[almostEqual(df_prune["time"], timeLists)]]
    ## to do 
    ## adapt plotting 
    ncol = 2
    nrow = int(num / ncol)

    fig,axes = plt.subplots(nrow,ncol)
    fig.tight_layout(pad = 5.0)
    count = 1
    for elem in normalityTestCol:
        for time in timeLists:
            rvVariables = df_prune[almostEqual(df_prune["time"], time)][elem]
            plt.subplot(nrow, ncol, count)
            rvVariables.plot.hist(bins = bins ) #ax = axes[count, 0]
            plt.title("time:{0:.2f}".format(time))
            plt.xlabel(elem)
            ## figSub.autofmt_xdate()
            count += 1            