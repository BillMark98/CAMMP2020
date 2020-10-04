import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import chisquare
from sklearn.linear_model import LinearRegression
import fileIO

DEFAULT_NUM = 10000
EPSILON = 1e-20
INFINITIY = 1000000
NEG_INFINITY = -1000000
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
    dataFrameCol = ["concentration", "samplingrate", "molecule", "lowPercentile","upPercentile", "alpha"] + normalityTestCol
    testResultFinalDf = pd.DataFrame(columns = dataFrameCol)
    concentrationLists = df["concentration"].unique()
    for samplingRate in timeStep:
        df_samplingLevel = df[df["samplingrate"] == samplingRate]
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
                    "alpha": [alpha]
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
    dataFrameCol = ["concentration", "samplingrate", "molecule", "lowPercentile","upPercentile","alpha", "distribution"] + normalityTestCol
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
                    "alpha": [alpha],
                    "distribution": [distribution]
                }
                )                    
                for key, value in testResultDict.items():
                    print("%-10s %10.5f"%(key, value))
                    df_testResult[key] = value
                testResultFinalDf = testResultFinalDf.append(df_testResult)
    return testResultFinalDf         

def histPlot(df, timePoints, timeNum , particleNum, bins = 50, normalityTestCol = ["MSD_xy"]):
    """
    plot the histogram of df

    -------
    Parameters:
    -------

    timePoints: array-like structure
    the time points from which to choose points to plot

    timeNum: int
    number of time points to choose

    particleNum: int
    number of particles to choose from

    """
    
    ## make sure the timePoints are unique
    timePoints = list(set(timePoints))
    concentrationLists = df["concentration"].unique()
    
    # select certain particles
    # calculate the total timeNum of trajectories
    numTrajectories = len(df["trajectory"].unique())    
    numTrajectories = min(numTrajectories, particleNum)
    trajectoryIndex = chooseIndex(df["trajectory"],particleNum)
    trajectories = np.array(df["trajectory"])[trajectoryIndex]
    # subset the df
    df = df[df["trajectory"].isin(trajectories)]

    columns = ["time"] + normalityTestCol
    df_prune = df[columns]    
    timePointsNum = len(timePoints)

    timeNum = makeDivisible(timeNum)
    numberOfPoints = min(timePointsNum, timeNum)
    if (numberOfPoints > timePointsNum):
        timeLists = np.random.choice(timePoints, size = numberOfPoints, replace = False)
    else:
        timeLists = timePoints

    timeLists.sort()
    # df_prune = df_prune[df_prune[almostEqual(df_prune["time"], timeLists)]]
    ## to do 
    ## adapt plotting 
    ncol = 2
    nrow = int(timeNum / ncol)



    fig,axes = plt.subplots(nrow,ncol)
    fig.tight_layout(pad = 5.0)
    fig.suptitle("total number of particles chosen: " + str(numTrajectories), fontsize = 20)
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

    plt.show()

def diffArr(arr):
    """

    calculates the difference between each elements of arr

    """
# not sure if some arr starts indexing with 0
# for example, doing a slicing of a dataframe, will cause arr starts at index 1
# df_prune = df[1:]
# diffArr(df_prune["MSD_xy"])  <---- index starts at 1
    lenArr = len(arr)
    arTemp = [None] * (lenArr - 1)
    count = 0
    for val in arr:
        if (count == 0):
            oldval = val
        else:
            arTemp[count - 1] = val - oldval
            oldval = val
        count += 1
    return np.array(arTemp)

def eliminateZeroTime(df):
    """
    eliminate the 0 time row, (doing log will cause problem)

    -----
    To do
    ------
    figure out how to drop 0 that emerge at rows other than the first line
    """

    if (almostEqual(df.iloc[0,:]["time"], 0)):
        return df[1:]
    else:
        return df

def findLowUpIndex(arr, lowBound = NEG_INFINITY, upBound = INFINITIY, startIndex = -1):
    """
    find the left, right index of an array, that has a continuous range lying between [lowBound, upBound]
    only count the first time this range is hit, and with consideration to the startIndex

    -------
    Parameters:
    -------
    arr: array like structure

    lowBound: float

    upBound: float

    startIndex: int

    ------
    To do:
    ------
    optimise the code for lowBound == NEG_INFINITY, upBOUND == INFINITY
    """
    
    # make sure it is a numpy.array
    arr = np.array(arr)
    boolArr = (arr < upBound) & (arr > lowBound)
    leftIndex = -1
    rightIndex = -1
    leftFound = False
    rightFound = False
    continuous = True
    for i in range(len(boolArr)):
        if (i >= startIndex):
            if ((boolArr[i] == True) and (leftFound == False)):
                leftFound = True
                leftIndex = i
            elif (leftFound == True):
                # need to first check if the leftIndex is already found
                if((boolArr[i]) and (not rightFound) and (continuous)):
                    rightIndex = i
                elif (rightFound == False):
                    # boolArr[i] == False
                    rightFound = True
                    continuous = False
                    break
    print("lowBound: " + str(lowBound))
    print("upBound: " + str(upBound))
    print("leftIndex: " + str(leftIndex))
    print("rightIndex: " + str(rightIndex))
    return leftIndex, rightIndex

def getLogLogSlope(df, yColumn = ["MSD_xy_mean"], xColumn = ["time"]):
    """
    find the slope k which is given by log(y) = k * log(x) + b
    """
    ## eliminate possible zero rows

    if (len(yColumn) > 1 or len(xColumn) > 1):
        raise Exception("Only support for one dimensional array!")
    if (len(yColumn) < 1 or len(xColumn) < 1):
        raise Exception("no entry in yColumn or xColumn!")
    df_prune = eliminateZeroTime(df)
    
    for xCol in xColumn:
        for yCol in yColumn:
            tLog = np.log(df_prune[xCol])
            msdLog = np.log(df_prune[yCol])
            tLogDiff = diffArr(tLog)
            msdLogDiff = diffArr(msdLog)
            slopes = msdLogDiff / tLogDiff
            return slopes

def divide3Region(df, columns = ["MSD_xy_mean"],threshold2_low = 0.3,threshold2_high = 0.6, threshold1_low = 1.6, threshold3_low = 0.8):
    """

    very specific for MSD analysis, given the msd, divide the series into three regions
    assume df has "time" as one of its column name
    ------
    Parameters
    -------

    return: three lists and the slopes
    example usage l1,l2,l3,slopes = divide3Region(df, columns = ["MSD_xy"])
    df_1 = df.iloc[l1]

    ------
    To do
    ------

    make several columns possible

    """
    # threshold1_low = 1.6
    # threshold3_low = 0.8
    if (threshold3_low < threshold2_high + EPSILON):
        threshold3_low += 0.09
    threshold3_high = 1.2
    df_prune = eliminateZeroTime(df)
    if (len(columns) > 1):
        raise Exception("currenlty only division works for one column")
    
    for col in columns:
        # timeSeries = df_prune["time",col]
        # tLog = np.log(df_prune["time"])
        # msdLog = np.log(df_prune[col])
        # tLogDiff = diffArr(tLog)
        # msdLogDiff = diffArr(msdLog)
        # slopes = msdLogDiff / tLogDiff
        slopes = getLogLogSlope(df_prune, yColumn = [col], xColumn= ["time"])
        ## find the first region
        l1Dummy, r1 = findLowUpIndex(slopes, lowBound= threshold1_low)
        l2,r2 = findLowUpIndex(slopes, lowBound=threshold2_low, upBound= threshold2_high)
        # have to make sure that l3 starts to count after the region 2
        l3,r3Dummy = findLowUpIndex(slopes, lowBound= threshold3_low, upBound = threshold3_high, startIndex=r2)
        list1 = list(range(0,r1 + 1))
        list2 = list(range(l2,r2 + 1))
        list3 = list(range(l3,len(slopes)))
        return list1, list2, list3, slopes

def simpleLinearRegression(xArr,yArr):
    """
    do a linear regression s.t yArr_approx = k * xArr + b

    ----
    Parameters:
    ----

    return: k,b
    """
    model = LinearRegression().fit(xArr,yArr)
    return model.coef_, model.intercept_

def msdRegression(df, columns = ["MSD_xy_mean"]):
    """
    do a linear regression s.t log(columns) = k * log(time) + b

    ----
    Pay Attention:
    ----
    Assume df has no zero Time

    -----
    Parmaeters:
    -----

    return: k, b

    ----
    To do:
    -----

    Making multiple columns possible

    """

    # pandas.Series do not support reshape function, and given a pandas.Series, the np.log() wont convert that to a ndarray
    # so need to convert to a ndarray first
    tLog = np.log(df["time"]).to_numpy().reshape((-1,1))

    for col in columns:
        msdLog = np.log(df[col])
        return simpleLinearRegression(tLog, msdLog)

def msdFittingPlot(df, columns = ["MSD_xy_mean"], threshold2_low = 0.3,threshold2_high = 0.6, threshold1_low = 1.6, threshold3_low = 0.8):
    """
    plot the msd and the curve fitting

    ----
    To do:
    ----
    df should eliminate 0?
    making multiple columns possible
    annotate the curve
    """
    df = eliminateZeroTime(df)

    for col in columns:
        list1, list2, list3, slopes = divide3Region(df,columns = [col], threshold2_low = threshold2_low, threshold2_high = threshold2_high, threshold1_low = threshold1_low, threshold3_low = threshold3_low)
        indexLists = [list1,list2,list3]
        dfLists = [df.iloc[indexLists[var]] for var in range(3)]
        fig, axes = plt.subplots()
        # plot the whole msd
        legends = ["MSD","ballistic","subdiffusion","brownian"]
        tLog = np.log(df["time"])
        msdLog = np.log(df[col])
        plt.plot(tLog,msdLog, label = legends[0])

        for i in range(3):
            k,b = msdRegression(dfLists[i], columns = [col])
            if (i == 0):
                intercept_1 = np.exp(b)
            elif ( i == 1):
                alpha_2 = k[0]
            else:
                intercept_3 = np.exp(b)
            # ts = np.log(dfLists[i]["time"])
            ts = np.log(df["time"])
            ys = k * ts + b
            plt.plot(ts, ys, label = legends[i + 1], linestyle = "dashed")
        plt.legend()
        plt.show()
    print("ballistic region, const (2d * kT/m): {:.4f}".format(intercept_1))
    print("subdiffusion region, potent alpha: {:.4f}".format(alpha_2))
    print("brownian region, const (2dD): {:.4f}".format(intercept_3))
    return intercept_1, alpha_2, intercept_3

if __name__ == "__main__":
    arr = np.array([2,3,4,5,6,8])
    arrT = diffArr(arr)
    print(arrT)

    charArr = ["hi","there","yo","yo1","haha"]
    tempIndx = chooseIndex(charArr,2, 0,0.5)
    print(tempIndx)