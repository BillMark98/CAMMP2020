import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import chisquare
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import fileIO

DEFAULT_NUM = 10000
EPSILON = 1e-20
INFINITY = 1000000
NEG_INFINITY = -1000000
def almostEqual(a,b,epsilon = EPSILON):
    return np.abs(a-b) < epsilon

def makeDivisible(a, d = 2):
    if a % d != 0:
        return d - a % d + a
    else:
        return a

def findCommonInterval(list1, list2, as_series = True):
    """
    find the common interval of list  1 and list 2

    ----
    Parameters:
    ----

    as_series: bool
    boolean indicating whether list1 and list2 are arithmetic sequence, if so, more efficient method can be used

    return: low1,up1, low2,up2, s.t list1[low1:up1] ~ list2[low2:up2]

    ------
    To do:
    ------
    create version for non-as series, and check also whether the given series are arithmetic
    check also that the two lists can have an intersection, also the calculation for high_small and low_large is very sloppy
    """ 
    # find which one is larger
    if (list1[1] < list2[1]):
        larger = list2
        smaller = list1
        inOrder = True
    else:
        larger = list1
        smaller = list2
        inOrder = False
    # epsilon = INFINITY
    # for count in range(len(list1)):
    #     difference = abs(list(1) - )
    

    ## alternative
    ## assuming arithmetic_sequence
    if (as_series):
        dstep_small = smaller[1] - smaller[0]
        dstep_large = larger[1] - larger[0]
        low_small = int(dstep_large / dstep_small) - 1
        high_small = len(smaller)  ### sloppy
        low_large = 1
        high_large = int(smaller[-1] / dstep_large) + 1
        if (inOrder):
            return low_small, high_small, low_large, high_large
        else:
            return low_large, high_large, low_small, high_small
    else:

        raise Exception("Non arithmetic increase not supported yet!")

def findCommonPoints(list1, list2, as_series = True):
    """
    find the common points (in the sense of almostEqual) of two lists

    -----
    Parameters:
    -----

    return: index1List, index2List, 

    ------
    To do:
    ---------
    The case for non as_series
    """
    # find which one is larger
    if (list1[1] < list2[1]):
        larger = list2
        smaller = list1
        inOrder = True
    else:
        larger = list1
        smaller = list2
        inOrder = False
    # epsilon = INFINITY
    # for count in range(len(list1)):
    #     difference = abs(list(1) - )
    

    ## alternative
    ## assuming arithmetic_sequence
    if (as_series):
        dstep_small = smaller[1] - smaller[0]
        dstep_large = larger[1] - larger[0]
        factor = int(dstep_large / dstep_small)

        low_large = 0
        low_small = int((larger[low_large] - smaller[0]) / dstep_small)
        high_small = len(smaller)  ### sloppy

        smallIndex = list(range(low_small, high_small, factor))
        
        high_large = int((smaller[-1] - larger[low_large]) / dstep_large) + 1
        largeIndex = list(range(low_large, high_large))
        if (inOrder):
            return smallIndex, largeIndex
        else:
            return largeIndex, smallIndex
    else:

        raise Exception("Non arithmetic increase not supported yet!")    

def calculateMeanVarMedian(df, columns = ["MSD_x", "MSD_y", "MSD_z", "MSD_xy"], dropXY = True):
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

def merge2TimeScale(df1, df2, columns = ["MSD_xy_mean","MSD_z"]):
    """
    given two dataframe saving the msd for two different time scales, merge the two together

    ------
    To do:
    ------
    Check first that df1 and df2 are of the same composition and molecule!

    ------
    Parameters:
    ----------

    df1: dataframe,

    df2: dataframe

    columns: list like, saving the columns to be merged

    return : dataframe with columns as specified in the variable columns

    ------
    Assuming
    -------
    dataframe has columns called "time"

    """

    ## To do, check if it is reasonable to merge these two

    ## find common interval
    l1, l2 = findCommonPoints(df1["time"], df2["time"], as_series = True)
    df1 = df1.iloc[l1]
    df2 = df2.iloc[l2]
    if (len(df1) != len(df2)):
        print(l1)
        print(l2)
        raise Exception("findCommonPoints error, returned list1 and 2 not equal length!")
    for col in columns:
        minDiff = INFINITY
        minIndex = -1
        for i in range(len(df1)):
            difference = abs(df1.iloc[i][col] - df2.iloc[i][col])
            if (difference < minDiff):
                minDiff = difference
                minIndex = i
        
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

def diffArr(arr, method = "forward"):
    """

    calculates the difference between each elements of arr

    -------
    Parameters:
    ------

    method: str
    method used for diffArr, "forward", "central", "backward"

    -------
    To do:
    -------
    for central diff, how to treat the first element, here use upward diff
    """
# not sure if some arr starts indexing with 0
# for example, doing a slicing of a dataframe, will cause arr starts at index 1
# df_prune = df[1:]
# diffArr(df_prune["MSD_xy"])  <---- index starts at 1
    lenArr = len(arr)
    arTemp = [None] * (lenArr - 1)
    count = 0
    if (method == "forward"):
        for val in arr:
            if (count == 0):
                oldval = val
            else:
                arTemp[count - 1] = val - oldval
                oldval = val
            count += 1
    elif (method == "central"):
        for val in arr:
            if (count == 0):
                oldval = val
            elif (count == 1):
                arTemp[0] = val - oldval # the diff for the first element use forward
            elif(count >= 2):
                arTemp[count - 1] = val - oldval
                oldval = arr[count - 1]
            count += 1
    elif (method == "backward"):
        raise Exception("temporarily not supported for backward")
    else:
        raise Exception("unknown type of method for calculation diff: " + method)

    return np.array(arTemp)

def getSlope(xArr, yArr, method = "forward"):
    """
    get the slope of yArr w.r.t xArr,

    ------
    Parameters:
    ------

    method: str
    method use for calculating slope, "forward", "central", "backward"

    ---
    To do:
    ----
    1.  realize the "backward" scheme
    2.  does central makes sense, if x is not arithmetic sequence?
    3.  could relax np.array(diffArr(...)) because diffArrr already ensures the returned is ndarray
    """
    if (len(xArr) != len(yArr)) :
        raise Exception("x and y not equal length, cant calculate the slope")
    if (method == "forward"):
        xArr_diff = np.array(diffArr(xArr, method = "forward"))
        yArr_diff = np.array(diffArr(yArr, method = "forward"))
    elif (method == "central"):
        xArr_diff = np.array(diffArr(xArr, method = "central"))
        yArr_diff = np.array(diffArr(yArr, method = "central"))
    elif (method == "backward"):
        xArr_diff = np.array(diffArr(xArr, method = "backward"))
        yArr_diff = np.array(diffArr(yArr, method = "backward"))
    else:
        raise Exception("Unknown type of method for slope calculation: " + method)

    slopes = yArr_diff/xArr_diff
    return slopes    
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

def findLowUpIndex(arr, lowBound = NEG_INFINITY, upBound = INFINITY, startIndex = -1):
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

def getLogLogSlope(df, yColumn = ["MSD_xy_mean"], xColumn = ["time"], method = "forward"):
    """
    find the slope k which is given by log(y) = k * log(x) + b

    ------
    Parameters:
    ------
    
    method: str
    method used for the calculation of slopes, "forward", "central", "backward"
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

def divide3Region(df, columns = ["MSD_xy_mean"],threshold2_low = 0.3,threshold2_high = 0.6, threshold1_low = 1.6, threshold3_low = 0.8, region3start = -1):
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

    1.  make several columns possible, make the search adaptive, do not require user to guess which threshold to set, maybe could try to use second order difference
    2.  region3Start, need  to check not too large?
    
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

        region3start = max(region3start, r2)
        if (region3start >= len(df[col])):
            raise Exception("region3start too large, already exceeds the series length!")
        l3,r3Dummy = findLowUpIndex(slopes, lowBound= threshold3_low, upBound = threshold3_high, startIndex=region3start)
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

    return: k,b,mean_squared_error, confidenceInterval of k, confidenceInterval of b
    """
    model = LinearRegression().fit(xArr,yArr)
    # yPred = LinearRegression().
    # mean_sqre_error = mean_sqre_error()
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

    return: k, b, mean_squared_error, confidenceInterval of k, confidenceInterval of b

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

def msdFittingPlot(df, columns = ["MSD_xy_mean"], threshold2_low = 0.6,threshold2_high = 0.8, threshold1_low = 1.6, threshold3_low = 0.8, region3start = -1, time_unit = "ps", msd_unit = "nm", outputUnit = "si"):
    """
    plot the msd and the curve fitting

    -----
    Parameters
    -----

    time_unit: str, 
    indicate the units of the timescale, default ps

    ----
    To do:
    ----
    1. df should eliminate 0?
    2. making multiple columns possible
    3. annotate the curve with slope and intercept
    4. given units, figure out the constant to change to SI seconds
    """
    df = eliminateZeroTime(df)

    if (time_unit == "ps" and msd_unit == "nm"):
        # factor = 1e-12
        offset_alpha = 12 * np.log(10) # each intercept calculated need to add this offset * alpha, e.g MSD * 1e-18 = k (1e-12 * t)^a, logMSD = logk + a * log(t) - 12 a * log(10) + 18 log(10)
        offset_const = -18 * np.log(10) 
    else:
        raise Exception("currently does not support other units")
    for col in columns:
        list1, list2, list3, slopes = divide3Region(df,columns = [col], threshold2_low = threshold2_low, threshold2_high = threshold2_high, threshold1_low = threshold1_low, threshold3_low = threshold3_low, region3start = region3start)
        indexLists = [list1,list2,list3]
        dfLists = [df.iloc[indexLists[var]] for var in range(3)]
        fig, axes = plt.subplots()
        # plot the whole msd
        legends = ["MSD","ballistic","subdiffusion","brownian"]
        tLog = np.log(df["time"])
        msdLog = np.log(df[col])
        plt.plot(np.exp(tLog),np.exp(msdLog), label = legends[0])
        # plt.plot(tLog,msdLog, label = legends[0])

        for i in range(3):
            k,b = msdRegression(dfLists[i], columns = [col])
            b_siunit = b + offset_alpha * k[0] + offset_const
            if (i == 0):
                intercept_1 = np.exp(b_siunit)
                if (outputUnit == "si"):
                    b_1 = intercept_1
                else:
                    b_1 = np.exp(b)
                k_1 = k[0]
            elif ( i == 1):
                intercept_2 = np.exp(b_siunit)
                alpha_2 = k[0]
            else:
                intercept_3 = np.exp(b_siunit)
                k_3 = k[0]
                if (outputUnit == "si"):
                    b_3 = intercept_3
                else:
                    b_3 = np.exp(b)                
            # ts = np.log(dfLists[i]["time"])
            ts = np.log(df["time"])
            ys = k * ts + b

            plt.loglog(np.exp(ts), np.exp(ys), label = legends[i + 1], linestyle = "dashed")
            # plt.plot(ts, ys, label = legends[i + 1], linestyle = "dashed")
        plt.legend()
        plt.show()
    constant1 = df.iloc[0][col] * 1e6 / np.square(df.iloc[0]["time"])
    print("ballistic region, one point calculated const (d * kT/m):{0:.4e}".format(constant1))
    print("ballistic region, const (d * kT/m): {0:.4e}, potent: {1:.4f}".format(intercept_1, k_1))
    print("subdiffusion region, intercept: {0:.4e}, potent alpha: {1:.4f}".format(intercept_2, alpha_2))
    print("brownian region, const (2dD): {0:.4e}, potent: {1:.4f}".format(intercept_3, k_3))
    return b_1, alpha_2, b_3

if __name__ == "__main__":
    arr = np.array([2,3,4,5,6,8])
    arrT = diffArr(arr)
    print(arrT)

    charArr = ["hi","there","yo","yo1","haha"]
    tempIndx = chooseIndex(charArr,2, 0,0.5)
    print(tempIndx)

    ar1 = np.arange(4,20,4)
    ar2 = np.arange(1,20,1)

    l1, l2 = findCommonPoints(ar1,ar2)
    print(ar1)
    print(ar2)
    print(l1)
    print(l2)