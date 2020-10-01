import pandas as pd
import os
import re
sizeInf = 1000000

# concentration , samplingrate, trajectory, molecule
def containsFiles(currentPath, extension = ".csv"):
    """
    check if the currentPath contains file with extension ".csv"

    ------
    Parameters:
    ----
    
    currentPath: str
    extension:str
    default ".csv"

    """
    
    for file in os.listdir(currentPath):
        if file.endswith(extension):
            return True
        
    return False

def getSubDir(currentPath, extension = ".csv"):
    """
    Get all subdirectories of the current path, that contains the ".csv" file

    ------
    Parameters
    -----

    currentPath: str
    string that indicate the current working directory

    """

    dirLists = []
    for root, dirs,_ in os.walk(currentPath):
        for d in dirs:
            if containsFiles(os.path.join(root,d), extension):
                dirLists.append(os.path.join(root,d))
    return dirLists    

def getFiles(fileDir, extension= ".csv", fileNum = sizeInf):
    """
    pick a random number of files (.xvg) from the current fileDir
    if no number is specified, all files will be returned

    ------
    Parameters
    -----

    fileDir: str
    the current fileDir

    extension: str
    extension of a file, default ".csv"

    fileNum: int
    the number of files extract

    return:
    a list of file names

    """
    
    # to do make it random
    fileLen = len(os.listdir(fileDir))
    fileTotalNum = min(fileLen, fileNum)
    count = 1
    fileNames = []
    for file in os.listdir(fileDir):
        if (count > fileTotalNum):
            break
        if file.endswith(extension):
            fileNames.append(os.path.join(fileDir,file))
            count += 1
    return fileNames

def getConcentration(dirName):

    dirLists = dirName.split("/")
    return dirLists[-2]

def getSampling(dirName):
    dirLists = dirName.split("/")[-1]
    tempName = dirLists.split("_")
    return tempName[-1]
def getMolecule(dirName):
    dirLists = dirName.split("/")[-1]
    tempName = dirLists.split("_")
    return tempName[-2]
def getMoleculeConentration(composition, delimiter = "_"):
    """
    get the molecule concentration from the name composition

    ------
    Parameters
    ------

    composition: str

    Note that the column name is called "concentration" in the dataframe

    delimiter: str
    used for the splitting the composition name
    """
    splittedName = composition.split(delimiter)
    moleculeCompositionDict = {}
    for names in splittedName:
        r = re.compile("([a-zA-Z]+)([0-9]+)")
        m = r.match(names)
        moleculeCompositionDict[m.group(1)] = m.group(2)
    print(moleculeCompositionDict)
    return moleculeCompositionDict
def getNumbering(fileName):
    """
    Get the numbering of the fileName
    """
    fileName = fileName.split("/")[-1]
    return fileName.split(".")[-2]
# concentration , samplingrate, trajectory, molecule
def getXvgDirDataFrame(currentPath):
    # get dirname
    dirLists = getSubDir(currentPath)
    concentrationLists = [getConcentration(var) for var in dirLists]
    samplingRates = [getSampling(var) for var in dirLists]
    molecules = [getMolecule(var) for var in dirLists]
    return pd.DataFrame({
        "diretory": dirLists,
        "concentration": concentrationLists,
        "sampling": samplingRates,
        "molecule": molecules
    })

def createPDF(currentPath, extension = ".csv" , fileNum = sizeInf):
    """
    create the pandas dataframe based on all the csv file from the currentPath
    
    """
    index = ["time","MSD_x","MSD_y","MSD_z","concentration","samplingrate","trajectory","molecule"]
    index2 = ["time","MSD_x","MSD_y","MSD_z"]
    df2 = pd.DataFrame(columns = index)
    fileNameLists = getFiles(currentPath, extension, fileNum)
    concentration = getConcentration(currentPath)
    samplingrate = getSampling(currentPath)
    molecule = getMolecule(currentPath)    
    for file in fileNameLists:
        df = pd.read_csv(file, delimiter=",", header = 0, decimal = ".", names = index2)
        df["trajectory"] = getNumbering(file)
        df["molecule"] = molecule
        df["samplingrate"] = samplingrate
        df["concentration"] = concentration
        df["MSD_xy"] = df["MSD_x"] + df["MSD_y"]
        df2 = df2.append(df)
    return df2

def createCompactPDF(currentPath, extension = ".csv", fileNum = sizeInf):
    """
    create the compact pdf

    """
    dirListName = getSubDir(currentPath, extension = ".csv")
    print("dirListName: ")
    print(dirListName)
    index = ["time","MSD_x","MSD_y","MSD_z","concentration","samplingrate","trajectory","molecule"]
    # index2 = ["time","MSD_x","MSD_y","MSD_z"]
    df2 = pd.DataFrame(columns = index)
    for dirName in dirListName:
        df = createPDF(dirName, extension, fileNum)
        df2 = df2.append(df)
    return df2

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    # dirLists = getSubDir(".")
    # print(dirLists)
    # print()
    # fileNames = getFiles(dirLists[0])
    # print(fileNames)

    # print("Numbering: " + str(getNumbering(fileNames[0])))

    # concentration = getConcentration(dirLists[0])
    # print(concentration)
    # sampling = getSampling(dirLists[0])
    # print(sampling)
    # molecule = getMolecule(dirLists[0])
    # print(molecule)
    # tempVar = getXvgDirDataFrame(".")
    # print(tempVar)

    # dirDF = getXvgDirDataFrame("../../MSD_Analysis")
    # dirListName = getSubDir("../../MSD_Analysis", extension = ".csv")
    # print("dirListName: ")
    # print(dirListName)
    # index = ["time","MSD_x","MSD_y","MSD_z","concentration","samplingrate","trajectory","molecule"]
    # index2 = ["time","MSD_x","MSD_y","MSD_z"]
    # df2 = pd.DataFrame(columns = index)
    # print("test data")
    # for dirName in dirListName:
    #     fileNameLists = getFiles(dirName, ".csv")
    #     concentration = getConcentration(dirName)
    #     samplingrate = getSampling(dirName)
    #     molecule = getMolecule(dirName)
    #     for file in fileNameLists:
    #         df = pd.read_csv(file, delimiter=",", header = 0, decimal = ".", names = index2)
    #         df["trajectory"] = getNumbering(file)
    #         df["molecule"] = molecule
    #         df["samplingrate"] = samplingrate
    #         df["MSD_xy"] = df["MSD_x"] + df["MSD_y"]
    #         df2 = df2.append(df)

    # df2.to_csv("./Plotting/jupyter/raw_data.csv")
    getMoleculeConentration("DOPC50_CHOL50")