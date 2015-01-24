import numpy as np
import os

'''
This function makes list of integers using a file.
arg: name of file (String)
returns: list of integers that represent frame action
'''
def fromFileToList(fileName):
    actionFrameList = []
    with open(fileName) as f:
        content = f.readlines()
    #string list -> int list
    for line in content:
        actionFrameList.append(int(line))
    return np.array(actionFrameList)

def temporalSmoothing(fileName, vectorList):
    actionFrameList = fromFileToList(fileName)
    
    #firstStep
    lenFrameL = len(actionFrameList)
    markedFrames = []
    #for every frame
    for i in range(1, lenFrameL - 1):
        #mark potentially erroneous frames
        if actionFrameList[i - 1] != actionFrameList[i] and actionFrameList[i] != actionFrameList[i + 1]:
            markedFrames.append(i)
   
    #secondStep
    for i in markedFrames:
        vectorSum = vectorList[i]
        K = min(i, lenFrameL - i - 1) + 1 
        for k in range (1, K):
            vectorSum = vectorSum + vectorList[i - k] + vectorList[i + k]
        vectorList[i] = vectorSum/(2*K - 1)
    return vectorList