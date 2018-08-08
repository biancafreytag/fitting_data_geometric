
import numpy as np
import subprocess
import sys
import os

from shutil import copyfile

try:
    from myExFile import *
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'myPythonLib'))
    from myExFile import *




def writeQuaternionExnode(quaternions):

    numberOfNodes = 8
    nodeNumberArray = np.linspace(1, numberOfNodes, numberOfNodes)

    quat = np.reshape(quaternions, (numberOfNodes, -1))
    field = np.vstack((nodeNumberArray, quat.T)).T

    writeExFile('syntheticQuaternions', field, 'Quaternions', ['1', '2', '3', '4'], type='node')

    return


def writeDataPointCoordiantes(X, Y, Z):

    vectorX = np.reshape(X,(1,-1))
    vectorY = np.reshape(Y,(1,-1))
    coordinates = np.empty((0,3), int)

    numberOfDatapoints = vectorX.shape[1] * len(Z)
    dataPointIdx = np.linspace(1, numberOfDatapoints, numberOfDatapoints)

    for z in Z:
        vectorZ = np.full((1, vectorX.shape[1]), z)
        coord = np.concatenate((vectorX, vectorY, vectorZ), axis=0).T
        coordinates = np.append(coordinates,coord,axis=0)

    field = np.vstack((dataPointIdx, coordinates.T)).T
    writeExFile('syntheticDataCoordinates', field, 'data_coordinates', ['x', 'y', 'z'],type='data')

    return




def getEigenvectorsAtDatapoints():

    if sys.platform == "darwin":
        out = subprocess.check_output(
            ["/Applications/Cmgui.app/Contents/MacOS/cmgui", "-no_display", "returnSyntheticEigenvectors.cmgui"], shell=False,
            stderr=subprocess.STDOUT).strip()
    else:
        out = subprocess.check_output(["cmgui_new -no_display returnSyntheticEigenvectors.cmgui"], shell=True,
                                      stderr=subprocess.STDOUT).strip()
    vectors = readExFile('syntheticEigenvectors.exdata')[:,1:]

    return vectors




def calculateTensorsFromEigenvectors(eigenvectors, eigenvalues):

    l = np.array([[eigenvalues[0],0,0],
               [0,eigenvalues[1],0],
               [0,0,eigenvalues[2]]])

    dataPointsNumber = eigenvectors.shape[0]
    tensors = np.zeros((dataPointsNumber,9))

    for dataPointIdx in range(dataPointsNumber):
        ev = np.resize(eigenvectors[dataPointIdx,:], (3, 3)).T
        tensor = np.dot(np.dot(ev,l),np.linalg.inv(ev))
        tensors[dataPointIdx,:] = np.resize(tensor, (1,9))

    return tensors



def writeTensors(tensors):

    numberOfNodes = tensors.shape[0]
    nodeNumberArray = np.linspace(1, numberOfNodes, numberOfNodes)

    quat = np.reshape(tensors, (numberOfNodes, -1))
    field = np.vstack((nodeNumberArray, quat.T)).T

    writeExFile('syntheticTensors', field, 'tensor', ['11', '12', '13','21', '22', '33','31', '32', '33'], type='data')

    return


def createCubeQuaternions(case):

    if case is 'a':

        # simple field with one direction only.
        angle = 45 / 180.0 * np.pi
        axis = np.array([0, 0, 1.0])
        axis_unit = axis**2 / (np.sum(axis**2))
        topNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        angle = 45 / 180.0 * np.pi
        axis = np.array([0, 0, 1.0])
        axis_unit = axis**2 / (np.sum(axis**2))
        bottomNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        return [bottomNodes,bottomNodes,bottomNodes,bottomNodes, topNodes,topNodes,topNodes,topNodes]

    elif case is 'b':

        # rotation in plane by 90 deg total
        angle = -45 / 180.0 * np.pi
        axis = np.array([0, 0, 1.0])
        axis_unit = axis**2 / (np.sum(axis**2))
        leftNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        angle = 45 / 180.0 * np.pi
        axis = np.array([0, 0, 1.0])
        axis_unit = axis**2 / (np.sum(axis**2))
        rightNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        return [leftNodes,rightNodes,leftNodes,rightNodes,leftNodes,rightNodes,leftNodes,rightNodes]

    elif case is 'c':

        # fibre like rotation
        angle = -60 / 180.0 * np.pi
        axis = np.array([1.0, 0, 0])
        axis_unit = axis**2 / (np.sum(axis**2))
        leftNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        angle = 60 / 180.0 * np.pi
        axis = np.array([1.0, 0, 0])
        axis_unit = axis**2 / (np.sum(axis**2))
        rightNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        return [leftNodes,rightNodes,leftNodes,rightNodes,leftNodes,rightNodes,leftNodes,rightNodes]


    elif case is 'd':

        # fibre like rotation
        angle = -60 / 180.0 * np.pi
        axis = np.array([0, 1.0, 0])
        axis_unit = axis**2 / (np.sum(axis**2))
        frontNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        angle = 60 / 180.0 * np.pi
        axis = np.array([0, 1.0, 0])
        axis_unit = axis**2 / (np.sum(axis**2))
        rearNodes = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

        return [frontNodes,frontNodes,rearNodes,rearNodes,frontNodes,frontNodes,rearNodes,rearNodes]




# define the data planes

inPlaneResolution = 10
l = np.linspace(0.01, 0.99, inPlaneResolution)

X,Y = np.meshgrid(l,l)
Z = [0.2,0.4,0.6,0.8]
# Z = [0.25,0.75]




print "start"
case = 'd'
cubeQuaternions = createCubeQuaternions(case)
case = 'd_4planes'


writeQuaternionExnode(cubeQuaternions)
writeDataPointCoordiantes(X, Y, Z)
eigenvectors = getEigenvectorsAtDatapoints()
tensors = calculateTensorsFromEigenvectors(eigenvectors, [0.3,0.2,0.15])
writeTensors(tensors)


print "moving files"

directoryOld = os.getcwd()+"/"
directoryNew = os.getcwd()+"/"+case+"/"

if not os.path.exists(directoryNew):
    os.makedirs(directoryNew)

os.rename(directoryOld+'syntheticQuaternions.exnode', directoryNew+'syntheticQuaternions.exnode')
os.rename(directoryOld+'syntheticTensors.exdata', directoryNew+'syntheticTensors.exdata')
os.rename(directoryOld+'syntheticDataCoordinates.exdata', directoryNew+'syntheticDataCoordinates.exdata')
copyfile(directoryOld+'UndeformedGeometry.exelem', directoryNew+'UndeformedGeometry.exelem')
copyfile(directoryOld+'viewCube.cmgui', directoryNew+'viewCube.cmgui')
os.remove(directoryOld+'syntheticEigenvectors.exdata')

print "done"


