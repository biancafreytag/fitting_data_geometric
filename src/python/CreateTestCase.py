
import numpy as np
import subprocess
import sys
import os

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


# def writeDataPointCoordiantes(X, Y, Z):
#
#
#
#     writeExFile('syntheticDataCoordinates', field, 'data_coordinates', ['x', 'y', 'z'],type='data')
#
#     return
#
#
#


def getEigenvectorsAtDatapoints():

    if sys.platform == "darwin":
        out = subprocess.check_output(
            ["/Applications/Cmgui.app/Contents/MacOS/Cmgui -no_display returnSyntheticEigenvectors.cmgui"], shell=False,
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
        ev = np.resize(eigenvectors[dataPointIdx,:], (3, 3))
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





# Create Quaternion values at top and bottom nodes
angle = 45 / 180.0 * np.pi
axis = np.array([0, 0, -1])
axis_unit = axis**2 / (np.sum(axis**2))
topNodesQuaternions = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

angle = 45 / 180.0 * np.pi
axis = np.array([0, 0, -1])
axis_unit = axis**2 / (np.sum(axis**2))
bottomNodesQuaternions = [np.cos(angle/2),np.sin(angle/2) * axis_unit[0],np.sin(angle/2) * axis_unit[1],np.sin(angle/2) * axis_unit[2]]

# write the quaternions into file
cubeQuaternions = [bottomNodesQuaternions,bottomNodesQuaternions,bottomNodesQuaternions,bottomNodesQuaternions, topNodesQuaternions,topNodesQuaternions,topNodesQuaternions,topNodesQuaternions]




# # define the data planes
#
# inPlaneResolution = 20
# longtitudinalPlaneLocations = [0.25,0.75]
#
# l = np.linspace(0.01, 0.99, inPlaneResolution)
# X,Y = np.meshgrid(l,l)
# Z = np.zeros_like(X)
#
#
# # write data coordinates into file
# writeDataPointCoordiantes(X, Y, Z)


writeQuaternionExnode(cubeQuaternions)
# writeDataPointCoordiantes(X, Y, Z)
eigenvectors = getEigenvectorsAtDatapoints()


tensors = calculateTensorsFromEigenvectors(eigenvectors, [0.3,0.2,0.15])
writeTensors(tensors)


print "done"


