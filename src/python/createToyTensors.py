from numpy import radians, sin, cos, linspace, cross, array,column_stack,dot,random
from numpy.linalg import inv
from itertools import chain
import time

"""
This script create the data_coordinates and the toyTensor exdata files for a cube.
We'd like to mimic the heart data which consists of high resolution planes with big gaps along the long axis.


"""


def addGaussianNoise(tensorList, noiseLevel):

    random.seed()

    tensorListRand=[0] * 9

    tensorListRand[0] = float( tensorList[0] + random.uniform(-1, 1) * noiseLevel)
    tensorListRand[4] = float(tensorList[4] + random.uniform(-1, 1) * noiseLevel)
    tensorListRand[8] = float(tensorList[8] + random.uniform(-1, 1) * noiseLevel)

    tensorListRand[1] = tensorListRand[3] =  float(tensorList[1] + random.uniform(-1, 1) * noiseLevel)
    tensorListRand[5] = tensorListRand[7] = float(tensorList[5] + random.uniform(-1, 1) * noiseLevel)
    tensorListRand[2] = tensorListRand[6] = float(tensorList[2] + random.uniform(-1, 1) * noiseLevel)

    return tensorListRand



def createToyTensors(noiseLevel, resolution):


    # Create a cube with points along each axis (x, y, z)
    # Define how many points you would like along each axis


    # to mimic the heart data, we choose to have high in plane resolution and sparse data along z.
    pointsX = resolution
    pointsY = resolution
    pointsZ = 2

    # define where along the z axis the two planes should lie
    planePositionTop = 0.25
    planePositionBottom = 0.75

    # define the angles for the top and bottom plane
    angleAtTopNode = radians(-45.0)
    angleAtBottomNode = radians(45.0)



    noPoints = pointsX*pointsY*pointsZ
    X=linspace(0.01,0.99,pointsX)
    Y=linspace(0.01,0.99,pointsY)
    Z=linspace(planePositionTop,planePositionBottom,pointsZ)

    angleTop = (angleAtBottomNode-angleAtTopNode)*planePositionTop+angleAtTopNode
    angleBottom = (angleAtBottomNode-angleAtTopNode)*planePositionBottom+angleAtTopNode



    # now we create the two sets of tensors
    # first define eigenvectors
    fTop = array([cos(angleTop),sin(angleTop),0])
    nTop = array([-sin(angleTop),cos(angleTop),0])
    sTop = array(cross(fTop,nTop))
    aTop = column_stack((fTop,sTop,nTop))

    fBottom = array([cos(angleBottom),sin(angleBottom),0])
    nBottom = array([-sin(angleBottom),cos(angleBottom),0])
    sBottom = array(cross(fBottom,nBottom))
    aBottom = column_stack((fBottom,sBottom,nBottom))

    # we also need a set of artificial eigenvalues
    l = array([[0.3,0,0],
               [0,0.2,0],
               [0,0,0.15]])


    # create tensor from eigenvectors and values
    dTop = dot(dot(aTop,l),inv(aTop))
    dBottom = dot(dot(aBottom,l),inv(aBottom))

    # turn into flat list
    dTopList = list(chain.from_iterable(dTop))
    dBottomList = list(chain.from_iterable(dBottom))

    coordinates = []
    tensors = []



    for i in range(pointsX):
        for j in range(pointsY):
            for k in range(pointsZ):
                coordinates.append([X[i],Y[j], Z[k]])

                if Z[k] == planePositionTop:
                    tensors.append(addGaussianNoise(dTopList,noiseLevel))

                else:
                    tensors.append(addGaussianNoise(dBottomList,noiseLevel))


    file = open('data_coordinates.exdata',"w")

    file.write("Region: / \n")
    file.write("Group name : data\n")
    file.write("#Fields=1\n")
    file.write("1) data_coordinates, coordinate, rectangular cartesian, #Components=3\n")
    file.write(" x.  Value index=1, #Derivatives=0, #Versions=1\n")
    file.write(" y.  Value index=2, #Derivatives=0, #Versions=1\n")
    file.write(" z.  Value index=3, #Derivatives=0, #Versions=1\n")

    for i in range(noPoints):
        file.write("Node:   "+str(i+1)+"\n")
        file.write(" "+ str(coordinates[i][0]) + " " + str(coordinates[i][1]) + " " + str(coordinates[i][2]) + "\n")


    file.close()


    file = open('toyTensor.exdata',"w")

    file.write("Region: / \n")
    file.write("!#nodeset datapoints\n")
    file.write("#Fields=1\n")
    file.write("1) tensor, field, rectangular cartesian, #Components=9\n")
    file.write(" 11.  Value index=1, #Derivatives=0, #Versions=1\n")
    file.write(" 12.  Value index=2, #Derivatives=0, #Versions=1\n")
    file.write(" 13.  Value index=3, #Derivatives=0, #Versions=1\n")
    file.write(" 21.  Value index=4, #Derivatives=0, #Versions=1\n")
    file.write(" 22.  Value index=5, #Derivatives=0, #Versions=1\n")
    file.write(" 23.  Value index=6, #Derivatives=0, #Versions=1\n")
    file.write(" 31.  Value index=7, #Derivatives=0, #Versions=1\n")
    file.write(" 32.  Value index=8, #Derivatives=0, #Versions=1\n")
    file.write(" 33.  Value index=9, #Derivatives=0, #Versions=1\n")


    for i in range(noPoints):
        file.write("Node:   "+str(i+1)+"\n")
        file.write(' {: 19.15e}'.format(tensors[i][0]) +' {: 19.15e}'.format(tensors[i][1]) +' {: 19.15e}'.format(tensors[i][2]) +' {: 19.15e}'.format(tensors[i][3]) +' {: 19.15e}'.format(tensors[i][4]) +' {: 19.15e}'.format(tensors[i][5]) +' {: 19.15e}'.format(tensors[i][6]) +' {: 19.15e}'.format(tensors[i][7]) +' {: 19.15e}'.format(tensors[i][8]) + "\n")

    file.close()

    print "New tensors with noise level =", noiseLevel, "and", str(noPoints), "datapoints"

    return 1
