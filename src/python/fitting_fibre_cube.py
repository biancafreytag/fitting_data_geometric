#!/usr/bin/env python

# Fitting example

#> Main script
# Add Python bindings directory to PATH
import sys, os
import numpy as np
from math import cos, sin, pi
import copy
import exfile

try:
    import myExFile
except ImportError:
    sys.path.insert(0, "/people/bfre608/myPythonLib")
    import myExFile

# Intialise OpenCMISS
from opencmiss.iron import iron

def fit(numberGlobalXElements, numberGlobalYElements, numberGlobalZElements,directory,
               useGeneratedMesh=True):


    width = 1.0
    length = 1.0# Set problem parameters - Unit cube
    height = 1.0

    numberOfDimensions = 3
    NumberOfGaussXi = 2
    numberOfFitIterations = 3
    nonlinearFit = True
    smoothing = False

    # Set Sobolev smoothing parameters
    if smoothing:
        smoothingType = iron.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE
        # Set Sobolev smoothing parameters
        tau = 0.0
        kappa = 0.0
    else:
        smoothingType = iron.EquationsSetFittingSmoothingTypes.NONE

    coordinateSystemUserNumber = 1
    regionUserNumber = 1
    basisUserNumber = 1
    pressureBasisUserNumber = 2
    generatedMeshUserNumber = 1
    meshUserNumber = 1
    decompositionUserNumber = 1
    geometricFieldUserNumber = 1
    fibreFieldUserNumber = 2
    materialFieldUserNumber = 3
    dependentFieldUserNumber = 4
    equationsSetFieldUserNumber = 5
    independentFieldUserNumber = 6
    equationsSetUserNumber = 1
    problemUserNumber = 1

    dataPointUserNumber = 1
    dataProjectionUserNumber = 1

    # Set all diganostic levels on for testing
    #iron.DiagnosticsSetOn(iron.DiagnosticTypes.ALL,[1,2,3,4,5],"Diagnostics",["DOMAIN_MAPPINGS_LOCAL_FROM_GLOBAL_CALCULATE"])

    numberOfMeshComponents = 1
    InterpolationType = 1
    if(numberGlobalZElements==0):
        numberOfXi = 2
    else:
        numberOfXi = 3

    # Get the number of computational nodes and this computational node number
    numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
    computationalNodeNumber = iron.ComputationalNodeNumberGet()

    # Create a 3D rectangular cartesian coordinate system
    coordinateSystem = iron.CoordinateSystem()
    coordinateSystem.CreateStart(coordinateSystemUserNumber)
    coordinateSystem.DimensionSet(3)
    coordinateSystem.CreateFinish()

    # Create a region and assign the coordinate system to the region
    region = iron.Region()
    region.CreateStart(regionUserNumber,iron.WorldRegion)
    region.LabelSet("Region")
    region.coordinateSystem = coordinateSystem
    region.CreateFinish()

    # Define basis
    basis = iron.Basis()
    basis.CreateStart(basisUserNumber)
    if InterpolationType in (1,2,3,4):
        basis.type = iron.BasisTypes.LAGRANGE_HERMITE_TP
    elif InterpolationType in (7,8,9):
        basis.type = iron.BasisTypes.SIMPLEX
    basis.numberOfXi = numberOfXi
    basis.interpolationXi = [iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfXi
    if(NumberOfGaussXi>0):
        basis.quadratureNumberOfGaussXi = [NumberOfGaussXi]*numberOfXi
    basis.CreateFinish()

    mesh = iron.Mesh()
    if useGeneratedMesh:
        # Start the creation of a generated mesh in the region
        generatedMesh = iron.GeneratedMesh()
        generatedMesh.CreateStart(generatedMeshUserNumber,region)
        generatedMesh.type = iron.GeneratedMeshTypes.REGULAR
        generatedMesh.basis = [basis]
        generatedMesh.extent = [width,length,height]
        generatedMesh.numberOfElements = [numberGlobalXElements,numberGlobalYElements,numberGlobalZElements]
        # Finish the creation of a generated mesh in the region
        generatedMesh.CreateFinish(meshUserNumber,mesh)

    else:
        totalNumberOfNodes = 8
        totalNumberOfElements = 1
        # Start the creation of a manually generated mesh in the region
        mesh = iron.Mesh()
        mesh.CreateStart(meshUserNumber,region,numberOfXi)
        mesh.NumberOfComponentsSet(numberOfMeshComponents)
        mesh.NumberOfElementsSet(totalNumberOfElements)

        #Define nodes for the mesh
        nodes = iron.Nodes()
        nodes.CreateStart(region,totalNumberOfNodes)
        nodes.CreateFinish()

        elements = iron.MeshElements()
        meshComponentNumber=1
        elements.CreateStart(mesh,meshComponentNumber,basis)
        elements.NodesSet(1,[1,2,3,4,5,6,7,8])
        elements.CreateFinish()

        mesh.CreateFinish()

    # Create a decomposition for the mesh
    decomposition = iron.Decomposition()
    decomposition.CreateStart(decompositionUserNumber,mesh)
    decomposition.type = iron.DecompositionTypes.CALCULATED
    decomposition.numberOfDomains = numberOfComputationalNodes
    decomposition.CreateFinish()

    # Create a field for the geometry
    geometricField = iron.Field()
    geometricField.CreateStart(geometricFieldUserNumber,region)
    geometricField.MeshDecompositionSet(decomposition)
    geometricField.TypeSet(iron.FieldTypes.GEOMETRIC)
    geometricField.VariableLabelSet(iron.FieldVariableTypes.U,"Geometry")
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,1,1)
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,2,1)
    geometricField.ComponentMeshComponentSet(iron.FieldVariableTypes.U,3,1)
    if InterpolationType == 4:
        geometricField.fieldScalingType = iron.FieldScalingTypes.ARITHMETIC_MEAN
    geometricField.CreateFinish()

    if useGeneratedMesh:
        # Update the geometric field parameters from generated mesh
        generatedMesh.GeometricParametersCalculate(geometricField)
    else:
        # Update the geometric field parameters manually
        geometricField.ParameterSetUpdateStart(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)
        # node 1
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,1,1,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,1,2,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,1,3,0.0)
        # node 2
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,2,1,height)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,2,2,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,2,3,0.0)
        # node 3
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,3,1,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,3,2,width)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,3,3,0.0)
        # node 4
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,4,1,height)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,4,2,width)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,4,3,0.0)
        # node 5
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,5,1,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,5,2,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,5,3,length)
        # node 6
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,6,1,height)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,6,2,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,6,3,length)
        # node 7
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,7,1,0.0)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,7,2,width)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,7,3,length)
        # node 8
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,8,1,height)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,8,2,width)
        geometricField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,8,3,length)
        geometricField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES)



    #### READ DATA IN #####################################


    # Create the data points
    dataPointsFromFile = myExFile.readExFile(directory+'data_coordinates.exdata')
    dataList = np.arange(1,dataPointsFromFile.shape[0]+1)
    numberOfDataPoints = len(dataList)
    print("Number of data points: " + str(numberOfDataPoints))
    dataPointLocations = dataPointsFromFile[:,1:4]


    dataPoints = iron.DataPoints()
    dataPoints.CreateStart(dataPointUserNumber, region, numberOfDataPoints)
    for dataPointIdx, dataPoint in enumerate(dataList):
        dataPoints.PositionSet(dataPoint,dataPointLocations[dataPointIdx,:])
    dataPoints.CreateFinish()

    # Set up data projection
    dataProjection = iron.DataProjection()
    dataProjection.CreateStart(dataProjectionUserNumber, dataPoints,
                               geometricField, iron.FieldVariableTypes.U)
    dataProjection.AbsoluteToleranceSet(1.0e-14)
    dataProjection.RelativeToleranceSet(1.0e-14)
    dataProjection.MaximumNumberOfIterationsSet(int(1e9))
    dataProjection.ProjectionTypeSet(
        iron.DataProjectionProjectionTypes.ALL_ELEMENTS)
    dataProjection.CreateFinish()

    # Evaluate data projection based on geometric field
    dataProjection.DataPointsProjectionEvaluate(
        iron.FieldParameterSetTypes.VALUES)
    # for dataPointIdx, dataPoint in enumerate(dataList):
    #     print('Data point: ' + str(dataPointLocations[dataPointIdx,:]))
    #     elementNumber = dataProjection.ResultElementNumberGet(dataPoint)
    #     print('    ElementNumber: ' + str(elementNumber))
    #     xi = dataProjection.ResultXiGet(dataPoint,numberOfDimensions)
    #     print('    Xi: ' + str(xi))
    #     distance = dataProjection.ResultDistanceGet(dataPoint)
    #     print('    Distance: ' + str(distance))

    # Create mesh topology for data projection
    mesh.TopologyDataPointsCalculateProjection(dataProjection)
    # Create decomposition topology for data projection
    decomposition.TopologyDataProjectionCalculate()

    # dataProjection.ResultAnalysisOutput("")

    dataErrorVector = np.zeros((numberOfDataPoints, numberOfDimensions))
    dataErrorDistance = np.zeros(numberOfDataPoints)
    for dataPointIdx, dataPoint in enumerate(dataList):
        dataErrorDistance[dataPointIdx] = dataProjection.ResultDistanceGet(dataPoint)
        dataErrorVector[dataPointIdx,:] = dataProjection.ResultProjectionVectorGet(dataPoint, numberOfDimensions)



    tensorsFromFile = myExFile.readExFile(directory+'syntheticTensors.exdata')
    D11 = tensorsFromFile[:,1]
    D12 = tensorsFromFile[:,2]
    D13 = tensorsFromFile[:,3]
    D22 = tensorsFromFile[:,5]
    D23 = tensorsFromFile[:,6]
    D33 = tensorsFromFile[:,9]







    # write data points to exdata file for CMGUI
    # offset = 0
    # exfile.writeExdataFile(
    #     "DataPoints.part" + str(computationalNodeNumber) + ".exdata",
    #     dataPointLocations, dataErrorVector, dataErrorDistance, offset)
    print("Projection complete")

    equationsSetField = iron.Field()
    equationsSet = iron.EquationsSet()

    if nonlinearFit:
        equationsSetSpecification = [iron.EquationsSetClasses.FITTING,
                                     iron.EquationsSetTypes.DATA_FITTING_EQUATION,
                                     iron.EquationsSetSubtypes.DIFFUSION_TENSOR_FIBRE_FITTING,
                                     smoothingType]
    else:
        equationsSetSpecification = [iron.EquationsSetClasses.FITTING,
                                     iron.EquationsSetTypes.DATA_FITTING_EQUATION,
                                     iron.EquationsSetSubtypes.DATA_POINT_FITTING,
                                     smoothingType]

    equationsSet.CreateStart(equationsSetUserNumber, region, geometricField,
                             equationsSetSpecification,
                             equationsSetFieldUserNumber, equationsSetField)
    equationsSet.CreateFinish()


    # Create the dependent field
    dependentField = iron.Field()
    equationsSet.DependentCreateStart(dependentFieldUserNumber, dependentField)
    dependentField.VariableLabelSet(iron.FieldVariableTypes.U,"Quaternions")
    dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.U, 4)
    dependentField.NumberOfComponentsSet(iron.FieldVariableTypes.DELUDELN, 4)
    if InterpolationType == 4:
        dependentField.fieldScalingType = iron.FieldScalingTypes.ARITHMETIC_MEAN
    equationsSet.DependentCreateFinish()

    angle = 90/180.0*pi
    axis = [0.5,0,0.5]
    magnitude = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
    axis_unit = [axis[0] / magnitude, axis[1] / magnitude, axis[2] / magnitude]

    a = cos(angle/2)
    b = sin(angle/2)*axis_unit[0]
    c = sin(angle/2)*axis_unit[1]
    d = sin(angle/2)*axis_unit[2]

    # initialise dependent field
    dependentField.ParameterSetUpdateStart(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
    # node 1
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 1, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 1, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 1, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 1, 4, d)
    # node 2
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 2, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 2, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 2, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 2, 4, d)
    # node 3
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 3, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 3, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 3, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 3, 4, d)
    # node 4
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 4, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 4, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 4, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 4, 4, d)
    # node 5
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 5, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 5, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 5, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 5, 4, d)
    # node 6
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 6, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 6, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 6, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 6, 4, d)
    # node 7
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 7, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 7, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 7, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 7, 4, d)
    # node 8
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 8, 1, a)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 8, 2, b)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 8, 3, c)
    dependentField.ParameterSetUpdateNodeDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, 1, 1, 8, 4, d)

    dependentField.ParameterSetUpdateFinish(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)



    # Create tensor field (independent field)
    independentField = iron.Field()
    independentField.CreateStart(independentFieldUserNumber, region)
    independentField.TypeSet(iron.FieldTypes.GENERAL)
    independentField.MeshDecompositionSet(decomposition)
    independentField.GeometricFieldSet(geometricField)
    independentField.DependentTypeSet(iron.FieldDependentTypes.INDEPENDENT)
    independentField.NumberOfVariablesSet(2)
    independentField.VariableTypesSet([iron.FieldVariableTypes.U, iron.FieldVariableTypes.V])
    independentField.VariableLabelSet(iron.FieldVariableTypes.U, "DiffusionTensor")
    independentField.VariableLabelSet(iron.FieldVariableTypes.V, "Weights")

    independentField.DataProjectionSet(dataProjection)

    independentField.NumberOfComponentsSet(iron.FieldVariableTypes.U, 6)
    independentField.NumberOfComponentsSet(iron.FieldVariableTypes.V, 6)

    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, 1, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, 2, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, 3, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, 4, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, 5, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.U, 6, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.V, 1, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.V, 2, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.V, 3, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.V, 4, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.V, 5, iron.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(iron.FieldVariableTypes.V, 6, iron.FieldInterpolationTypes.DATA_POINT_BASED)

    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 1, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 2, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 3, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 4, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 5, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 6, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.V, 1, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.V, 2, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.V, 3, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.V, 4, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.V, 5, 1)
    independentField.ComponentMeshComponentSet(iron.FieldVariableTypes.V, 6, 1)



    independentField.CreateFinish()

    equationsSet.IndependentCreateStart(independentFieldUserNumber, independentField)
    equationsSet.IndependentCreateFinish()


    # loop over each element's data points and set independent field values to data point locations
    elementNumber = 1
    elementDomain = decomposition.ElementDomainGet(elementNumber)
    if (elementDomain == computationalNodeNumber):
        numberOfProjectedDataPoints = decomposition.TopologyNumberOfElementDataPointsGet(1)
        for dataPoint in range(numberOfProjectedDataPoints):
            dataPointId = dataPoint +1

            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 1, D11[dataPoint])
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 2, D12[dataPoint])
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 3, D13[dataPoint])
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 4, D22[dataPoint])
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 5, D23[dataPoint])
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 6, D33[dataPoint])

            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 1, 1)
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 2, 1)
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 3, 1)
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 4, 1)
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 5, 1)
            independentField.ParameterSetUpdateElementDataPointDP(iron.FieldVariableTypes.V, iron.FieldParameterSetTypes.VALUES, elementNumber, dataPointId, 6, 1)

    # Create material field (Sobolev parameters)

    if smoothing:
        materialField = iron.Field()
        equationsSet.MaterialsCreateStart(materialFieldUserNumber,materialField)
        materialField.VariableLabelSet(iron.FieldVariableTypes.U,"SmoothingParameters")
        equationsSet.MaterialsCreateFinish()

        # Set kappa and tau - Sobolev smoothing parameters
        materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,tau)
        materialField.ComponentValuesInitialiseDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,2,kappa)

    # Create equations
    equations = iron.Equations()
    equationsSet.EquationsCreateStart(equations)
    equations.sparsityType = iron.EquationsSparsityTypes.SPARSE
    # equations.outputType = iron.EquationsOutputTypes.ELEMENT_MATRIX
    # equations.outputType = iron.EquationsOutputTypes.MATRIX
    equationsSet.EquationsCreateFinish()

    # =================================================================
    # Problem setup
    # =================================================================

    # Create fitting problem
    problem = iron.Problem()
    if nonlinearFit:
        problemSpecification = [iron.ProblemClasses.FITTING,
                                iron.ProblemTypes.FIBRE_FITTING,
                                iron.ProblemSubtypes.STATIC_NONLINEAR_FITTING]
    else:
        problemSpecification = [iron.ProblemClasses.FITTING,
                                iron.ProblemTypes.DATA_FITTING,
                                iron.ProblemSubtypes.STATIC_FITTING]
    problem.CreateStart(problemUserNumber, problemSpecification)
    problem.CreateFinish()

    # Create control loops
    problem.ControlLoopCreateStart()
    problem.ControlLoopCreateFinish()

    # Create problem solver
    if nonlinearFit:
        # Create problem solver
        nonLinearSolver = iron.Solver()
        linearSolver = iron.Solver()
        problem.SolversCreateStart()
        problem.SolverGet([iron.ControlLoopIdentifiers.NODE],1,nonLinearSolver)
        nonLinearSolver.outputType = iron.SolverOutputTypes.PROGRESS
        nonLinearSolver.NewtonJacobianCalculationTypeSet(iron.JacobianCalculationTypes.FD)
        nonLinearSolver.NewtonLinearSolverGet(linearSolver)
        nonLinearSolver.NewtonAbsoluteToleranceSet(1E-14)
        nonLinearSolver.NewtonSolutionToleranceSet(1E-14)
        nonLinearSolver.NewtonRelativeToleranceSet(1E-14)
        nonLinearSolver.NewtonMaximumIterationsSet(1000)
        linearSolver.linearType = iron.LinearSolverTypes.ITERATIVE
        problem.SolversCreateFinish()
    else:
        # Create problem solver
        solver = iron.Solver()
        problem.SolversCreateStart()
        problem.SolverGet([iron.ControlLoopIdentifiers.NODE], 1, solver)
        solver.outputType = iron.SolverOutputTypes.NONE
        # solver.outputType = iron.SolverOutputTypes.MATRIX
        solver.linearType = iron.LinearSolverTypes.ITERATIVE
        # solver.LibraryTypeSet(iron.SolverLibraries.UMFPACK) # UMFPACK/SUPERLU
        solver.linearIterativeAbsoluteTolerance = 1.0E-10
        solver.linearIterativeRelativeTolerance = 1.0E-05
        problem.SolversCreateFinish()

    # Create solver equations and add equations set to solver equations
    solver = iron.Solver()
    solverEquations = iron.SolverEquations()
    problem.SolverEquationsCreateStart()
    problem.SolverGet([iron.ControlLoopIdentifiers.NODE], 1, solver)
    solver.SolverEquationsGet(solverEquations)
    solverEquations.sparsityType = iron.SolverEquationsSparsityTypes.SPARSE
    equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
    problem.SolverEquationsCreateFinish()

    # # =================================================================
    # # Boundary Conditions
    # # =================================================================
    #
    # Create boundary conditions and set first and last nodes to 0.0 and 1.0
    boundaryConditions = iron.BoundaryConditions()
    solverEquations.BoundaryConditionsCreateStart(boundaryConditions)
    #
    meshNodes = iron.MeshNodes()
    mesh.NodesGet(1,meshNodes)
    # for node in range(1,meshNodes.NumberOfNodesGet()+1):
    #     xValue = geometricField.ParameterSetGetNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,node,1)
    #     yValue = geometricField.ParameterSetGetNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,node,2)
    #     zValue = geometricField.ParameterSetGetNodeDP(iron.FieldVariableTypes.U,iron.FieldParameterSetTypes.VALUES,1,1,node,3)
    #     # Set x=0 nodes to no x,y,z displacment
    #     if np.isclose(xValue, 0.0):
    #         for component in [1,2,3]:
    #             boundaryConditions.AddNode(dependentField,
    #                                        iron.FieldVariableTypes.U, 1, 1,
    #                                        node, component,
    #                                        iron.BoundaryConditionsTypes.FIXED,
    #                                        0.0)
    solverEquations.BoundaryConditionsCreateFinish()


    # Export undeformed mesh geometry
    print("Writing undeformed geometry")
    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("UndeformedGeometry", "FORTRAN")
    fields.ElementsExport("UndeformedGeometry", "FORTRAN")
    fields.Finalise()



    # =================================================================
    # S o l v e    a n d    E x p o r t    D a t a
    # =================================================================
    for iteration in range(1, numberOfFitIterations + 1):

        # Solve the problem
        print("Solving fitting problem, iteration: " + str(iteration))
        problem.Solve()

        # Export fields
        print("Writing deformed geometry")
        fields = iron.Fields()
        fields.CreateRegion(region)
        fields.NodesExport("DeformedGeometry" + str(iteration), "FORTRAN")
        fields.ElementsExport("DeformedGeometry" + str(iteration), "FORTRAN")
        fields.Finalise()

    problem.Destroy()
    if useGeneratedMesh:
      generatedMesh.Destroy()
    basis.Destroy()
    region.Destroy()
    coordinateSystem.Destroy()


    iron.Finalise()


if __name__ == "__main__":
    # Solving a model
    numberGlobalXElements = 1
    numberGlobalYElements = 1
    numberGlobalZElements = 1
    directory = os.getcwd()+"/d/"
    fit(numberGlobalXElements, numberGlobalYElements,numberGlobalZElements,directory)
