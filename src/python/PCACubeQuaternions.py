import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import subprocess
import sys
import os
try:
    from myExFile import *
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'myPythonLib'))
    from myExFile import *
from matplotlib import cm
import nlopt
import createToyTensors as tensors
# import myIO as io



def evaluateObjectiveFct(quaternions,grad=None):

    # we need the option grad for the optimiser, but it doesnt make sense when only evaluating function value
    if grad:
        if grad.size > 0:
            raise NotImplemented('Gradient requested')

    # create an input exnode file that defines the fibre angles at each node.
    #writeLinearCubeQuaternionAroundZNodeFile("linear_quat.exnode", float(angle[0]), float(angle[1]))

    numberOfNodes = 8
    nodeNumberArray = np.linspace(1,numberOfNodes,numberOfNodes)

    quat = np.reshape(quaternions, (numberOfNodes,-1))
    field = np.vstack((nodeNumberArray,quat.T)).T

    writeExFile('linear_quat', field, 'Quaternions', ['1', '2', '3', '4'])


    # call cmgui to read above exnode file and evaluate the objective function.
    if sys.platform == "darwin":
        out = subprocess.check_output(["/Applications/Cmgui.app/Contents/MacOS/Cmgui -no_display quaternionEvaluateObjF.cmgui"], shell=False, stderr=subprocess.STDOUT).strip()
    else:
        out = subprocess.check_output(["cmgui_new -no_display quaternionEvaluateObjF.cmgui"], shell=True, stderr=subprocess.STDOUT).strip()

    # cmgui returns the objective function as a command; we need to extract the obj fct value
    #objStr = out[out.index("composite") + len("composite"):-1].strip()
    #objF,objN = [float(i) for i in objStr.split()]

    objF = readObjFunctionfromExFile('objF.exdata', 7)[0]
    # print out
    # print quaternions


    return objF


# def evaluateObjectiveFct(angle,grad=None):
#
#     # we need the option grad for the optimiser, but it doesnt make sense when only evaluating function value
#     if grad:
#         if grad.size > 0:
#             raise NotImplemented('Gradient requested')
#
#     a = angle[0]
#     b = angle[1]
#
#     #objF = a**2 + b ** 2
#
#     #booth function
#     objF = (a+2*b-7)**2 + (2*a+b-5)**2
#
#     #Himmelsblau function
#     #objF = (a**2+b-11)**2+(a+b**2-7)**2
#
#     return objF




def evaluateHessian(optPoint, stepSize):
    n = len(optPoint)
    h = stepSize

    ee = np.zeros([n, n])
    for i in range(0, n):
        ee[i, i] = h

    # First order derivatives
    A = np.zeros(n)
    B = np.zeros(n)
    g = np.zeros(n)

    for i in range(0, n):
        # Central difference approximation
        A[i] = evaluateObjectiveFct(optPoint+ee[:, i])
        B[i] = evaluateObjectiveFct(optPoint-ee[:, i])
        g[i] = (A[i] - B[i])/(2*ee[i,i])

    # Second order derivatives.
    C = np.zeros(n)
    D = np.zeros(n)
    E = np.zeros(n)
    F = np.zeros(n)
    H = np.zeros([n, n])

    for i in range(0, n):
        C = evaluateObjectiveFct(optPoint + ee[:, i] + ee[:, i])
        E = evaluateObjectiveFct(optPoint)
        F = evaluateObjectiveFct(optPoint - ee[:, i] - ee[:, i])

        H[i, i] = (-C + 16 * A[i] - 30 * E + 16 * B[i] - F) / (12 * ee[i, i] * ee[i, i])
        for j in range(i + 1, n):
            G = evaluateObjectiveFct(optPoint + ee[:, i] + ee[:, j])
            I = evaluateObjectiveFct(optPoint + ee[:, i] - ee[:, j])
            J = evaluateObjectiveFct(optPoint - ee[:, i] + ee[:, j])
            K = evaluateObjectiveFct(optPoint - ee[:, i] - ee[:, j])
            H[i, j] = (G - I - J + K) / (4 * ee[i, i] * ee[j, j])
            H[j, i] = H[i, j]

    return H



def optimize(tol,init):

    opt = nlopt.opt(nlopt.LN_COBYLA, len(init))
    opt.set_min_objective(evaluateObjectiveFct)

    #opt.set_xtol_rel(tol)
    opt.set_ftol_rel(tol)

    return opt.optimize(init)




def evaluateHessianAndPerfomPCA(x=None, stepSizeHessian = None):

    if x is None:
        optimize()

    if stepSizeHessian is None:
        stepSizeHessian = 0.01

    print "Minimum at ",x
    min = evaluateObjectiveFct(x)
    print "Obj Fct at Min", min

    h = evaluateHessian(x,stepSizeHessian)

    print "-----------------"
    print "Hessian matrix with stepsize", stepSizeHessian
    if np.linalg.det(h) > 0 and h[0][0]:
        print "local minimum"
    else:
        print "not a local minimum"

    print "Hessian at minimum", np.dot(np.dot(np.transpose(x), h), x)
    l,e = np.linalg.eig(h)
    e1 = e[0,:]

    print "eigenvalues", l
    print "eigenvectors"
    print e

    s_squared = min*288/(288-32)
    c = 2*s_squared*np.linalg.inv(h)

    print "-----------------"
    print "Covariance matrix"
    print c

    l_c,e_c = np.linalg.eig(c)

    print "eigenvalues", l_c
    print "eigenvectors "
    print e_c

    print "percentage of variance", abs(l_c)/np.sum(abs(l_c))*100

    print "-----------------"
    print "First and second mode"

    xMode1p = x+e_c[:,0]*l_c[0]
    xMode1m = x-e_c[:,0]*l_c[0]
    xMode2p = x+e_c[:,1]*l_c[1]
    xMode2m = x-e_c[:,1]*l_c[1]

    print xMode1p
    print xMode1m
    print xMode2p
    print xMode2m

    return h, c, x





def plotFctAndMatrices(x,h,c):


    l_c,e_c = np.linalg.eig(c)
    xMode1p = x+e_c[:,0]*l_c[0]
    xMode1m = x-e_c[:,0]*l_c[0]
    xMode2p = x+e_c[:,1]*l_c[1]
    xMode2m = x-e_c[:,1]*l_c[1]

    value=np.amax(np.abs([xMode2p-x,xMode1p-x]))*2
    steps = 10
    X = np.linspace(x[0]-value,x[0]+value,steps)
    Y = np.linspace(x[1]-value,x[1]+value,steps)

    Xv, Yv = np.meshgrid(X, Y)
    Obj = np.array([evaluateObjectiveFct([a, b]) for a, b in zip(np.ravel(Xv), np.ravel(Yv))])
    Hes = np.array([np.dot(np.dot(np.transpose([x[0]-a, x[1]-b]),h),[x[0]-a, x[1]-b]) for a, b in zip(np.ravel(Xv), np.ravel(Yv))])
    Cov = np.array([np.dot(np.dot(np.transpose([x[0]-a, x[1]-b]),c),[x[0]-a, x[1]-b]) for a, b in zip(np.ravel(Xv), np.ravel(Yv))])

    ObjV = Obj.reshape(Xv.shape)
    HesV = Hes.reshape(Xv.shape)
    CovV = Cov.reshape(Xv.shape)

    fig1 = plt.figure()
    fig1.suptitle('Objective function')
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(Xv,Yv,ObjV,cmap=cm.coolwarm)
    ax1.scatter(x[0], x[1], evaluateObjectiveFct(x))
    ax1.scatter(xMode1p[0], xMode1p[1], evaluateObjectiveFct(xMode1p))
    ax1.scatter(xMode1m[0], xMode1m[1], evaluateObjectiveFct(xMode1m))
    ax1.scatter(xMode2p[0], xMode2p[1], evaluateObjectiveFct(xMode2p))
    ax1.scatter(xMode2m[0], xMode2m[1], evaluateObjectiveFct(xMode2m))

    fig2 = plt.figure()
    fig2.suptitle('Hessian')
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(Xv,Yv,HesV)

    fig3 = plt.figure()
    fig3.suptitle('Covariance')
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(Xv,Yv,CovV)

    return 1


def plotObjFct(x, region):

    steps = 21
    value = region

    X = np.linspace(x[0]-value,x[0]+value,steps)
    Y = np.linspace(x[1]-value,x[1]+value,steps)

    Xv, Yv = np.meshgrid(X, Y)
    Obj = np.array([evaluateObjectiveFct([a, b]) for a, b in zip(np.ravel(Xv), np.ravel(Yv))])
    ObjV = Obj.reshape(Xv.shape)

    Xv = Xv-x[0]
    Yv = Yv - x[1]
    fig1 = plt.figure()
    fig1.suptitle('Objective function')
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(Xv,Yv,ObjV,cmap=cm.coolwarm)
    ax1.scatter(0, 0, evaluateObjectiveFct(x), c='r')

    return 1


def optimiseHessianStepsize(x):

    # The stepsize of the hessian cannot be smaller than the tolerance - otherwise the optimum might not actually be a minimum
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111, projection='3d')
    steps = 11
    value = 1

    X = np.linspace(x[0]-value,x[0]+value,steps)
    Y = np.linspace(x[1]-value,x[1]+value,steps)
    Xv, Yv = np.meshgrid(X, Y)

    points = []
    minObj = evaluateObjectiveFct(x)
    #for h in np.linspace(0.2,10,10):
    for h in np.logspace(-3, 0, 11):
        xS = x + np.full((1,32), h)
        hes = evaluateHessian(x, h)
        det = np.linalg.det(hes)
        #print det

        #Hes = np.array([np.dot(np.dot(np.transpose([x[0] - a, x[1] - b]), hes), [x[0] - a, x[1] - b]) for a, b in zip(np.ravel(Xv), np.ravel(Yv))])
        #HesV = Hes.reshape(Xv.shape)
        #ax0.plot_surface(Xv, Yv, HesV, alpha=0.2)
        points.append([np.log10(h),evaluateObjectiveFct(xS)-minObj,det, hes[0,0], hes[1,1], hes[0,1],np.dot(np.dot(np.transpose(x), hes), x)/1000,evaluateObjectiveFct(xS)])


    pointsArray = np.vstack(points)
    fig1 = plt.figure()
    pointsArray= np.vstack(points)
    plt.plot(pointsArray[:,0],pointsArray[:,1],label='Obj change')
    plt.plot(pointsArray[:,0],pointsArray[:,2],label='det')
    plt.plot(pointsArray[:,0],pointsArray[:,3],label='hes 11')
    plt.plot(pointsArray[:,0],pointsArray[:,4],label='hes 22')
    plt.plot(pointsArray[:, 0], pointsArray[:, 5],label='hes 12=21')
    #fig2 = plt.figure()
    #plt.plot(pointsArray[:, 0], pointsArray[:, 6],label='hes value')

    plt.legend()

    return 1



def optimiseMinimisingStepsize(init):

    opt1 = nlopt.opt(nlopt.LN_COBYLA, 2)
    opt1.set_min_objective(evaluateObjectiveFct)
    opt2 = nlopt.opt(nlopt.LN_COBYLA, 2)
    opt2.set_min_objective(evaluateObjectiveFct)

    points1=[]
    points2 = []

    for h in np.logspace(-5, -1, 5):
        opt1.set_ftol_rel(h)
        x1 = opt1.optimize(init)
        print "ftol of 1e", str(np.log10(h)), "with optimum of",evaluateObjectiveFct(x1), "at", x1
        points1.append([np.log10(h),evaluateObjectiveFct(x1)])

    print "-----"

    for i in np.logspace(-5, -1, 5):
        opt2.set_xtol_rel(i)
        x2 = opt2.optimize(init)
        print "xtol of", str(np.log10(i)), "with optimum of", evaluateObjectiveFct(x2), "at", x2
        points2.append([np.log10(i), evaluateObjectiveFct(x2)])

    pointsArray1 = np.vstack(points1)
    pointsArray2 = np.vstack(points2)
    fig1 = plt.figure()
    plt.plot(pointsArray1[:,0],pointsArray1[:,1],label='Obj fct')
    plt.plot(pointsArray2[:, 0], pointsArray2[:, 1], label='Obj fct')

    return 1


def writeFirstMode(x,c):

    l_c,e_c = np.linalg.eig(c)
    xMode1p = x+e_c[:,0]*l_c[0]
    xMode1m = x-e_c[:,0]*l_c[0]

    numberOfNodes = 8
    nodeNumberArray = np.linspace(1,numberOfNodes,numberOfNodes)

    mode1 = np.reshape(xMode1p, (numberOfNodes,-1))
    field1 = np.vstack((nodeNumberArray,mode1.T)).T

    mode2 = np.reshape(xMode1m, (numberOfNodes,-1))
    field2 = np.vstack((nodeNumberArray,mode2.T)).T

    writeExFile('linear_quat_mode1p', field1, 'quaternionField', ['w', 'x', 'y', 'z'])
    writeExFile('linear_quat_mode1m', field2, 'quaternionField', ['w', 'x', 'y', 'z'])

    return 1
#
# init = [-45, 45]
# noise = 0.01
#
# # 0) create a new toy problem
# tensors.createToyTensors(noise,12)
#
# # 1) check for convergence of function value tolerance to pick the correct tolerance
# optimiseMinimisingStepsize(init)
#
# # 2) find optimal point for choosen tolerance
# xmin = optimize(1e-5,init)
#
# # 3) look at objective function surface very close to optimum point (region is size of the region around the function opt)
# plotObjFct(xmin, 0.001)
#
# # 4) look at Hessian stepsizes
# optimiseHessianStepsize(xmin)
#
# # 5) evaluate Hessian and Covariance as well as eigenanalysis
# h,c,_ = evaluateHessianAndPerfomPCA(x=xmin,stepSizeHessian=0.001)
#
# # 6) plot results
# plotFctAndMatrices(xmin,h,c)
#
#
# plt.show()

angle = 40/180.0*np.pi
a = np.cos(angle/2)
b = np.sin(angle/2)
c = 0
d = 0

init = [a,b,c,d, a,b,c,d, a,b,c,d, a,b,c,d, a,b,c,d, a,b,c,d, a,b,c,d, a,b,c,d]


evaluateObjectiveFct(init)


h,c,_ = evaluateHessianAndPerfomPCA(x=init,stepSizeHessian=0.00001)
