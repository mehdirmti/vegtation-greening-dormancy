"""
Created on Wednsday Sep 28 08:00:00 2022
Revised on Tuesday May 23 13:15:00 2023


@author: Mehdi Rahmati
E-mail: mehdirmti@gmail.com, m.rahmati@fz-juelich.de

Description:
This script provides functions to implement the Group Method of Data Handeling (GMDH) framework (DOI: 10.1080/00031305.1981.10479358).
The algorithm develops a gray-box network to predict the target variable from several other independent variables.

More detailed information can be found in the article published at the following link:
https://www.XXX.XXX

Inputs and outputs of main functions:
    ====================================================================================
    The main fucntion to train the networkd is named:
    GMDH(params, X, Y):

    @inputs:
    --------
    params: is a dict with the initial sets of parameters of netwrok: 

        params = {'MaxLayerNeurons': 5, 'MaxLayers': 1, 'alpha':0.85, 'pTrain':0.7}

    X: is a matrix (m*n) with predictors
    each row corresponds to a variable        
    each column corresponds to a point in the sampling range (it can be a spatial or a temporal range)
    y: is a vector (1*n) containing the target variables for the same sampling time as in X

    @Outputs:
    ---------
    The output of this function is the developed GMDH network
    
    ====================================================================================
    When the network is developed, you can use the ApplyGMDH function to use the developed networkd for further applications.
    ApplyGMDH(gmdh, X)

    gmdh: is the network developed by GMDH(params, X, Y) function.
    X: a matrix (m*n) of predictors

    ====================================================================================
    To determine which variables are included in the developed model, you can use: 
    IdentifySelectedVariables(GMDH_Network)

    This function takes the developed mesh and returns the input variables that are entered into the model


To be cited as: 
Rahmati et al., 2023.The continuous increase in evaporative demand shortened the growing season of European ecosystems 
in the last decade. Comunications Earth & Environement, XX, XX-XX. 

"""
import numpy as np
from collections import defaultdict

def CreateReressorsMatrix (x):
    X = np.empty((6, x.shape[1]))
    X[0,:] = np.ones(x.shape[1])
    X[1,:] = x[0,:]
    X[2,:] = x[1,:]
    X[3,:] = x[0,:]**2
    X[4,:] = x[1,:]**2
    X[5,:] = x[0,:]*x[1,:]
    return X

def FitPolynomial (x1,Y1, x2, Y2, vars):
    X1 = CreateReressorsMatrix (x1)
    
    c = np.matmul(Y1, np.linalg.pinv(X1))

    Y1hat = np.matmul(c, X1)
    e1 = Y1 - Y1hat
    MSE1 = np.mean(e1**2)
    RMSE1 = np.sqrt(MSE1)
    
    def f(x):
        return np.matmul(c, CreateReressorsMatrix (x))
    
    Y2hat = f(x2) 

    e2 = Y2 - Y2hat
    MSE2 = np.mean(e2**2)
    RMSE2 = np.sqrt(MSE2)
    d2 = np.sum(e2**2)
    p = {'vars': vars, 'c': c, 'f': f, 'Y1hat': Y1hat, 'MSE1': MSE1, 'RMSE1': RMSE1, 'Y2hat': Y2hat, 'MSE2': MSE2, 'RMSE2': RMSE2, 'd2': d2}
    return p     

def GetPolynomialLayer (X1,Y1, X2, Y2):
    n = X1.shape[0]
    N = int(n * (n-1)/2)
    template = FitPolynomial(np.random.rand(2,10), np.random.rand(1,10), np.random.rand(2,10), np.random.rand(1,10), [])
    L = defaultdict(list)
    for k in range(N):
        L[k].append(template)
    k=0
    for i in range(n-1):
        for j in range(i+1,n):
            L[k] = FitPolynomial(X1[[i,j],:], Y1, X2[[i,j],:], Y2, [i,j])
            k += 1
    LRMSE2 = [L[k]['RMSE2'] for k in range(len(L))] 

    # to screen out non fited models  
    ind = np.isnan(LRMSE2)
    id = np.arange(0, len(L))
    id = id[~ind]
    L = [L[k] for k in id]
    LRMSE2 = [LRMSE2[k] for k in id]
    sortorder = np.argsort(LRMSE2)
    L2 = [L[k] for k in sortorder]
    return L2

def GMDH(params, X, Y):

    MaxLayerNeurons = params['MaxLayerNeurons']
    MaxLayers = params['MaxLayers']
    alpha = params['alpha']

    nData = X.shape[1]
    
    # Shuffle Data
    Permutation = np.random.permutation(nData)
    X = X[:,Permutation]
    Y = Y[:,Permutation]

    # Divide Data
    pTrainData = params['pTrain']
    nTrainData = round(pTrainData*nData)
    X1 = X[:,0:nTrainData]
    Y1 = Y[:,0:nTrainData]
    pTestData = 1 - pTrainData
    nTestData = nData - nTrainData
    X2 = X[:,nTrainData:nData]
    Y2 = Y[:,nTrainData:nData]

    Layers = []

    Z1=X1
    Z2=X2

    for l in range(MaxLayers):

        # Polynomial Generration
        L = GetPolynomialLayer (Z1,Y1,Z2,Y2)

        if len(L) == 0:
            break

        # Criteria for Selection
        ec = alpha*L[0]['RMSE2'] + (1-alpha)*L[len(L)-1]['RMSE2']
        ec = max(ec, L[0]['RMSE2'])
        LRMSE2 = [L[k]['RMSE2'] for k in range(len(L))]  
        ind = np.arange(0, len(L))[LRMSE2 <=ec]

        if ind.shape[0] != 1:
            L = L[0:np.max(ind)]

        # Puting a limit on Neurons Number 
        if len(L) > MaxLayerNeurons:
            L = L[0:MaxLayerNeurons-1]
        
        # Setting the Neurons number of the last layer to be 1
        if np.logical_and(l == MaxLayers-1, len(L) > 1):
            L2 = L[0]
            L = []
            L.append(L2) 
            
        # Storing L's in cell
        Layers.append(L)
        Z1 = np.empty((len(L), nTrainData))
        Z2 = np.empty((len(L), nTestData))
        for i in range(len(L)):
            Z1[i,:] = L[i]['Y1hat']
            Z2[i,:] = L[i]['Y2hat']
            
        if len (L) == 1:
            break

    # Equalizing the number of the layers to what it is calculated
    Layers = Layers[0:l+1]
    gmdh = {'Layers': Layers}
    return gmdh

def ApplyGMDH(gmdh, X):
        
    nLayer = len(gmdh['Layers'])
    Z = X
    for l in range(nLayer):
        Z = GetLayerOutput(gmdh['Layers'][l], Z)
    Yhat =Z 
    return Yhat

def GetLayerOutput(L, X):
    m = X.shape[1]
    N = len(L)
    Z = np.zeros((N,m))
    for k in range(N):
        vars = L[k]['vars']
        x = X[vars,:]
        Z[k,:] =L[k]['f'](x)           
    return Z

# define a function to identify selected variables
def IdentifySelectedVariables(GMDH_Network):
    # this function works with maximum layer number of 3 or less. 
    # if you set the number of layers to higher than 3, you will need to modify this function according to maximum layer number
    Selected_vars = np.empty(8)
    Selected_vars[:] = np.nan
    nL = len(GMDH_Network['Layers'])
    if nL == 3:
        last_vars = np.squeeze(GMDH_Network['Layers'][2]).tolist()['vars']

        mid_vars = np.empty((2,2))
        mid_vars[:] = np.nan
        for j in range(2):
            mid_vars[j,:] = np.squeeze(GMDH_Network['Layers'][1][int(last_vars[j])]).tolist()['vars']
        
        z = 0
        for j in range(2):
            for k in range(2):
                Selected_vars[z] = np.squeeze(GMDH_Network['Layers'][0][int(mid_vars[j,k])]).tolist()['vars'][0]
                Selected_vars[z+1] = np.squeeze(GMDH_Network['Layers'][0][int(mid_vars[j,k])]).tolist()['vars'][1]
                z = z+ 2
    elif nL == 2:
        last_vars = np.squeeze(GMDH_Network['Layers'][1]).tolist()['vars']
        z = 0
        for j in range(2):
            Selected_vars[z] = np.squeeze(GMDH_Network['Layers'][0][int(last_vars[j])]).tolist()['vars'][0]
            Selected_vars[z+1] = np.squeeze(GMDH_Network['Layers'][0][int(last_vars[j])]).tolist()['vars'][1]
            z = z + 2
    elif nL == 1:
            Selected_vars[0] = np.squeeze(GMDH_Network['Layers'][0]).tolist()['vars'][0]
            Selected_vars[1] = np.squeeze(GMDH_Network['Layers'][0]).tolist()['vars'][1]
    return Selected_vars
