#from http.client import _DataType
import numpy as np
import warnings

warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')

# importing the personal module of GreenCalc
from GroupMethodDataHandling import *

# define a function to prepare the input variables for GMDH analysis
def HydroMeteoDataAssignment(X, DATA, pixel_id, n_range, VAR):
    if VAR == 'OG':
        X[n_range[0],:] = DATA['winter'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['winter'][:, pixel_id[0], pixel_id[1]])
        X[n_range[1],:] = DATA['earlyspring'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['earlyspring'][:, pixel_id[0], pixel_id[1]])

    if VAR == 'OD':
        X[n_range[0],:] = DATA['winter'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['winter'][:, pixel_id[0], pixel_id[1]])
        X[n_range[1],:] = DATA['spring'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['spring'][:, pixel_id[0], pixel_id[1]])
        X[n_range[2],:] = DATA['summer'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['summer'][:, pixel_id[0], pixel_id[1]])
        X[n_range[3],:] = DATA['latesummer'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['latesummer'][:, pixel_id[0], pixel_id[1]])
    
    if VAR == 'LGS':
        X[n_range[0],:] = DATA['winter'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['winter'][:, pixel_id[0], pixel_id[1]])
        X[n_range[1],:] = DATA['earlyspring'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['earlyspring'][:, pixel_id[0], pixel_id[1]])
        X[n_range[2],:] = DATA['spring'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['spring'][:, pixel_id[0], pixel_id[1]])
        X[n_range[3],:] = DATA['summer'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['summer'][:, pixel_id[0], pixel_id[1]])
        X[n_range[4],:] = DATA['latesummer'][:, pixel_id[0], pixel_id[1]] - np.nanmean(DATA['latesummer'][:, pixel_id[0], pixel_id[1]])
    return X
# end of function


# define a function to do GMDH analysis
def DoGMDHAnalysis(TARGET, VAR):

    # allocate space for outputs
    DominantVariables = np.full((2, TARGET.shape[1], TARGET.shape[2]), np.nan)
    DominantVariables_importance = np.full_like(DominantVariables, np.nan)
    FitAccuracy = np.full_like(DominantVariables, np.nan)

    # do GMDH analysis pixel-wisely
    for pixel, _ in np.ndenumerate(TARGET[0,:,:]):

        # get target data for pixel take the anomaly and put it in y
        y = TARGET[:, pixel[0], pixel[1]] - np.nanmean(TARGET[:, pixel[0], pixel[1]])
 
        # determine the number of variables used for prediction
        if VAR == 'OG':
            n_variables = 10
        elif VAR == 'OD':
            n_variables = 20
        elif VAR == 'LGS':
            n_variables = 25
        
        # allocate a spcae to put the inputs
        x = np.empty((n_variables, 39))
        x[:] = np.nan

        # store variables name for future use
        Variables_name = [''] * (n_variables-1)

        if VAR == 'OG':
            Variables_name[0:n_variables] = ['SSM_w', 'SSM_es', 
                                'RSM_w', 'RSM_es', 
                                'T_w', 'T_es',  
                                'P_w', 'P_es',  
                                'VPD_w', 'VPD_es']
            m = 2
        elif VAR == 'OD':
            Variables_name[0:n_variables] = ['SSM_w', 'SSM_sp', 'SSM_su', 'SSM_ls', 
                                'RSM_w', 'RSM_sp', 'RSM_su', 'RSM_ls',
                                'T_w', 'T_sp', 'T_su', 'T_ls',
                                'P_w', 'P_sp', 'P_su', 'P_ls', 
                                'VPD_w', 'VPD_sp', 'VPD_su', 'VPD_ls']            
            m = 4
        elif VAR == 'LGS':
            Variables_name[0:n_variables] = ['SSM_w', 'SSM_es', 'SSM_sp', 'SSM_su', 'SSM_ls', 
                    'RSM_w', 'RSM_es', 'RSM_sp', 'RSM_su', 'RSM_ls',
                    'T_w', 'T_es', 'T_sp', 'T_su', 'T_ls',
                    'P_w', 'P_es', 'P_sp', 'P_su', 'P_ls',
                    'VPD_w', 'VPD_es', 'VPD_sp', 'VPD_su', 'VPD_ls'] 
            m = 5


        # check if pixel has not data then jomp to next pixel and skip the folowing commands
        if len(y[~np.isnan(y)]) == 0:
            continue

        # Hydro-meteorological data assignment
        x = HydroMeteoDataAssignment(x, SSM, pixel, np.arange(0*m,1*m), VAR)
        x = HydroMeteoDataAssignment(x, RSM, pixel, np.arange(1*m,2*m), VAR)
        x = HydroMeteoDataAssignment(x, T, pixel, np.arange(2*m,3*m), VAR)
        x = HydroMeteoDataAssignment(x, P, pixel, np.arange(3*m,4*m), VAR)
        x = HydroMeteoDataAssignment(x, VPD , pixel, np.arange(4*m,5*m), VAR)
   
       # define GMDH networks parameters
        params = {'MaxLayerNeurons': 5, 'MaxLayers': 1, 'alpha':0.85, 'pTrain':0.7}

        # define target and inputs
        # Inputs: m*n array
        # Targets: 1*n array
        # each row in X reperesents a variable (m variable)
        # each column in X and Y represents a sampling (n data point)
        Inputs = x
        Targets = y.reshape(1, y.shape[0])

        # checking for nan data column wisely
        ind = ~np.logical_or(np.sum(np.isnan(Inputs), axis=0) != 0, np.isnan(Targets)).flatten()
        
        # Screen out the nan data
        Inputs = Inputs[:, ind]
        Targets = Targets[:, ind]

        # check if you data after removing NaNs, otherwise jump to next pixel
        if Targets.shape[1] == 0:
            continue

        n_repeat = 100 # the number of repeatation

        # allocate space for output
        gmdh ={}
        Rt = np.full(n_repeat, np.nan)
        Re = np.full(n_repeat, np.nan)

        # Repeat GMDH analysis for each set for n_repeat times
        for i in range(n_repeat):
            
            # determine number of data
            nData = Inputs.shape[1]

            # indexation for permutation
            Perm = np.random.permutation(nData)
            
            # devide data into train and test subsets
            # Train Data
            pTrain = 0.6
            nTrainData = round(pTrain*nData)
            TrainInd = Perm[0:nTrainData]
            TrainInputs = Inputs[:,TrainInd]
            TrainTargets = Targets[:,TrainInd]
            
            # Test Data
            TestInd = Perm[nTrainData:nData]
            TestInputs = Inputs[:,TestInd]
            TestTargets = Targets[:,TestInd]

            ## develop and Train GMDH Network
            gmdh.update({str(i): GMDH(params, TrainInputs, TrainTargets)})

            #  GMDH Network Evaluation
            TrainOutputs = ApplyGMDH(gmdh[str(i)], TrainInputs) 
            TestOutputs = ApplyGMDH(gmdh[str(i)], TestInputs)

            Rt[i] = np.corrcoef(TrainOutputs,TrainTargets)[0,1]
            Re[i] = np.corrcoef(TestOutputs,TestTargets)[0,1]

        ind = Re >= 0.7 # screening out the networks with Re < 0.7
        Re = Re[ind]
        Rt = Rt[ind]

        # check if all models are screened out, then jump to next pixel
        if len(Re) == 0:
            continue

        FitAccuracy[0, pixel[0], pixel[1]] = np.nanmean(Rt)
        FitAccuracy[1, pixel[0], pixel[1]] = np.nanmean(Re)

        # dominant variable selection
        Selected_vars = np.full((n_repeat, 2), np.nan)
        for i in range(n_repeat):
            if len(gmdh[str(i)]) != 0:
                Selected_vars[i,:] = np.squeeze(gmdh[str(i)]['Layers'][0]).tolist()['vars']

        Selected_vars = Selected_vars[ind,:]

        unique_vars, _, _, unique_counts = np.unique(Selected_vars[~np.isnan(Selected_vars)], return_index = True, return_inverse = True, return_counts = True, equal_nan = True)

        ind = np.flip(np.argsort(unique_counts))
        unique_counts = unique_counts[ind]
        unique_vars = unique_vars[ind]

        unique_vars_importance = (unique_counts/Selected_vars.shape[0])*100

        DominantVariables[:, pixel[0], pixel[1]] = unique_vars[:2]
        DominantVariables_importance[:, pixel[0], pixel[1]] = unique_vars_importance[:2]

    out = {'DominantVariables':DominantVariables, 
            'DominantVariables_importance':DominantVariables_importance,
            'FitAccuracy': FitAccuracy,
            'Variables_name': Variables_name}
    return out
# end of function

#################################################################################################################
# Calculations
#################################################################################################################
# load data
data = np.load('#Outputs/8_PixelWiseDataForGMDHAnalysis.npz' , allow_pickle=True)

# Convert numpy.lib.npyio.NpzFile object to dict
data = dict(zip((data.files), (data[k] for k in data)))

lat = data['lat']
lon = data['lon']

# extract data for different variables
SSM = data['SSM'].tolist()
RSM = data['RSM'].tolist()
T = data['T'].tolist()
P = data['P'].tolist()
VPD = data['VPD'].tolist()

# DO the analysis for Onset data
print('Analysis started for: OG')
OG = DoGMDHAnalysis(data['OG'], 'OG')

print('Previous analysis finished and new analysis satrted for: OD')
# DO the analysis for Offset data
OD = DoGMDHAnalysis(data['OD'], 'OD')

print('Previous analysis finished and new analysis satrted for: LGS')
# DO the analysis for SeasonLength data
LGS = DoGMDHAnalysis(data['LGS'], 'LGS')

print('Analysis finished. Storing results...')
# Store Results
np.savez('#Outputs/9_GMDH_Results', OG = OG, OD = OD, LGS = LGS, lat = lat, lon = lon)

print('Results stored. Run another analysis')