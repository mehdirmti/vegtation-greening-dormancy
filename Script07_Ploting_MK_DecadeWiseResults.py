import numpy as np
from matplotlib import pyplot as plt

# importing the User defined Function
import UserDefined_Auxiliary_Functions as AuxiliaryFunctions 

# Define a function to do histogram plotting for variables
def DoHistograminPlotting(DATA, Label, txt_x, txt_y, Xlabel, XLIM, OutFileName, YLIM):
 
    # define a figure with 1 subplot
    fig, ax1 = plt.subplots()

    # set some initial settings on the figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})
    fig.set_figheight(4)
    fig.set_figwidth(7)

    # define four different colors to be used for plotting
    colr = ('blue', 'green', 'red', 'm')

    # allocate space to store mean and sigma value of data for different periods
    mu = np.empty((4))
    sigma = np.empty((4))
    skewness = np.empty((4))
    kurtosis = np.empty((4))

    # do histogram plotting for each period
    for k in range(4):
        X = DATA[:,k]
        X = X[~np.isnan(X)]
        mu[k], sigma[k], skewness[k], kurtosis[k] = AuxiliaryFunctions.Histogram_plot(ax1, X, 200, '', colr[k], XLIM, YLIM, Xlabel)
    
    # add legend
    lg = ax1.legend(['$\mu$={} d/y, $\sigma$={}'.format('%.2f'%(mu[0]), '%.2f'%(sigma[0])), 
                    '$\mu$={} d/y, $\sigma$={}'.format('%.2f'%(mu[1]), '%.2f'%(sigma[1])), 
                    '$\mu$={} d/y, $\sigma$={}'.format('%.2f'%(mu[2]), '%.2f'%(sigma[2])),
                    '$\mu$={} d/y, $\sigma$={}'.format('%.2f'%(mu[3]), '%.2f'%(sigma[3]))],
                    loc='upper right')
    
    # set legend box color
    lg.get_frame().set_facecolor('none')
    lg.get_frame().set_edgecolor('none')

    if Label == 'Europe':
        # # define a text of mean and sigma of data to be added to plot
        TxT = ['1982-1990', 
                '1991-2000', 
                '2001-2010',
                '2011-2020']
        
        # add text to plot    
        ax1.text(txt_x, txt_y, TxT[0], color=colr[0])
        ax1.text(txt_x, txt_y - 0.030, TxT[1], color=colr[1])
        ax1.text(txt_x, txt_y - 0.060, TxT[2], color=colr[2])
        ax1.text(txt_x, txt_y - 0.090, TxT[3], color=colr[3])

    ax1.text(-4.5, 0.55, Label)
    # save figure
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.925, wspace=0.3, hspace=None)
    plt.savefig(OutFileName + Label + '.svg', format='svg', dpi=300) # uncoment when high quality figure is needed
    plt.savefig(OutFileName + Label + '.jpg', format='jpg', dpi=300)
    #plt.show()
# End of the function #############################################################

############################################################# Plotting ############################
# load decade wise MK results
data = np.load('#Outputs/7_MannKendall_Results_DecadeWise.npz', allow_pickle=True)

# Convert numpy.lib.npyio.NpzFile object to dict
data = dict(zip((data.files), (data[k] for k in data))) 

# read land cover classification map
LU = np.load('#Outputs/4_PixelsLanduses.npz' , allow_pickle=True)
LU = dict(zip((LU.files), (LU[k] for k in LU)))  # Convert numpy.lib.npyio.NpzFile object to dict
landuse = LU['landuse']
landuse_metadata = LU['info']
LU_type = ['Crp', 'ENF', 'MxF', 'OpS', 'WdT', 'Grs', 'MxT', 'Others']

# extract MK results for different variables from data
OG = data['OG'].tolist()
OD = data['OD'].tolist()
LGS = data['LGS'].tolist()

#############################################################
# Do histogram plotting for OG
OutFileName = r'#Outputs/Script 7/OG-' # change it accordingly
X = OG['slope']
X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
DoHistograminPlotting(X, 'Europe', -1, 0.150, 'Trend Slope [day/year]', [-5, 5], OutFileName, [0, 0.6])

for i in range(8):
    X = OG['slope'][landuse == i,:]
    DoHistograminPlotting(X, LU_type[i], -1, 0.150, 'Trend Slope [day/year]', [-5, 5], OutFileName, [0, 0.6])

# Do histogram plotting for OD
OutFileName = r'#Outputs/Script 7/OD-' # change it accordingly
X = OD['slope']
X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
DoHistograminPlotting(X, 'Europe', -0.5, 0.150, 'Trend Slope [day/year]', [-5, 5], OutFileName, [0, 0.6])

for i in range(8):
    if i == 1:
        YLIM = [0, 0.8]
    elif i == 4:
        YLIM = [0, 1]
    else:
        YLIM = [0, 0.6]
    X = OD['slope'][landuse == i,:]
    DoHistograminPlotting(X, LU_type[i], -1, 0.150, 'Trend Slope [day/year]', [-5, 5], OutFileName, YLIM)

# Do histogram plotting for LGS
OutFileName = r'#Outputs/Script 7/LGS-' # change it accordingly
X = LGS['slope']
X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
DoHistograminPlotting(X, 'Europe', -0.75, 0.125, 'Trend Slope [day/year]', [-5, 5], OutFileName, [0, 0.6])

for i in range(8):
    X = LGS['slope'][landuse == i,:]
    DoHistograminPlotting(X, LU_type[i], -1, 0.150, 'Trend Slope [day/year]', [-5, 5], OutFileName, [0, 0.6])
