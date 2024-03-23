import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
from matplotlib.colors import LogNorm


# Read data
data = np.load('#Outputs/9_GMDH_Results.npz' , allow_pickle=True)

# Convert numpy.lib.npyio.NpzFile object to dict
data = dict(zip((data.files), (data[k] for k in data))) 

# extracting results for OG, OD, and LGS
OG = data['OG'].tolist()
OD = data['OD'].tolist()
LGS = data['LGS'].tolist()

# read land cover classification map
LU = np.load('#Outputs/4_PixelsLanduses.npz' , allow_pickle=True)
LU = dict(zip((LU.files), (LU[k] for k in LU)))  # Convert numpy.lib.npyio.NpzFile object to dict
landuse = LU['landuse']
landuse_metadata = LU['info']
LU_type = ['Crp', 'ENF', 'MxF', 'OpS', 'WdT', 'Grs', 'MxT', 'Others']

# define a function to do the job
def PlotResults(DominantVariables, VariablesName, LU, landCover, VAR):

    # exclude pixels with land cover other than given one
    DominantVariable_1 = DominantVariables[0, landuse == LU]
    DominantVariable_2 = DominantVariables[1, landuse == LU]

    # put variables names into shape that can be interpreted as Latex (to have subscripts and superscripts in text on plots)
    Varlabel = pd.DataFrame({'Txt': VariablesName})
    Varlabel = np.array(Varlabel['Txt'].str.split('_').tolist())
    Varlabel = ['$' + Varlabel[i,0] + '_{' + Varlabel[i,1] + '}$' for i in range(Varlabel.shape[0])]
    Varlabel = np.array(Varlabel)

    # Allocate a new space to put frequency of variables among pixels
    Frequaency = np.full((len(VariablesName)+1, len(VariablesName)+1), np.nan)

    # determine number pixels where GMDH has a result
    n = len(DominantVariable_1[~np.isnan(DominantVariable_1)])
    for i in range(len(VariablesName)):
        for j in range(len(VariablesName)):
            Frequaency[i,j] = (sum(np.logical_and(DominantVariable_1 == i, 
                                                    DominantVariable_2 == j))/n)*100
    row_sum_f = np.nansum(Frequaency, axis=1)
    col_sum_f = np.nansum(Frequaency, axis=0)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 12)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12})
    im = sb.heatmap(Frequaency, cmap='plasma', cbar_kws={'label': 'Frequency [%]'}, cbar=True, norm=LogNorm(vmin=0.01, vmax=10))
    ax.set_yticklabels(np.insert(Varlabel, len(Varlabel), '$\sum$'), rotation = 0)
    ax.set_xticklabels(np.insert(Varlabel, len(Varlabel), '$\sum$'), rotation = -15)
    ax.set_ylabel('First important factor')
    ax.set_xlabel('Second important factor')
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    for i in range(len(VariablesName)):
        ax.text(i + 0.1, len(VariablesName) + 0.75, '{}'.format('%.1f'%(col_sum_f[i])))
        ax.text(len(VariablesName) + 0.2, i + 0.75, '{}'.format('%.1f'%(row_sum_f[i])))

    plt.savefig('#Outputs/Script 12/' + VAR + '-'+ landCover + '.jpg', format='jpg', dpi=300)
    # plt.show()

for i in range(7):
    PlotResults(LGS['DominantVariables'], LGS['Variables_name'], i, LU_type[i], 'LGS')
    PlotResults(OG['DominantVariables'], OG['Variables_name'], i, LU_type[i], 'OG')
    PlotResults(OD['DominantVariables'], OD['Variables_name'], i, LU_type[i], 'OD')  
