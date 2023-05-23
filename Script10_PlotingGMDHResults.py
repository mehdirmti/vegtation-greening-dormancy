import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Proj, transform, CRS
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature


# importing the User defined Function
import UserDefined_Auxiliary_Functions as AuxiliaryFunctions 

# Read data
data = np.load('#Outputs/9_GMDH_Results.npz' , allow_pickle=True)

# Convert numpy.lib.npyio.NpzFile object to dict
data = dict(zip((data.files), (data[k] for k in data))) 

# extracting results for OG, OD, and LGS
OG = data['OG'].tolist()
OD = data['OD'].tolist()
LGS = data['LGS'].tolist()

lat = data['lat']
lon = data['lon']

SHP = gpd.read_file("..\..\#1 Analysis\shapefiles\The_Euro_Bourders.shp")
#SHP = gpd.read_file("..\..\#1 Analysis\shapefiles\Euro_Landuse.shp")

x, y = np.meshgrid(lon, lat)

# read north arrow image to be added on plots
north_arrow = plt.imread('#Outputs/north arrow.png')

# define a function to do the job
def PlotResults(DominantVariables, VariablesName):

    # put variables names into shape that can be interpreted as Latex (to have subscripts and superscripts in text on plots)
    Varlabel = pd.DataFrame({'Txt': VariablesName})
    Varlabel = np.array(Varlabel['Txt'].str.split('_').tolist())
    Varlabel = ['$' + Varlabel[i,0] + '_{' + Varlabel[i,1] + '}$' for i in range(Varlabel.shape[0])]
    Varlabel = np.array(Varlabel)

    # Allocate a new space to put frequency of variables among pixels
    Frequaency = np.full((len(VariablesName)+1, len(VariablesName)+1), np.nan)

    # determine number pixels where GMDH has a result
    n = len(DominantVariables[~np.isnan(DominantVariables)])/2
    for i in range(len(VariablesName)):
        for j in range(len(VariablesName)):
            Frequaency[i,j] = (sum(np.logical_and(DominantVariables[0,:,:].flatten() == i, 
                                                    DominantVariables[1,:,:].flatten() == j))/n)*100
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
    plt.show()

# PlotResults(LGS['DominantVariables'], LGS['Variables_name'])
# PlotResults(OG['DominantVariables'], OG['Variables_name'])
# PlotResults(OD['DominantVariables'], OD['Variables_name'])

# surface plot of accuracy of fititing
def FittingAccuracySpatialDistrbution(data, OutFileName):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # set some initial settings on the figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})
    fig.set_figheight(8)
    fig.set_figwidth(14)

    # Training set
    im, cbar = AuxiliaryFunctions.Surface_Plot(ax1, SHP, x, y, data[0,:,:], [0,1], 'R [-]', 'jet', [-40, 80], [30, 90])
    ax1.set_title('Training set')

    # remove colorbar
    cbar.remove()

    # Evaluation set
    im, cbar = AuxiliaryFunctions.Surface_Plot(ax2, SHP, x, y, data[1,:,:], [0,1], 'R [-]', 'jet', [-40, 80], [30, 90])
    ax2.set_title('Evaluation set')

    # remove colorbar
    cbar.remove()

    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=None)

    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.87, 0.2, 0.01, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("R [-]", rotation=270)

    newax = fig.add_axes([0.05,0.68,0.1,0.1], anchor='NE', zorder=1)
    newax.imshow(north_arrow)
    newax.axis('off')

    newax = fig.add_axes([0.483,0.68,0.1,0.1], anchor='NE', zorder=1)
    newax.imshow(north_arrow)
    newax.axis('off')

    fig.savefig('#Outputs/Script 10/' + OutFileName + '.jpg', dpi=600, transparent=False)   # save the figure to file


# Spatial distribution of Fitting accuracy
FittingAccuracySpatialDistrbution(LGS['FitAccuracy'], 'LGS Fit accuracy')
FittingAccuracySpatialDistrbution(OG['FitAccuracy'], 'OG Fit accuracy')
FittingAccuracySpatialDistrbution(OD['FitAccuracy'], 'OD Fit accuracy')
plt.show()


# %% Spatial distribution of dominant variables
# Define a function to do surface ploting for geospatial data 
def SpatialDistrbution(ax, euro_shp, longitude, latitude, data, colorbar_limits, colorbar_label, xlim, ylim):
    # ax: the plot's axis
    # bourder_shapefile: geopandas shapfile of bourders of region you want to do plotting for
    # longitude and latitude (both in n*m shape): the meshgrid of lat and lon vectors attached to data
    # data (n*m shape): the data to be plotted
    # colorbar_limits:  the upper and lower limits of colorbar
    # colorbar_label: the label to be attached to colorbar
    # xlim and ylim: the upper and lower limits of x and y axes

    # get discrete colormap
    cmap = plt.get_cmap('jet', colorbar_limits[1] - colorbar_limits[0])

    euro_shp.set_crs(epsg=4326, inplace=True)

    im = ax.pcolor(longitude, latitude, data, 
                   edgecolors='none', linewidths=2, 
                   cmap= cmap, 
                   vmin=colorbar_limits[0] - 0.5,
                   vmax=colorbar_limits[1] + 0.5)

  
    im.set_clim(colorbar_limits)
    ax.set_xlabel('Longitude [DD]', **{'fontname':'Times New Roman', 'size':'13'})
    ax.set_ylabel('Latitude [DD]', **{'fontname':'Times New Roman', 'size':'13'})
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(visible=True, which='major', axis='both')

    euro_shp.geometry.boundary.plot(color='k', edgecolor='k', linewidth = 1, ax=ax) #Use your second dataframe   
    cbar = plt.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(colorbar_label, rotation = 270)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im, cbar

# surface plot of dominant variables
def SpatialDistDominVar(data, VarNames, OutFileName):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # set some initial settings on the figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})
    fig.set_figheight(8)
    fig.set_figwidth(14)

    k = 0
    for i in range(0, int(len(VarNames)), int(len(VarNames)/5)):
        for j in range(i, int(i+len(VarNames)/5)):
            data[data==j] = k
        k += 1

    Varlabel = ['SSM', 'RSM', 'T', 'P', 'VPD']
    CLIM = [0, 5]

    # First dominant variable
    im, cbar = SpatialDistrbution(ax1, SHP, x, y, data[0,:,:], CLIM, 'Control factors', [-30, 80], [30, 90])
    ax1.set_title('First dominant control factor', **{'fontname':'Times New Roman', 'size':'13'})

    # remove colorbar
    cbar.remove()

    # Second dominant variable
    im, cbar = SpatialDistrbution(ax2, SHP, x, y, data[1,:,:], CLIM, 'Control factors', [-30, 80], [30, 90])
    ax2.set_title('Second dominant control factor', **{'fontname':'Times New Roman', 'size':'13'})

    # remove colorbar
    cbar.remove()

    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.95, top=0.925, wspace=0.3, hspace=None)

    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(CLIM[0]+0.5, CLIM[1]))
    cbar.ax.set_yticklabels(Varlabel)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("Control factors", rotation=270, **{'fontname':'Times New Roman', 'size':'13'})

    newax = fig.add_axes([0.05,0.7,0.1,0.1], anchor='NE', zorder=1)
    newax.imshow(north_arrow)
    newax.axis('off')

    newax = fig.add_axes([0.483,0.7,0.1,0.1], anchor='NE', zorder=1)
    newax.imshow(north_arrow)
    newax.axis('off')

    fig.savefig('#Outputs/Script 10/' + OutFileName + '.jpg', dpi=600, transparent=False)   # save the figure to file

    plt.show()

#SpatialDistDominVar(LGS['DominantVariables'], LGS['Variables_name'], 'LGS Spatial Distributio'n)
SpatialDistDominVar(OG['DominantVariables'], OG['Variables_name'], 'OG Spatial Distribution')
SpatialDistDominVar(OD['DominantVariables'], OD['Variables_name'], 'OD Spatial Distribution')
