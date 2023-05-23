import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd

# importing the User defined Function
import UserDefined_Auxiliary_Functions as AuxiliaryFunctions 

####################################################################################
# --------------------  Calculations ----------------------------------------------#
data = np.load('#Outputs/6_MannKendall_Results_GeneralTrend.npz', allow_pickle=True)
data = dict(zip((data.files), (data[k] for k in data))) # Convert numpy.lib.npyio.NpzFile object to dict

# Read shapefile of Euro bourders
EuroShapeFile = gpd.read_file("..\..\#1 Analysis\shapefiles\The_Euro_Bourders.shp")

# load pixels landuse identification
LU = np.load('#Outputs/4_PixelsLanduses.npz' , allow_pickle=True)
LU = dict(zip((LU.files), (LU[k] for k in LU)))  # Convert numpy.lib.npyio.NpzFile object to dict
landuse = LU['landuse']
landuse_metadata = LU['info']


lat = data['lat']
lon = data['lon']
x, y = np.meshgrid(lon, lat)

OG_MkT = data['OG'].tolist()['trendType']
OD_MkT = data['OD'].tolist()['trendType']
LGS_MkT = data['LGS'].tolist()['trendType']

OG_MkS = data['OG'].tolist()['slope']
OD_MkS = data['OD'].tolist()['slope']
LGS_MkS = data['LGS'].tolist()['slope']

# Read LFD results
data = np.load('#Outputs/5_LFD_Results_Averaged.npz', allow_pickle=True)
data = dict(zip((data.files), (data[k] for k in data)))  # Convert numpy.lib.npyio.NpzFile object to dict

year = data['year']
OG = data['OG']
OD = data['OD']
LGS = data['LGS']

# read north arrow image to be added on plots
north_arrow = plt.imread('#Outputs/north arrow.png')

# define a function to do surface plotting of different variables
def SurfacePlotting(DATA, CLIM, cbar_label, CMAP, OutFileName):   
    fig, ax = plt.subplots()

    # set some initial settings on the figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})
    fig.set_figheight(8)
    fig.set_figwidth(14)


    im, cbar = AuxiliaryFunctions.Surface_Plot(ax, EuroShapeFile, x, y, DATA, CLIM, cbar_label, CMAP, [-40, 80], [30, 90])

    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.95, top=0.925, wspace=0.3, hspace=None)

    newax = fig.add_axes([0.275,0.75,0.15,0.15], anchor='NE', zorder=1)
    newax.imshow(north_arrow)
    newax.axis('off')

    fig.savefig('#Outputs/Script 5/' + OutFileName + '.jpg', dpi=600, transparent=False)   # save the figure to file
    #fig.savefig('E:/ET-SWC paper/#6 Final Check/' + OutFileName + '.jpg', dpi=600, transparent=False)   # save the figure to file

    return im, cbar
## end of function ####################################################################################

# plot spatial varation of long-term mean of LGS, OG, and OD
# LGS
SurfacePlotting(np.nanmean(LGS, axis=0), [120, 220], 'Length of growing season  [days]', 'jet', 'LGS longterm mean')
# OG
SurfacePlotting(np.nanmean(OG, axis=0), [60, 160], 'Onset of greening [DoY]', 'jet', 'OG longterm mean')
# OD
SurfacePlotting(np.nanmean(OD, axis=0), [240, 320], 'Onset of dormancy [DoY]', 'jet', 'OD longterm mean')
plt.show()

# plot spatial varation of trend slope of LGS, OG, and OD
# LGS
SurfacePlotting(LGS_MkS, [-1, 1], "Trending rate [day/year]", 'seismic', 'LGS MKS Surface Plot')
# OG
SurfacePlotting(OG_MkS, [-1, 1], "Trending rate [day/year]", 'seismic', 'OG MKS Surface Plot')
# OD
SurfacePlotting(OD_MkS, [-1, 1], "Trending rate [day/year]", 'seismic', 'OD MKS Surface Plot')
plt.show() 


# define a function to do plotting
def HistogramPlotting(Phenology_data, data_Mk, XLIM, YLIM, XLABEL, xtext):

    # define a figure with 3 subplots
    #plt.figure(figsize=(6,12))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # set some initial settings on the figure
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})
    fig.set_figheight(8)
    fig.set_figwidth(14)

    # define a color vector to be used in plots
    colr = ('blue', 'green', 'red', 'm')
  
    # define periods for whcih you want to plot data seperatly
    period_limits = (1982, 1991, 2001, 2011, 2021)
    periods = ('1982-1990', '1991-2000', '2001-2010', '2011-2020')

    # do plotting for different ranges defined as above seperatly
    for i in range(len(period_limits)-1):

        # average data within each period
        data_M = []
        data_M = np.nanmean(Phenology_data[np.logical_and(year >= period_limits[i], year < period_limits[i+1]), :, :], axis=0)
        
        # extract averaged data for different regions with different trends
        x = data_M[data_Mk == 'i'] # area with increasing trend
        y = data_M[data_Mk == 'd'] # area with decreasing trend
        z = data_M[data_Mk == 'n'] # area with no trend

        # exclude NaN data
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        z = z[~np.isnan(z)]

        # do histograming for each area in on subplot
        # Increasing trend
        mu, sigma, _, _ = AuxiliaryFunctions.Histogram_plot(ax1, 
                                                            x, 
                                                            200, 
                                                            periods[i], 
                                                            colr[i], 
                                                            XLIM, 
                                                            YLIM, 
                                                            XLABEL)
        ax1.text(xtext, YLIM[1] - (i+1)*YLIM[1]/30, '{}: $\mu$ ={} $\pm$ {} [DoY] (n = {}%)'.format(periods[i], 
                                                                                                    '%.0f'%(mu), 
                                                                                                    '%.0f'%(sigma), 
                                                                                                    '%.0f'%(len(x)/(len(x)+len(y)+len(z))*100)), 
                                                                                                    color=colr[i])

        # No trend
        mu, sigma, _, _ = AuxiliaryFunctions.Histogram_plot(ax2, 
                                                            z, 
                                                            200, 
                                                            periods[i], 
                                                            colr[i], 
                                                            XLIM, 
                                                            YLIM, 
                                                            XLABEL)
        ax2.text(xtext, YLIM[1] - (i+1)*YLIM[1]/30, '{}: $\mu$ ={} $\pm$ {} [DoY] (n = {}%)'.format(periods[i], 
                                                                                                    '%.0f'%(mu), 
                                                                                                    '%.0f'%(sigma), 
                                                                                                    '%.0f'%(len(z)/(len(x)+len(y)+len(z))*100)), 
                                                                                                    color=colr[i])

        # decreasing trend
        mu, sigma, _, _ = AuxiliaryFunctions.Histogram_plot(ax3, 
                                                            y, 
                                                            200, 
                                                            periods[i], 
                                                            colr[i], 
                                                            XLIM, 
                                                            YLIM, 
                                                            XLABEL)# Increasing trend
        ax3.text(xtext, YLIM[1] - (i+1)*YLIM[1]/30, '{}: $\mu$ ={} $\pm$ {} [DoY] (n = {}%)'.format(periods[i], 
                                                                                                    '%.0f'%(mu), 
                                                                                                    '%.0f'%(sigma), 
                                                                                                    '%.0f'%(len(y)/(len(x)+len(y)+len(z))*100)), 
                                                                                                    color=colr[i])

        # add title for subplots
        title_font = {'fontname':'Times New Roman', 'size':'13'}

        if i == 0:   
            ax1.set_title('Percentage of pixels with increasing trend', **title_font)
            ax2.set_title('Percentage of pixels with no trend', **title_font)
            ax3.set_title('Percentage of pixels with decreasing trend', **title_font)
    
    
    plt.subplots_adjust(left=0.075, bottom=0.1, right=0.95, top=0.925, wspace=0.3, hspace=None)

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    fig.savefig('#Outputs/Script 5/' + XLABEL + '.jpg', dpi=600, transparent=False)   # save the figure to file
    fig.savefig('E:/ET-SWC paper/#6 Final Check/' + XLABEL + '.jpg', dpi=600, transparent=False)   # save the figure to file

    plt.show()
## end of function ####################################################################################

# Histogram plotting for different periods and areas with defirent trend types (increasing, decreasing, no trend)
# LGS
HistogramPlotting(LGS, LGS_MkT, [100, 250], [0, 0.050], 'Length of growing season [days]', 110)
# OG
HistogramPlotting(OG, OG_MkT, [50, 175], [0, 0.040], 'Onset of greening [DoY]', 60)
# OD
HistogramPlotting(OD, OD_MkT, [240, 320], [0, 0.090], 'Onset of dormancy [DoY]', 250)
