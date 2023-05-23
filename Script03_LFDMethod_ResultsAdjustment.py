#from cProfile import label
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

# importing the User defined Function
import UserDefined_Auxiliary_Functions as AuxiliaryFunctions 


####################################################################################
# define a function to check the relationship between offset and onset data of GIMMS and MODIS/AVHRR
def Adjustment_MOD_AVH_ToGIMMS(Dat1, Dat2, landuse, year_1, year_2, commonPeriod): 
    # extract data for common period of both data series (Dat1 and Dat2)
    x1 = Dat2[np.logical_and(year_2>=commonPeriod[0], year_2<=commonPeriod[1]), :, :]
    y1 = Dat1[np.logical_and(year_1>=commonPeriod[0], year_1<=commonPeriod[1]), :, :]

    # Fit a linear model over data
    def FitType(x, a):
        return a*x

    def DoFittingForIndividualYears(x, y):
        # exclude NaN data from analysis
        ind = ~np.logical_or(np.isnan(x), np.isnan(y))

        # Do fitting
        popt, pcov = optimize.curve_fit(FitType, x[ind], y[ind], p0=0.85, method="lm")
        return popt[0]     

    # keep data pixels having a specific landuse type
    Coefficient_m = np.empty((8,2))
    Coefficient_m[:] = np.nan

    for i in range(8): # we have 8 unique landuse types
        x2 = x1[:, landuse == i]    
        y2 = y1[:, landuse == i]    

        Coefficient = np.array([DoFittingForIndividualYears( x2[j,:], y2[j,:]) for j in range(x2.shape[0])])
        Coefficient_m[i,0] = np.nanmean(Coefficient)
        Coefficient_m[i,1] = np.nanstd(Coefficient)
        # adjust data
        Dat2[:,landuse == i] = Dat2[:,landuse == i] * Coefficient_m[i,0]

    # Return adjusted data along with coefficients
    return Dat2, Coefficient_m
## end of function ####################################################################################

# define a function to average data from differnt products
def AvergatingData(GIMMS, MODIS, AVHRR, years):
    def NaN_array(a, b, c):
        X = np.empty((a, b, c))
        X[:] = np.nan
        return X
    GIMMS = np.vstack((GIMMS, NaN_array((len(years) - GIMMS.shape[0]), GIMMS.shape[1], GIMMS.shape[2])))
    MODIS = np.vstack((NaN_array((len(years) - MODIS.shape[0]), MODIS.shape[1], MODIS.shape[2]), MODIS))
    AVHRR = np.vstack((AVHRR, NaN_array((len(years) - AVHRR.shape[0]), AVHRR.shape[1], AVHRR.shape[2])))
    Data_Mean = np.nanmean(np.array([GIMMS, MODIS, AVHRR]), axis= 0)   
    return Data_Mean
## end of function ####################################################################################

# *********** Calculations ***************************************************************************#
# load pixels landuse identification
LU = np.load('#Outputs/4_PixelsLanduses.npz' , allow_pickle=True)
# Convert numpy.lib.npyio.NpzFile object to dict
LU = dict(zip((LU.files), (LU[k] for k in LU))) 
landuse = LU['landuse']
landuse_metadata = LU['info']

# load LFD results
GIM = np.load('#Outputs/1_LFD_Results_GIMSS.npz' , allow_pickle=True)
MOD = np.load('#Outputs/2_LFD_Results_MODIS.npz' , allow_pickle=True)
AVH = np.load('#Outputs/3_LFD_Results_AVHRR.npz' , allow_pickle=True)

# Convert numpy.lib.npyio.NpzFile object to dict
GIM = dict(zip((GIM.files), (GIM[k] for k in GIM))) 
MOD = dict(zip((MOD.files), (MOD[k] for k in MOD))) 
AVH = dict(zip((AVH.files), (AVH[k] for k in AVH))) 

# # extract lat and lon vectors
lat = GIM['lat']
lon = GIM['lon']

# creat years arrays for different data
GIM_Y = np.arange(1982,2016,1)
MOD_Y = np.arange(2001,2021,1)
AVH_Y = np.arange(1982, 2018,1)

year = np.arange(1982, 2021,1) # complete period

# compare GIMMS OG with those of MODIS and AVHRR and then modify MOD and VAH
OG_GIM = GIM['OG']
OG_MOD, OG_Coefficient_MOD = Adjustment_MOD_AVH_ToGIMMS(GIM['OG'], MOD['OG'], landuse, GIM_Y, MOD_Y, (2001, 2015))
OG_AVH, OG_Coefficient_AVH = Adjustment_MOD_AVH_ToGIMMS(GIM['OG'], AVH['OG'], landuse, GIM_Y, AVH_Y, (1982, 2015))

# compare GIMMS OD with those of MODIS and AVHRR and then modify MOD and VAH 
OD_GIM = GIM['OD']
OD_MOD, OD_Coefficient_MOD = Adjustment_MOD_AVH_ToGIMMS(GIM['OD'], MOD['OD'], landuse, GIM_Y, MOD_Y, (2001, 2015))
OD_AVH, OD_Coefficient_AVH = Adjustment_MOD_AVH_ToGIMMS(GIM['OD'], AVH['OD'], landuse, GIM_Y, AVH_Y, (1982, 2015))

# compare GIMMS Peaking time with those of MODIS and AVHRR and then modify MOD and VAH 
PT_GIM = GIM['peak_Timing']
PT_MOD, PT_Coefficient_MOD = Adjustment_MOD_AVH_ToGIMMS(GIM['peak_Timing'], MOD['peak_Timing'], landuse, GIM_Y, MOD_Y, (2001, 2015))
PT_AVH, PT_Coefficient_AVH = Adjustment_MOD_AVH_ToGIMMS(GIM['peak_Timing'], AVH['peak_Timing'], landuse, GIM_Y, AVH_Y, (1982, 2015))

# OGG = np.concatenate((OG_Coefficient_MOD, OG_Coefficient_AVH, OD_Coefficient_MOD, OD_Coefficient_AVH), axis=1)
# ODD = np.concatenate(OD_Coefficient_MOD, OD_Coefficient_AVH)

Coefficient =pd.DataFrame(np.concatenate((OG_Coefficient_MOD, 
                                          OG_Coefficient_AVH, 
                                          OD_Coefficient_MOD, 
                                          OD_Coefficient_AVH), axis=1),
                          columns=['OG-MOD-mean',
                                   'OG-MOD-std',
                                   'OG-AVH-mean',
                                   'OG-AVH-std',
                                   'OD-MOD-mean',
                                   'OD-MOD-std',
                                   'OD-AVH-mean',
                                   'OD-AVH-std'],
                          index=landuse_metadata[:,0])
Coefficient.to_excel(r"E:\ET-SWC paper\#1 Analysis\# Final Analysis\#Outputs\Script 3\Scallingfactors.xlsx")
# take Europe mean for each year
OGG_mean = np.nanmean(np.nanmean(OG_GIM, axis=1), axis=1)
OGM_mean = np.nanmean(np.nanmean(OG_MOD, axis=1), axis=1)
OGA_mean = np.nanmean(np.nanmean(OG_AVH, axis=1), axis=1)

# take Europe mean for each year
ODG_mean = np.nanmean(np.nanmean(OD_GIM, axis=1), axis=1)
ODM_mean = np.nanmean(np.nanmean(OD_MOD, axis=1), axis=1)
ODA_mean = np.nanmean(np.nanmean(OD_AVH, axis=1), axis=1)

x = np.hstack((GIM_Y, MOD_Y, AVH_Y))
y_OG = np.hstack((OGG_mean, OGM_mean, OGA_mean))
y_OD = np.hstack((ODG_mean, ODM_mean, ODA_mean))

# y_OG = AuxiliaryFunctions.remove_outlier(y_OG) 
# y_OD = AuxiliaryFunctions.remove_outlier(y_OD) 

P_OG = np.polyfit(x[~np.isnan(y_OG)], y_OG[~np.isnan(y_OG)], 1)
P_OD = np.polyfit(x[~np.isnan(y_OD)], y_OD[~np.isnan(y_OD)], 1)

# plot mean values for different years
fig, (ax1, ax2) = plt.subplots(1,2)
# set some initial settings on the figure
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.family':'Times New Roman', 'font.size':'14'})
fig.set_figheight(8)
fig.set_figwidth(14)

ax1.scatter(GIM_Y, OGG_mean, label= 'GIMMS')
ax1.scatter(MOD_Y, OGM_mean, label = 'MODIS')
ax1.scatter(AVH_Y, OGA_mean, label = 'AVHRR')
ax1.plot(year, np.polyval(P_OG, year), label = 'best fit')
ax1.set_ylim([90, 115])
ax1.set_xlim([1980, 2021])
ax1.legend(loc= 'upper right')
ax1.set_xlabel('years',**{'fontname':'Times New Roman', 'size':'14'})
ax1.set_ylabel('Onset of greening [DoY]',**{'fontname':'Times New Roman', 'size':'14'})
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.text(1976, 113, 'a)',**{'fontname':'Times New Roman', 'size':'14'})
# Set the tick labels font
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(14)
ax2.scatter(GIM_Y, ODG_mean, label= 'GIMMS')
ax2.scatter(MOD_Y, ODM_mean, label = 'MODIS')
ax2.scatter(AVH_Y, ODA_mean, label = 'AVHRR')
ax2.plot(year, np.polyval(P_OD, year), label = 'best fit')
ax2.set_ylim([275, 295])
ax2.set_xlim([1980, 2021])
ax2.set_yticks([275, 280, 285, 290, 295])
ax2.legend(loc= 'upper right',)
ax2.set_xlabel('years',**{'fontname':'Times New Roman', 'size':'14'})
ax2.set_ylabel('Onset of Dormancy [DoY]',**{'fontname':'Times New Roman', 'size':'14'})
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)
ax2.text(1976, 293.6, 'b)',**{'fontname':'Times New Roman', 'size':'14'})
# Set the tick labels font
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(14)

fig.savefig('#Outputs/Script 3/OG and OD over Europe.jpg', dpi=600, transparent=False)   # save the figure to file
#fig.savefig('E:/ET-SWC paper/#6 Final Check/OG and OD over Europe.jpg', dpi=600, transparent=False)   # save the figure to file
plt.show()



# averaging data for further analysis
OG = AvergatingData(OG_GIM, OG_MOD, OG_AVH, year)
OD = AvergatingData(OD_GIM, OD_MOD, OD_AVH, year)
PT = AvergatingData(PT_GIM, PT_MOD, PT_AVH, year)

np.savez('#Outputs/5_LFD_Results_Averaged', 
        OG=OG, OD=OD, LGS=OD-OG, PeakTime=PT, 
        lat=lat, lon=lon, year=year)
