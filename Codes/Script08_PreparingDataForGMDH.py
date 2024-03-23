import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings


warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')

# define a function to do averaging for each individual year of each pixel
def Daily2Period_IndividualYears(x, OG_year, PK_year, OD_year):
    # Allocate space for outputs
    winter = np.nan
    earlyspring = np.nan
    spring = np.nan
    summer = np.nan
    latesummer = np.nan

    # check if OG_year, PK_year, or OD_year are not NaN
    if np.logical_or(np.logical_or(np.isnan(OG_year), np.isnan(PK_year)), np.isnan(OD_year)):
        return winter, earlyspring, spring, summer, latesummer
    
    # define the critical values for calculation
    last_day_WI = max(1, round(OG_year - 30)) # the last DoY considered as winter
    last_day_LSP = round(OG_year) # the last day of year considered as early spring
    last_day_SP = round(PK_year) # the last day of year considered as spring
    last_day_SU = max(last_day_SP + 7, round(OD_year - 14)) # the last day of the year considered as summer
    last_day_LSU = round(OD_year) # the last day of year considered as late summer

    # average data for defined periods
    winter =  np.nanmean(x[0:int(last_day_WI)])
    earlyspring =  np.nanmean(x[int(last_day_WI):int(last_day_LSP)])
    spring =  np.nanmean(x[int(last_day_LSP):int(last_day_SP)])
    summer =  np.nanmean(x[int(last_day_SP):int(last_day_SU)])
    latesummer = np.nanmean(x[int(last_day_SU):int(last_day_LSU + 1)])
    return winter, earlyspring, spring, summer, latesummer

# define a function to do averaging for each pixel
def Daily2Period_IndividualPixels(x, date, OG_year, PK_year, OD_year):
    years = np.array(date.year)
    year = np.unique(years)

    # Allocate space for outputs
    winter = np.full(len(year), np.nan)
    earlyspring = np.full(len(year), np.nan)
    spring = np.full(len(year), np.nan)
    summer = np.full(len(year), np.nan)
    latesummer = np.full(len(year), np.nan)
    
    for i in range(len(year)): 
        winter[i], earlyspring[i], spring[i], summer[i], latesummer[i] = Daily2Period_IndividualYears(x[years == year[i]], 
                                                                                        OG_year[i], 
                                                                                        PK_year[i], 
                                                                                        OD_year[i]) 
    return winter, earlyspring, spring, summer, latesummer

# define a function to do averaging for all pixels
def Daily2Period(data, date, OG, OD, PK):
    # Allocate space for outputs
    winter = np.full_like(OG, np.nan)
    earlyspring = np.full_like(OG, np.nan)
    spring = np.full_like(OG, np.nan)
    summer = np.full_like(OG, np.nan)
    latesummer = np.full_like(OG, np.nan)

    for pixel, _ in np.ndenumerate(OG[0,:,:]):
        x = data[:, pixel[0], pixel[1]]
        OG_pixel = OG[:, pixel[0], pixel[1]]
        OD_pixel = OD[:, pixel[0], pixel[1]]
        PK_pixel = PK[:, pixel[0], pixel[1]]
        if len(x[~np.isnan(x)]) == 0:
            pass
        else:
            x1, x2, x3, x4, x5 = Daily2Period_IndividualPixels(x, date, OG_pixel, PK_pixel, OD_pixel)
            winter[:, pixel[0], pixel[1]] = x1
            earlyspring[:, pixel[0], pixel[1]] = x2 
            spring[:, pixel[0], pixel[1]] = x3 
            summer[:, pixel[0], pixel[1]] = x4
            latesummer[:, pixel[0], pixel[1]] = x5
    return {'winter':winter, 'earlyspring':earlyspring, 'spring':spring, 'summer':summer, 'latesummer':latesummer}

# define a function to do PCA analysis between different variables from GLD, GLM, and ERA5
def principal_componant_analysis(x1, x2, *arg):

    if len(arg) == 0:
        x = np.vstack((x1, x2)).transpose()
    else:
        x = np.vstack((x1, x2, arg[0])).transpose()

    n_col = x.shape[1]

    # allocate space to put the PCA 1 data
    out = np.full(x.shape[0], np.nan)
    Explained_Variance = np.nan

    # check if NaN exist in data
    ind = np.array([np.isnan(x[:,i]) for i in range(n_col)])
    ind = np.sum(ind, axis = 0)
    ind = ind != 0

    # check if pixel has no data then return
    if np.sum(ind) == len(ind):
        return out, Explained_Variance

    # screen out the nan data
    x = x[~ind, :]

    # do PCA aanalsys
    pca = PCA(n_components=1)

    # compute first component
    out[~ind] = pca.fit_transform(x).flat

    # puting back the PCA 1 data into the real range of target variable
    out = (out - np.nanmin(out))/(np.nanmax(out) - np.nanmin(out))
    out = out*(max(x.flat) - min(x.flat)) + min(x.flat)

    # stroing the the explained variance by PCA 1 in output file
    Explained_Variance = pca.explained_variance_ratio_

    return out, Explained_Variance

########################################### calculations ######################################
# read OG, OD, and LGS data
LFD = np.load('#Outputs/5_LFD_Results_Averaged.npz', allow_pickle=True)
LFD = dict(zip((LFD.files), (LFD[k] for k in LFD))) # Convert numpy.lib.npyio.NpzFile object to dict

# extract OG, OD, and LGS data from LFD and Keep only pixels falling inside Europe
OG = LFD['OG']
OD = LFD['OD']
LGS = LFD['LGS']
PK = LFD['PeakTime']

lat = LFD['lat']
lon = LFD['lon']

x, y = np.meshgrid(lon, lat)

###################################################
## Do calculations for GLDAS data
print('Reading GLDAS data initiated!')

# read data
GLDAS = np.load('#Outputs/#1 Data/GLDAS_v20_v21-Euro.npz', allow_pickle=True)
# Convert numpy.lib.npyio.NpzFile object to dict
GLDAS = dict(zip((GLDAS.files), (GLDAS[k] for k in GLDAS))) 

date = pd.period_range('1982-01-01', '2022-02-28', freq='D')
ind = date < '2021-01-01'
date = date[ind]
ET_GLD = GLDAS['arr_0'][ind, :, :]
SSM_GLD = GLDAS['arr_1'][ind, :, :]
RSM = GLDAS['arr_2'][ind, :, :]
T = GLDAS['arr_3'][ind, :, :]
P = GLDAS['arr_4'][ind, :, :]
VPD = GLDAS['arr_5'][ind, :, :]

# do periodic averaging for T, P, VPD, and RSM
# ET, and SSM will be fed into PCA analysis then will be averaged
T = Daily2Period(T, date, OG, OD, PK) 
P = Daily2Period(P, date, OG, OD, PK) 
VPD = Daily2Period(VPD, date, OG, OD, PK) 
RSM = Daily2Period(RSM, date, OG, OD, PK) 

# report the status
print('Reading GLDAS data finished!')

###################################################
# Do calculations for GLEAM data
print('Reading GLEAM data initiated!')

# read data
nc = netCDF4.Dataset('#Outputs/#1 Data/GLEAM-Euro.nc')
# define the date corresponding the data
date = pd.period_range('1982-01-01', '2021-12-31', freq='D')

# indexation to exclude the data out of phenological data range
ind = np.logical_and(date >= '1982-01-01', date <= '2020-12-31')
date = date[ind]
ET_GLM = np.array(nc.variables['ET'])[ind, :, :]
PET = np.array(nc.variables['PET'])[ind, :, :]
SSM_GLM = np.array(nc.variables['SSM'])[ind, :, :]

# do periodic averaging for PET
# ET, and SSM will be fed into PCA analysis then will be averaged
PET = Daily2Period(PET, date, OG, OD, PK)

# report the status
print('Reading GLEAM data finished!')

#############################################################
## Do calculations for ERA5 data
print('Reading ERA5 data initiated!')

# read data
nc = netCDF4.Dataset('#Outputs/#1 Data/ERA5Land-Euro.nc')
# define the date corresponding the data
date = pd.period_range('1982-01-01', '2021-12-31', freq='D')

# indexation to exclude the data out of phenological data range
ind = np.logical_and(date >= '1982-01-01', date <= '2020-12-31')
date = date[ind]

ET_ERA = np.array(nc.variables['ET'])[ind, :, :]
SSM_ERA = np.array(nc.variables['SSM'])[ind, :, :]

print('Reading ERA5 data finished!')

#############################################################
## PCA analysis
# initiate a figure with 3 subplots to plot the PCA descriptive results
fig, axes = plt.subplots(3,1)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
axes = axes.flat

# do the PCA over ET and SSM, and Trans data
ET = np.full_like(ET_GLD, np.nan)
SSM = np.full_like(SSM_GLD, np.nan)
Explained_Variance_ET = np.full((ET_GLD.shape[1], ET_GLD.shape[2]), np.nan)
Explained_Variance_SSM = np.full((ET_GLD.shape[1], ET_GLD.shape[2]), np.nan)

for pixel, _ in np.ndenumerate(ET[0,:,:]):
    ET[:, pixel[0], pixel[1]], 
    Explained_Variance_ET[pixel[0], pixel[1]] = principal_componant_analysis(ET_GLD[:, pixel[0], pixel[1]], 
                                                                            ET_GLM[:, pixel[0], pixel[1]], 
                                                                            ET_ERA[:, pixel[0], pixel[1]])
    SSM[:, pixel[0], pixel[1]], 
    Explained_Variance_SSM[pixel[0], pixel[1]] = principal_componant_analysis(SSM_GLD[:, pixel[0], pixel[1]], 
                                                                            SSM_GLM[:, pixel[0], pixel[1]], 
                                                                            SSM_ERA[:, pixel[0], pixel[1]])

# plotting the histogram of the explained variance 
fig, (ax1, ax2) = plt.subplots(2,1)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

# ET
ax1.hist(Explained_Variance_ET, 1000, density=True, stacked=True, facecolor='gray', alpha=0.75)
ax1.set_xlabel('Explained variance')
ax1.set_ylabel('Probability')
ax1.set_ylim([0, 25])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.set_title('ET')

# SSM
ax2.hist(Explained_Variance_SSM, 1000, density=True, stacked=True, facecolor='gray', alpha=0.75)
ax2.set_xlabel('Explained variance')
ax2.set_ylabel('Probability')
ax2.set_ylim([0, 25])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)
ax2.set_title('SSM')

# get current figure
figure = plt.gcf() 
figure.set_size_inches(16, 12)
plt.savefig('#Outputs/Script 8/Explained Variance PC1.jpg', format='jpg', dpi=300)
#plt.show()


# do periodic averaging for ET and SSM
# ET, and SSM will be fed into PCA analysis then will be averaged
ET = Daily2Period(ET, date, OG, OD, PK)
SSM = Daily2Period(SSM, date, OG, OD, PK)

np.savez('#Outputs/8_PixelWiseDataForGMDHAnalysis', 
        ET=ET, SSM=SSM, PET=PET, RSM=RSM, T=T, P= P, VPD=VPD, 
        OG=OG, OD=OD, LGS=LGS, PeakTime=PK, 
        lat=lat, lon=lon, year=np.arange(1982, 2021))
print('data is stored in "#Output"')  
