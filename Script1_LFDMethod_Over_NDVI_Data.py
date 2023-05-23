# importing librarries and packages
import numpy as np
import netCDF4
import pandas as pd

# importing the personal module of LFD_NDVI_Method
from LFD_NDVI_Method import LFD_NDVI

# define a function to apply LFD method over different pixels
def LFD_Over_Pixels(y, date):
    # Define an internal function to apply LFD over different years
    def LFD_Over_Years(YR):

        # indexation to seperate data for each specific year
        ind = Year == YR
        x_in = x[ind]
        y_in = y[ind]

        # call LFD method to do calculation for each individual year
        OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation = LFD_NDVI(x_in, y_in, YR, 'off')
        return OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation
    # end of function

    ###### Do the calculations ####################################################################
    # get corresponding day of year for dates
    x = np.array(date.day_of_year)

    # get corresponding years of dates
    Year = np.array(date.year)

    # find unique years during the examined period
    Unique_year = np.unique(np.array(date.year))
    Unique_year = Unique_year.reshape((1, len(Unique_year)))

    # Call internal Function to apply LFD method over different years
    OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation = np.apply_along_axis(LFD_Over_Years, 0, Unique_year)
    return OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation
# end of function

####################### calculations: GIMMS data #############################
nc = netCDF4.Dataset('#Outputs/#1 Data/ndvi-Euro-GIMMS.nc')
ndvi = nc.variables['ndvi']
time = np.array(nc.variables['time'])
lat = np.array(nc.variables['lat'])
lon = np.array(nc.variables['lon'])

# convert biweekly data to monthly
# For each month, GIMSS provides only two data, so we averaged them to be representative for each month.
ndvi_mo = np.empty((int(ndvi.shape[0]/2), ndvi.shape[1], ndvi.shape[2])) 
k = 0
for i in range(0, ndvi.shape[0], 2):
    ndvi_mo[k,:,:] = np.nanmean(ndvi[i:i+2,:,:], axis=0)
    k += 1

# Re-define the date vector in Monthly resolution
date = pd.period_range(start="1982-01-01",end="2015-12-31",freq="M")

# Call LFD method and apply it for axis = 0 (time axis)
OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation = np.apply_along_axis(LFD_Over_Pixels, 0, ndvi_mo, date)

# Store the outputs for further use
np.savez('#Outputs/1_LFD_Results_GIMSS', 
        OG = OG, OD = OD, 
        OG_ndviC = OG_ndviC, 
        OD_ndviC = OD_ndviC, 
        peak_Timing=peak_Timing, 
        PearsonCorrelation = PearsonCorrelation,       
        lat=lat, lon=lon, year=np.unique(date.year))

print("Results are stored in #Outputs/ Folder")

####################### calculations: MODIS data #############################
nc = netCDF4.Dataset('#Outputs/#1 Data/ndvi-Euro-MODIS.nc')
ndvi = nc.variables['ndvi']
lat = np.array(nc.variables['lat'])
lon = np.array(nc.variables['lon'])

# Define the date vector in Monthly resolution
date = pd.period_range(start="2001-01-01",end="2020-12-31",freq="M")

# Call LFD method and apply it for axis = 0 (time axis)
OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation = np.apply_along_axis(LFD_Over_Pixels, 0, ndvi, date)

# Store the outputs for further use
np.savez('#Outputs/2_LFD_Results_MODIS', 
        OG = OG, OD = OD, 
        OG_ndviC = OG_ndviC, 
        OD_ndviC = OD_ndviC, 
        peak_Timing=peak_Timing, 
        PearsonCorrelation = PearsonCorrelation, 
        lat=lat, lon=lon, year=np.unique(date.year))

print("Results are stored in #Outputs/ Folder")

####################### calculations: AVHRR #############################
nc = netCDF4.Dataset('#Outputs/#1 Data/ndvi-euro-AVHRR.nc')
ndvi = nc.variables['ndvi']
lat = np.array(nc.variables['lat'])
lon = np.array(nc.variables['lon'])
time = np.array(nc.variables['time'])

print('Data is loaded succesfully!')

# Define the date vector in daily resolution
date = pd.period_range(start="1982-01-01",end="2017-12-31",freq="d")

# define a function to convert daily data to monthly data
def Day2Month(y, date):
    # define a function to do nanmean calculation
    def NanMean(array_like):
        if any(pd.isnull(array_like)):
            return np.nanmean(np.asarray(array_like))
        else:
            return array_like.mean()
    # create a grouing vector to make averaging over months
    group = np.array([str(i.year) + "-" + str(i.month) for i in date])

    df = pd.DataFrame({'group': group, 'data':y})
    y_m = df.groupby(['group']).apply(NanMean)
    return np.array(y_m.values)

# convert daily data to monthly
ndvi_m = np.apply_along_axis(Day2Month, 0, ndvi, date)
print('Daily data succesfuly converted to monthly resolution')

# Re-define the date vector in Monthly resolution
date = pd.period_range(start="1982-01-01", end="2017-12-31",freq="M")

# Call LFD method and apply it for axis = 0 (time axis)
OG, OD, OG_ndviC, OD_ndviC, peak_Timing, PearsonCorrelation = np.apply_along_axis(LFD_Over_Pixels, 0, ndvi_m, date)

# Store the outputs for further use
np.savez('#Outputs/3_LFD_Results_AVHRR', 
        OG = OG, OD = OD, 
        OG_ndviC = OG_ndviC, 
        OD_ndviC = OD_ndviC, 
        peak_Timing=peak_Timing, 
        PearsonCorrelation = PearsonCorrelation, 
        lat=lat, lon=lon, 
        year=np.unique(date.year))

print("Results are stored in #Outputs/ Folder")