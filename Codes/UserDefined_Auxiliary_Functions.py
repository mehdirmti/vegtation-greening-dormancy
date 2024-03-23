"""
Created on Wednsday Sep 28 08:00:00 2022
@author: Mehdi Rahmati
E-mail: mehdirmti@gmail.com, m.rahmati@fz-juelich.de

Description:
This module provides some functions for calculations applied in our paper.

More detailed information can be found in the article published at the following link:
https://www.XXX.XXX


To be cited as: 
Rahmati et al., 2023.The continuous increase in evaporative demand shortened the growing season of European ecosystems 
in the last decade. Comunications Earth & Environement, XX, XX-XX. 

"""

import numpy as np
from geopandas.tools import sjoin
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as sst
import numpy.ma as ma
from sklearn.metrics import mean_squared_error
from scipy import interpolate

# define a function to get the first index key in a dictionery
def get_first_key(dictionary):
    for key in dictionary:
        return key
    raise IndexError

# Define a function to check accuracy of interpolations in DataResolutionConvertor Fucntion
def InterpolationAccuracyExaminer(original_data, interpolated_data, gepspatial_limits):
    
    if gepspatial_limits['lon_strat'] > gepspatial_limits['lon_end']:
        lon_step = -2.5
    else:
        lon_step = 2.5

    if gepspatial_limits['lat_start'] > gepspatial_limits['lat_end']: 
        lat_step = -2.5
    else:
        lat_step = 2.5

    in_mean = []
    out_mean = []
    for longitude in np.arange(gepspatial_limits['lon_strat'], gepspatial_limits['lon_end'], lon_step):
        for latitude in np.arange(gepspatial_limits['lat_start'], gepspatial_limits['lat_end'], lat_step):
            if lat_step > 0:
                X = original_data['data'][np.logical_and(original_data['lat']>= latitude, original_data['lat'] < latitude+lat_step),:]
                Y = interpolated_data['data'][np.logical_and(interpolated_data['lat']>= latitude, interpolated_data['lat'] < latitude+lat_step),:]
            else:
                X = original_data['data'][np.logical_and(original_data['lat']<= latitude, original_data['lat'] > latitude+lat_step),:]
                Y = interpolated_data['data'][np.logical_and(interpolated_data['lat'] <= latitude, interpolated_data['lat'] > latitude+lat_step),:]

            if lon_step > 0:
                X = X[:, np.logical_and(original_data['lon']>= longitude, original_data['lon'] < longitude+lon_step)]
                Y = Y[:, np.logical_and(interpolated_data['lon']>= longitude, interpolated_data['lon'] < longitude+lon_step)]
            else:
                X = X[:, np.logical_and(original_data['lon'] <= longitude, original_data['lon'] > longitude+lon_step)]
                Y = Y[:, np.logical_and(interpolated_data['lon']<= longitude, interpolated_data['lon'] > longitude+lon_step)]                    

            in_mean.append(np.nanmean(X.flatten()))
            out_mean.append(np.nanmean(Y.flatten()))
    in_mean = np.array(in_mean)
    out_mean = np.array(out_mean)
    id = np.logical_and(~np.isnan(in_mean), ~np.isnan(out_mean))

    R = np.corrcoef(in_mean[id], out_mean[id])[0,1]
    RMSE = np.sqrt(mean_squared_error(in_mean[id], out_mean[id]))

    SSE = np.sum((in_mean[id]-out_mean[id])**2)
    SSE_ob = np.sum((in_mean[id]-np.mean(in_mean[id]))**2)
    E = 1 - SSE/SSE_ob

    STD_original = np.nanstd(original_data['data'].flatten())
    STD_interpolated = np.nanstd(interpolated_data['data'].flatten())
    STD_change = (STD_interpolated - STD_original)/STD_original*100

    return {'R': R, 'E': E, 'RMSE' : RMSE, 'STD_orig': STD_original, 'STD_interp': STD_interpolated, 'STD_change': STD_change}

# define a function to convert the resolution of data
def DataResolutionConvertor(data_in, LAT, LON):

    # define new lat and lon vectors that you want to interpolate for these points
    lat_1 = np.round(LAT[0])
    lat_end = np.round(LAT[len(LAT)-1])
    if lat_1 > lat_end:
        lat_step = -0.25
    elif lat_1 < lat_end:
        lat_step = 0.25

    lat = np.arange(lat_1 + lat_step/2, lat_end, lat_step) 

    lon_1 = np.round(LON[0])
    lon_end = np.round(LON[len(LON)-1])
    if lon_1 > lon_end:
        lon_step = -0.25
    elif lon_1 < lon_end:
        lon_step = 0.25

    lon = np.arange(lon_1 + lon_step/2, lon_end, lon_step) 

    # meshgrid
    grid_x, grid_y = np.meshgrid(lon, lat)

    if len(data_in.shape) == 2:
        # fit a function to do inerpolation 
        interp = interpolate.RegularGridInterpolator((LAT, LON), data_in)
        
        # do interpolation
        data_out = interp((grid_y, grid_x))

        # Evaluate accuracy of interpolations
        Accuracy = InterpolationAccuracyExaminer(original_data = {'data': data_in, 'lat': LAT, 'lon': LON}, 
                                          interpolated_data = {'data': data_out, 'lat': lat, 'lon':lon}, 
                                          gepspatial_limits = {'lon_strat': lon_1, 'lon_end': lon_end, 'lat_start': lat_1, 'lat_end': lat_end})

    elif len(data_in.shape) == 3:
        Accuracy = []
        data_out = np.full((data_in.shape[0], len(lat), len(lon)), np.nan)
        for j in range(data_in.shape[0]):
            # fit a function to do inerpolation 
            interp = interpolate.RegularGridInterpolator((LAT, LON), np.squeeze(data_in[j,:,:]))
            # do interpolation
            data_out[j,:,:] = interp((grid_y, grid_x))  

            # Evaluate accuracy of interpolations 
            Accuracy.append(InterpolationAccuracyExaminer(original_data = {'data': np.squeeze(data_in[j,:,:]), 'lat': LAT, 'lon': LON},
                                                          interpolated_data = {'data': np.squeeze(data_out[j,:,:]), 'lat': lat, 'lon':lon},
                                                          gepspatial_limits = {'lon_strat': lon_1, 'lon_end': lon_end, 'lat_start': lat_1, 'lat_end': lat_end}))
    return data_out, lat, lon, Accuracy
# end of function

# Define a function to find pixels falling inside Europe Bourders
def InPolygon(poly, longitude, latitude):
    # convert lat and lon vectors of data to GeodataFrame
    x, y = np.meshgrid(longitude, latitude)
    points = pd.DataFrame({'lon' : x.flatten(), 'lat' : y.flatten()})
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.lon, points.lat))

    # read Europe Bourder shapefile from file
    # poly  = gpd.GeoDataFrame.from_file(shapfile)

    # set Coordinate System (CRS) of the points geometry to CRS oo poly shapefile
    points = points.set_crs(poly.crs)

    # indexation of pixels falling into Europe Bourder defined by poly
    pointInPolys = sjoin(points, poly, how='left')
    grouped = pointInPolys.groupby('index_right')
    KEY = get_first_key(grouped.groups)
    ind = np.array(grouped.groups[KEY])

    points_in = np.empty(points.shape[0], dtype=bool)
    points_in[:] = False
    points_in[ind] = True
    points_in = points_in.reshape((len(latitude), len(longitude)))
    return points_in
# end of function 

# function to create acronym
def Find_Acronym(stng):
   
    # add first letter
    oupt = stng[0]
     
    # iterate over string
    for i in range(1, len(stng)):
        if stng[i-1] == ' ':
           
            # add letter next to space
            oupt += stng[i]
             
    # uppercase oupt
    oupt = oupt.upper()
    return oupt

# define a function to remove outliers
def remove_outlier(df):
    df = pd.DataFrame(df)
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1

    # Values exceeding the following criterion ((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)) are considered as outliers
    df[((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))] = np.nan
    out = np.squeeze(np.array(df))
    return out

# Define a fucntion to sort data in way that both lat and lon vectors are in ascending orders.
# this is neccesry to make sure that data from different products are in similar shape
def DataHarmonizer(x, latitude, longitude):
    if len(x.shape) == 2:
        x = x[np.argsort(latitude), :][:, np.argsort(longitude)]
    elif len(x.shape) == 3:
        x = x[:, np.argsort(latitude), :][:, :, np.argsort(longitude)]

    latitude = latitude[np.argsort(latitude)]
    longitude = longitude[np.argsort(longitude)]
    return x, latitude, longitude
# end of function

# define a function to compue VPD from T, SH, and Ps
def VPD_Claculator(temprature, Specific_humididty, pressure):
    #Bolton, D., 1980: The computation of equivalent potential temperature. Mon. Wea. Rev., 108, 1046-1053.
    #temp in Kelvin
    #Specific_humididty: Near surface specific humidity in kg/kg
    #pressure: Surface pressure in Pa

    # Replcae missing values (-9999.0) with NaN
    temprature[temprature == -9999.0] = np.nan
    Specific_humididty[Specific_humididty == -9999.0] = np.nan
    pressure[pressure == -9999.0] = np.nan

    T = temprature - 273.15 # [°C];

    M_wet = 18.0152
    M_dry = 28.9644
    c = M_wet/ M_dry
    e = (Specific_humididty * pressure) / (c + (1 - c) * Specific_humididty)

    es = (0.6112 * np.exp(17.67 * T / (T + 243.5)))* 1000 # bar

    RH = 100 * e / es

    vpsat = 610.78 * np.exp(T / (T + 238.3) * 17.2694) #different approach to calculate vpsat

    VPD = vpsat * (1 - RH/100)

    VPD = VPD / 1000 # convert to kPa
    return VPD
# end of function

# Define a function to do surface ploting for geospatial data 
def Surface_Plot(ax, bourder_shapefile, longitude, latitude, data, colorbar_limits, colorbar_label, CMAP, xlim, ylim):
    # ax: the plot's axis
    # bourder_shapefile: geopandas shapfile of bourders of region you want to do plotting for
    # longitude and latitude (both in n*m shape): the meshgrid of lat and lon vectors attached to data
    # data (n*m shape): the data to be plotted
    # colorbar_limits:  the upper and lower limits of colorbar
    # colorbar_label: the label to be attached to colorbar
    # CMAP: colormap
    # xlim and ylim: the upper and lower limits of x and y axes

    im = ax.pcolor(longitude, latitude, data, edgecolors='none', linewidths=2, cmap= CMAP)
    im.set_clim(colorbar_limits)
    ax.set_xlabel('Longitude [DD]', **{'fontname':'Times New Roman', 'size':'13'})
    ax.set_ylabel('Latitude [DD]', **{'fontname':'Times New Roman', 'size':'13'})
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(visible=True, which='major', axis='both')

    bourder_shapefile.geometry.boundary.plot(color=None, edgecolor='k', linewidth = 1, ax=ax) #Use your second dataframe
    cbar = plt.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(colorbar_label, rotation = 270)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im, cbar
## end of function 

# define a function to plot the histogram of given data
def Histogram_plot(ax, data, n_bins, LABEL, line_color, xlim, ylim, Xlabel):
    # ax: the plot's axis
    # data: the vector of data
    # LABEL: lebel of data to be used in legend
    # line_color: the color to be used for best fit line
    # xlim and ylim: the upper and lower limits of x and y axes
    # Xlabel: the label to be used for x axis

    # fit a normal distribution over data provided in x
    (mu, sigma) = sst.norm.fit(data)
    kurtosis = sst.kurtosis(data, axis = 0, fisher = True, bias = True)
    skewness = sst.skew(data, axis = 0, bias = True)

    # plot histogram
    n, bins, patches = ax.hist(data, n_bins, density=True, stacked=True, facecolor='lightgray', alpha=0.75)
    
    # add a 'best fit' line
    y = sst.norm.pdf( bins, mu, sigma)
    ax.plot(bins, y, '--', color= line_color, linewidth=2, label =LABEL)    
    
    axis_font = {'fontname':'Times New Roman', 'size':'14'}
    # set some basic attrebutes of plot
    ax.set_xlabel(Xlabel, **axis_font)
    ax.set_ylabel('Probability density', **axis_font)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    return mu, sigma, skewness, kurtosis
## end of function ####################################################################################

# define a function to identify outliers in a 1D array using the Z-score method
def find_outliers(arr):
    """Identify outliers in a 1D array using the Z-score method."""
    z_scores = (arr - np.mean(arr)) / np.std(arr)
    abs_z_scores = np.abs(z_scores)
    threshold = 3 # threshold for identifying outliers
    return np.where(abs_z_scores > threshold)
