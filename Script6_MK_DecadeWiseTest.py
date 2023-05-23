import numpy as np
import pymannkendall as mk

# define a function to apply MK test
def Apply_MK_OverEachPixel(x):
    # allocate free space for the outputs
    slope = np.nan
    intcp = np.nan
    pValue = np.nan
    trendType = ''

    # determine the number of data
    n = len(x)

    # check if pixel has no data, then retrun
    if len(x[~np.isnan(x)]) != n:
        return slope, intcp, pValue, trendType

    # adjusting for autocorrelation with lag -1
    xv = x - np.nanmean(x)
    sum1 = n *np.nansum(xv[0:(n-1)] * xv[1:n])
    sum2 = (n-1)*np.nansum(xv**2)
    r = sum1/sum2
    x = x[1:n] - r*x[0:n-1]

    # apply original MK test over data    
    # to ignore the error due to existing NaN data
    try:
        Mann_Kendall_Test = mk.sens_slope(x[~np.isnan(x)])

        # stores the outputs
        slope = Mann_Kendall_Test.slope
        intcp = Mann_Kendall_Test.intercept
        # pValue = Mann_Kendall_Test.p
        # trendType = Mann_Kendall_Test.trend
    except:
        pass
    return slope, intcp, pValue, trendType
# end of function


def Apply_MK_OverEurope(X, year, peiords):
    slope = np.full((X.shape[1], X.shape[2], 4), np.nan)
    intcp = np.full((X.shape[1], X.shape[2], 4), np.nan)
    pValue = np.full((X.shape[1], X.shape[2], 4), np.nan)
    trendType = np.full((X.shape[1], X.shape[2], 4), '', dtype=str, )

    for pixel, _ in np.ndenumerate(X[0,:,:]):
        for k in range(len(peiords)-1):
            ind = np.logical_and(year >= peiords[k], year < peiords[k+1])
            slope[pixel[0], pixel[1], k], intcp[pixel[0], pixel[1], k], pValue[pixel[0], pixel[1], k], trendType[pixel[0], pixel[1], k] = Apply_MK_OverEachPixel(np.squeeze(X[ind, pixel[0], pixel[1]]))
    return slope, intcp, pValue, trendType


###################### calculations #######################################
# load data
data = np.load('#Outputs/5_LFD_Results_Averaged.npz', allow_pickle=True)
data = dict(zip((data.files), (data[k] for k in data)))  # Convert numpy.lib.npyio.NpzFile object to dict

year = data['year']
lon = data['lon']
lat = data['lat']

peiods = (1982, 1991, 2001, 2011, 2021)

# Do MK analysis
# OG
slope, intcp, pValue, trendType = Apply_MK_OverEurope(data['OG'], year, peiods)
OG = {'slope': slope, 'intercept': intcp, 'pvalue': pValue, 'trendType': trendType}

# OD
slope, intcp, pValue, trendType = Apply_MK_OverEurope(data['OD'], year, peiods)
OD = {'slope': slope, 'intercept': intcp, 'pvalue': pValue, 'trendType': trendType}

# LGS
slope, intcp, pValue, trendType = Apply_MK_OverEurope(data['LGS'], year, peiods)
LGS = {'slope': slope, 'intercept': intcp, 'pvalue': pValue, 'trendType': trendType}

np.savez('#Outputs/7_MannKendall_Results_DecadeWise', 
        OG=OG, OD=OD, LGS=LGS, lon=lon, lat=lat, periods = peiods)