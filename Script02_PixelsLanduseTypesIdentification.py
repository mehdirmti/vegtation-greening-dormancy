import geopandas as gpd
import numpy as np
import netCDF4
import pandas as pd
from matplotlib import pyplot as plt

# importing the User defined Function
import UserDefined_Auxiliary_Functions as AuxiliaryFunctions 


# read lat and lon data from one of used data
nc = netCDF4.Dataset('#Outputs/#1 Data/ndvi-Euro-GIMMS.nc')
lat = np.array(nc.variables['lat'])
lon = np.array(nc.variables['lon'])
ndvi = np.array(nc.variables['ndvi'])

# Identify landuses of each pixel
poly  = gpd.GeoDataFrame.from_file('#shapefiles/Euro_Landuse.shp')
landuses = list(poly.Landuse)

# read north arrow image to be added on plots
north_arrow = plt.imread('#Outputs/north arrow.png')

# get discrete colormap
cmap = plt.get_cmap('jet', len(landuses))

# plot land use shapefile
fig, ax = plt.subplots(1, 1)
# set some initial settings on the figure
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})
fig.set_figheight(8)
fig.set_figwidth(14)
poly.plot(column='Landuse', 
          cmap = cmap, 
          ax=ax, 
          legend=True,
          legend_kwds={'loc':'center left', 'bbox_to_anchor':(1, 0.5)}, 
          vmin= 0 - 0.5,
          vmax=len(landuses) + 0.5) #Use your second dataframe
ax.grid(visible=True, which='major', axis='both')
ax.set_xlim([-30, 80])
ax.set_ylim([30, 90])
ax.set_xlabel('Longitude [DD]', **{'fontname':'Times New Roman', 'size':'13'})
ax.set_ylabel('Latitude [DD]', **{'fontname':'Times New Roman', 'size':'13'})
plt.subplots_adjust(left=0.075, bottom=0.1, right=0.85, top=0.925, wspace=0.3, hspace=None)

newax = fig.add_axes([0.175,0.75,0.15,0.15], anchor='NE', zorder=1)
newax.imshow(north_arrow)
newax.axis('off')

fig.savefig('#Outputs/Script 2/Europe_Landuse.jpg', dpi=600, transparent=False)   # save the figure to file
plt.show()

# allocate space to store landuses type for pixels
LU = np.empty((len(lat), len(lon)))
LU[:] = np.nan
for i in range (len(landuses)):
    LU[AuxiliaryFunctions.InPolygon(poly[poly.Landuse == landuses[i]], lon, lat)] = i

# Check dominant landuse types in Europe
LU_uniq, LU_Counts = np.unique(LU[~np.isnan(LU)], return_counts=True)
for i in range(len(LU_uniq)):
    if LU_Counts[i] < 250:
        LU[LU == i] = -1
        landuses[i] = 'Others'

landuses = [landuses[i] for i in np.flip(np.argsort(LU_Counts))]
LU_uniq = LU_uniq[np.flip(np.argsort(LU_Counts))]

# redefine the indexes for the prodominant landuse types
for i  in range(7):
    LU[LU == LU_uniq[i]] = i
LU[LU == -1] = 7 # set index for others to be 7

landuses = [landuses[i] for i in range(8)]

# check one again the unique landuse types to make sure everything is fine
LU_uniq, LU_Counts = np.unique(LU[~np.isnan(LU)], return_counts=True)

Stat = pd.DataFrame({'landuse': landuses, 'index': LU_uniq, 'Count': LU_Counts})

# store results for further uses
np.savez('#Outputs/4_PixelsLanduses', info = Stat, landuse = LU)
