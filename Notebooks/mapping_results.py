#@title Python imports
# A bit of imports
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point, Polygon
sns.set(color_codes=True)
from sklearn import linear_model
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import statsmodels.api as statm

import libpysal as ps
from mgwr.gwr import GWR
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from spglm.family import Gaussian, Binomial, Poisson
import multiprocessing as mp
pool = mp.Pool()
import io

census = pd.read_csv("MGWR_session_results.csv")
b_cols = ['beta_Intercept', 'beta_avg_age','beta_avg_water_dist', 'beta_avg_sqft', 'beta_avg_basement','beta_avg_index', 'beta_avg_unemp', 'beta_avg_tech']
bt_cols = ['bt_constant','bt_age','bt_water_dist','bt_sqft','bt_round_basement','bt_index','bt_unemp','bt_tech']
t_cols = ['t_Intercept','t_avg_age', 't_avg_water_dist', 't_avg_sqft', 't_avg_basement','t_avg_index', 't_avg_unemp', 't_avg_tech']
t_crit = [2.92,2.94,2.67,2.99,3.01,2.14,2.20,2.41]

for i in range(8):
    census.loc[census[t_cols[i]] >=t_crit[i], bt_cols[i]] = census[b_cols[i]]
    census.loc[census[t_cols[i]] <=-t_crit[i], bt_cols[i]] = census[b_cols[i]]


tr = pd.read_csv("census_tracts/census_tracts.csv")

c='census_tracts/census_tracts.shp'
crs = {'EPSG':'4326'}
geo = gpd.read_file(c,crs=crs)[['geometry','objectid']]
fig,ax = plt.subplots(figsize=(20,15))
geo.plot(ax=ax)

coords = np.array(list(zip(census['x_coor'],census['y_coor'])))
geom_points = [Point(xy) for xy in coords]
geo_df = gpd.GeoDataFrame(census,crs={'init':'epsg:4326'},geometry=geom_points)
geo_df = geo_df.rename(columns={'OBJECTID':'index'})
geo_df = geo_df.to_crs(epsg=3857)
final_geo = gpd.sjoin(geo, geo_df, how='inner',op='contains',lsuffix='left',rsuffix='right')

fig,ax = plt.subplots(figsize=(20,15))
ax.set_facecolor('white')
final_geo.plot(ax=ax, color='gold')
geo_df.plot(ax=ax, markersize=8,alpha=1,color='tomato',marker="o")

import scipy as sp
import shapefile as shp

import matplotlib as mpl
import matplotlib.pyplot as plt
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))

label = gpd.read_file("more_labels_new/more_labels_new.shp")

label=label.to_crs(epsg=3857)
label=label.drop(label.index[3])
label=label.reset_index()
label=label.drop(label.index[5])
label=label.reset_index()
final_geo=final_geo.fillna(0.0)
def mapp(name,color,filename,normal=False):
    vmi=min(final_geo[name])
    vma=max(final_geo[name])
    figsize=(14,10)
    colors = 10
    norm = MidpointNormalize(vmin=vmi, vmax=vma, midpoint=0)
    colors = 6
    fig, ax = plt.subplots(1, figsize=(14, 14))

    if normal==True:
        final_geo.plot(column=name, ax=ax,cmap=color,figsize=figsize,k=colors, linewidth=0.5,norm=norm)
    else:
        final_geo.plot(column=name, ax=ax,cmap=color,figsize=figsize,k=colors, linewidth=0.5)

    ax.axis("off")

    Scalebar = ScaleBar(100000,location='lower left') # 1 pixel = 0.2 meter
    scatter = ax.collections[-1]

    plt.colorbar(scatter, ax=ax, extend='min',orientation='horizontal',fraction=0.046, pad=0.04)
    final_geo.boundary.plot(ax=ax,color='grey',alpha=0.2)

    texts = []
    for x, y, lab in zip(label.geometry.x, label.geometry.y, label["names"]):
      texts.append(plt.text(x-1500, y-700, lab, fontsize = 8,horizontalalignment='left',verticalalignment='baseline',bbox=dict(facecolor='white', alpha=0.7,linewidth=0.0)))
    label.plot(ax=ax,alpha=1,color='black',linewidth=0.4)


    plt.savefig("../images/"+filename)
