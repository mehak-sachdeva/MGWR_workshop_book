

```python
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
%matplotlib inline
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
```


```python
census = pd.read_csv("MGWR_session_results.csv")
```


```python
census.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
           'Unnamed: 0.1.1.1.1', 'ind', 'x_coor', 'y_coor', 'y', 'ols_residual',
           'mgwr_yhat', 'mgwr_residual', 'localR2', 'beta_Intercept',
           'beta_avg_age', 'beta_avg_basement', 'beta_avg_sqft',
           'beta_avg_water_dist', 'beta_avg_unemp', 'beta_avg_tech',
           'beta_avg_index', 'se_Intercept', 'se_avg_age', 'se_avg_basement',
           'se_avg_sqft', 'se_avg_water_dist', 'se_avg_unemp', 'se_avg_tech',
           'se_avg_index', 't_Intercept', 't_avg_age', 't_avg_basement',
           't_avg_sqft', 't_avg_water_dist', 't_avg_unemp', 't_avg_tech',
           't_avg_index', 'p_Intercept', 'p_avg_age', 'p_avg_basement',
           'p_avg_sqft', 'p_avg_water_dist', 'p_avg_unemp', 'p_avg_tech',
           'p_avg_index', 'sumW_Intercept', 'sumW_avg_age', 'sumW_avg_basement',
           'sumW_avg_sqft', 'sumW_avg_water_dist', 'sumW_avg_unemp',
           'sumW_avg_tech', 'sumW_avg_index', 'w_43', 'w_164', 'w_226', 'w_81',
           'w_354'],
          dtype='object')




```python
b_cols = ['beta_Intercept', 'beta_avg_age','beta_avg_water_dist', 'beta_avg_sqft', 'beta_avg_basement','beta_avg_index', 'beta_avg_unemp', 'beta_avg_tech']
bt_cols = ['bt_constant','bt_age','bt_water_dist','bt_sqft','bt_round_basement','bt_index','bt_unemp','bt_tech']
t_cols = ['t_Intercept','t_avg_age', 't_avg_water_dist', 't_avg_sqft', 't_avg_basement','t_avg_index', 't_avg_unemp', 't_avg_tech']
t_crit = [2.92,2.94,2.67,2.99,3.01,2.14,2.20,2.41]
```


```python
for i in range(8):
    census.loc[census[t_cols[i]] >=t_crit[i], bt_cols[i]] = census[b_cols[i]]
    census.loc[census[t_cols[i]] <=-t_crit[i], bt_cols[i]] = census[b_cols[i]]
```


```python
tr = pd.read_csv("census_tracts/census_tracts.csv")
```


```python
c='census_tracts/census_tracts.shp'
crs = {'EPSG':'4326'}
geo = gpd.read_file(c,crs=crs)[['geometry','objectid']]
fig,ax = plt.subplots(figsize=(20,15))
geo.plot(ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x25763abaeb8>




![png](Mapping_code_files/Mapping_code_6_1.png)



```python
geo.crs
```




    {'init': 'epsg:3857'}




```python
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
```




    <matplotlib.axes._subplots.AxesSubplot at 0x257655d8278>




![png](Mapping_code_files/Mapping_code_8_1.png)



```python
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
```


```python
label = gpd.read_file("more_labels_new/more_labels_new.shp")
```


```python
label=label.to_crs(epsg=3857)
label=label.drop(label.index[3])
label=label.reset_index()
label=label.drop(label.index[5])
label=label.reset_index()
label
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_0</th>
      <th>index</th>
      <th>FID_</th>
      <th>Field1</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>names</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47.600000</td>
      <td>-122.3000</td>
      <td>Seattle</td>
      <td>POINT (-13614373.72401736 6040565.208625006)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>47.610100</td>
      <td>-122.2015</td>
      <td>Bellevue</td>
      <td>POINT (-13603408.75417422 6042232.762266758)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>47.674000</td>
      <td>-122.1215</td>
      <td>Redmond</td>
      <td>POINT (-13594503.19491076 6052790.400185969)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>47.627740</td>
      <td>-122.2420</td>
      <td>Bill Gates' House</td>
      <td>POINT (-13607917.19355135 6045145.975053936)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>47.655548</td>
      <td>-122.2950</td>
      <td>University of Washington</td>
      <td>POINT (-13613817.12656339 6049740.411229585)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>47.488300</td>
      <td>-121.9467</td>
      <td>Tiger Mountain State Forest</td>
      <td>POINT (-13575044.54792009 6022144.481821851)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>47.479700</td>
      <td>-122.2079</td>
      <td>Renton</td>
      <td>POINT (-13604121.1989153 6020727.859753064)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>47.380900</td>
      <td>-122.2348</td>
      <td>Kent</td>
      <td>POINT (-13607115.69321764 6004469.784188417)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>47.322300</td>
      <td>-122.3126</td>
      <td>Federal Way</td>
      <td>POINT (-13615776.34960135 5994841.227999952)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>47.680000</td>
      <td>-122.2290</td>
      <td>Lake Washington</td>
      <td>POINT (-13606470.04017103 6053782.391355693)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>47.608500</td>
      <td>-122.0878</td>
      <td>Lake Sammamish</td>
      <td>POINT (-13590751.72807102 6041968.573888839)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>47.450200</td>
      <td>-122.3088</td>
      <td>SeaTac Airport</td>
      <td>POINT (-13615353.33553634 6015870.277732232)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>47.585900</td>
      <td>-122.4014</td>
      <td>Alki Beach Park</td>
      <td>POINT (-13625661.5203838 6038237.776014488)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>47.625300</td>
      <td>-122.3222</td>
      <td>Capitol Hill</td>
      <td>POINT (-13616845.01671297 6044742.955066294)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>47.679200</td>
      <td>-122.3860</td>
      <td>Ballard</td>
      <td>POINT (-13623947.20022558 6053650.119275319)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>47.657300</td>
      <td>-122.4055</td>
      <td>Discovery Park</td>
      <td>POINT (-13626117.93029605 6050029.958486892)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>47.756000</td>
      <td>-122.3457</td>
      <td>Shoreline</td>
      <td>POINT (-13619461.02474661 6066357.500503332)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>47.758400</td>
      <td>-122.2497</td>
      <td>Kenmore</td>
      <td>POINT (-13608774.35363046 6066754.908203133)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>47.528300</td>
      <td>-122.0997</td>
      <td>Cougar Mountain Park</td>
      <td>POINT (-13592076.43001146 6028736.472030034)</td>
    </tr>
  </tbody>
</table>
</div>




```python
def mapp(col,name,color,filename,normal=False):  
    vmi=min(col[name])
    vma=max(col[name])
    figsize=(14,10)
    colors = 10
    norm = MidpointNormalize(vmin=vmi, vmax=vma, midpoint=0)
    colors = 6
    fig, ax = plt.subplots(1, figsize=(14, 14))

    if normal==True:
        col.plot(column=name, ax=ax,cmap=color,figsize=figsize,k=colors, linewidth=0.5,norm=norm)
    else:
        col.plot(column=name, ax=ax,cmap=color,figsize=figsize,k=colors, linewidth=0.5)

    ax.axis("off")

    Scalebar = ScaleBar(100000,location='lower left') # 1 pixel = 0.2 meter
    scatter = ax.collections[-1]

    plt.colorbar(scatter, ax=ax, extend='min',orientation='horizontal',fraction=0.046, pad=0.04)
    col.boundary.plot(ax=ax,color='grey',alpha=0.2)

    texts = []
    for x, y, lab in zip(label.geometry.x, label.geometry.y, label["names"]):
      texts.append(plt.text(x-1500, y-700, lab, fontsize = 8,horizontalalignment='left',verticalalignment='baseline',bbox=dict(facecolor='white', alpha=0.7,linewidth=0.0)))
    label.plot(ax=ax,alpha=1,color='black',linewidth=0.4)
    

    plt.savefig("../images/"+filename)
```


```python
final_geo.columns
```




    Index(['geometry', 'objectid', 'index_right', 'Unnamed: 0', 'Unnamed: 0.1',
           'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1', 'ind',
           'x_coor', 'y_coor', 'y', 'ols_residual', 'mgwr_yhat', 'mgwr_residual',
           'localR2', 'beta_Intercept', 'beta_avg_age', 'beta_avg_basement',
           'beta_avg_sqft', 'beta_avg_water_dist', 'beta_avg_unemp',
           'beta_avg_tech', 'beta_avg_index', 'se_Intercept', 'se_avg_age',
           'se_avg_basement', 'se_avg_sqft', 'se_avg_water_dist', 'se_avg_unemp',
           'se_avg_tech', 'se_avg_index', 't_Intercept', 't_avg_age',
           't_avg_basement', 't_avg_sqft', 't_avg_water_dist', 't_avg_unemp',
           't_avg_tech', 't_avg_index', 'p_Intercept', 'p_avg_age',
           'p_avg_basement', 'p_avg_sqft', 'p_avg_water_dist', 'p_avg_unemp',
           'p_avg_tech', 'p_avg_index', 'sumW_Intercept', 'sumW_avg_age',
           'sumW_avg_basement', 'sumW_avg_sqft', 'sumW_avg_water_dist',
           'sumW_avg_unemp', 'sumW_avg_tech', 'sumW_avg_index', 'w_43', 'w_164',
           'w_226', 'w_81', 'w_354', 'bt_constant', 'bt_age', 'bt_water_dist',
           'bt_sqft', 'bt_round_basement', 'bt_index', 'bt_unemp', 'bt_tech'],
          dtype='object')




```python
final_geo['btn_water']=final_geo['bt_water_dist'].fillna(0.0)
```


```python
mapp(col=final_geo,name='btn_water',color='Blues_r',filename="trial",normal=False)
```


![png](Mapping_code_files/Mapping_code_15_0.png)



```python
final_geo['bt_age']
```




    0           NaN
    1           NaN
    2      0.099947
    3           NaN
    4           NaN
             ...   
    352         NaN
    353         NaN
    354         NaN
    355         NaN
    356         NaN
    Name: bt_age, Length: 363, dtype: float64




```python

```
