
# <center> Mapping Parameter Coefficients </center>

**Notebook Outline:**  
  
**An example of hedonic house price modeling using MGWR**
- [Code to Mask the Significant Coefficients](#Mask-the-Significant-Coefficients)
- [Spatial Join - Results to Shapefile](#Spatial-Join---Results-to-Shapefile)
- [Maps of all Parameter Coeffcients](#Maps-of-all-Parameter-Coefficients) 
- [Interpretation of Maps](#Interpretation-of-Maps)<br><br>

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_workshop_book/)


### If you want to follow along with the code, follow [this link](https://colab.research.google.com/drive/1oqnwg_HkY-L_MdRTT2qg5EL-LRrOzmbd?usp=sharing)

### The shapefiles for mapping the parameter coefficients can be [downloaded here](https://github.com/mehak-sachdeva/MGWR_Workshop_2020/archive/master.zip)

# Mask the Significant Coefficients


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

# Spatial Join - Results to Shapefile


```python
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
```

# Maps of all Parameter Coefficients


```python
import mapping_results as maps
```

### All parameter estimates for age covariate


```python
maps.mapp(name='beta_avg_age',color='RdBu_r',filename="b_age",normal=False)
```


![png](Mapping_files/Mapping_10_0.png)


### Only significant parameter estimates for age covariate


```python
maps.mapp(name='bt_age',color='Reds',filename="bt_age",normal=False)
```


![png](Mapping_files/Mapping_12_0.png)



```python
maps.mapp(name='bt_water_dist',color='Blues_r',filename="bt_water_dist",normal=False)
```


![png](Mapping_files/Mapping_13_0.png)



```python
maps.mapp(name='bt_sqft',color='Reds',filename="bt_sqft",normal=False)
```


![png](Mapping_files/Mapping_14_0.png)



```python
maps.mapp(name='bt_unemp',color='Blues_r',filename="bt_unemp",normal=False)
```


![png](Mapping_files/Mapping_15_0.png)



```python
maps.mapp(name='bt_tech',color='Reds',filename="bt_tech",normal=False)
```


![png](Mapping_files/Mapping_16_0.png)



```python
maps.mapp(name='bt_constant',color='RdBu_r',filename="bt_constant",normal=False)
```


![png](Mapping_files/Mapping_17_0.png)


# Interpretation of Maps

### Interpreting the map with distance to nearest waterfront


```python
maps.mapp(name='bt_water_dist',color='Blues_r',filename="bt_water_dist",normal=False)
```


![png](Mapping_files/Mapping_20_0.png)


#### Scale of spatial units used is essential in interpreting the results of the parameter coefficients

The model that we are using is a log-linear model, where only the dependent variable (*y*) is log-transformed. The interpretation of *Beta* would hence be calculated using the following formula:

**One unit increase in *x* would change the house price by (exponent(Beta)-1) x 100 %**

[Source](https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/)


```python
import numpy as np
(np.exp(-0.6)-1)*100
```




    -45.11883639059735



From the map above, with every unit increase in the average distance of houses to the waterfront in the census tract containing University of Washington and Bill Gate's House, holding all other variable constant, the house price would decrease by approximately **45%** (since the Beta is -0.6). This is a huge change and is observed to be more prominent around the areas adjacent to Lake Washington and near Discovery park and Alki beach.

As clearly seen in the map, the effect of change of average distance to the waterfront for all census tracts in the county is not the same. In fact, in some region the parameter is not even significant.

If we try and recall the parameter coefficient from the Global regression model for the same covariate, the value of beta was -0.21. This result would suggest, that no matter which census tract is in question, a unit decrease in average distance to the waterfront for the houses, would reduce the house prices by:
**18%**. <br><br>
As seen in this example (and as can be expanded using other parameter coefficient results), results using the MGWR technique provide a lot more information by localizing the results. Since it is unrealistic to expect that the same stimulus would create the same response irrespective of the geography or location, MGWR results demonstrate the power of local regression by producing a much better model fit and more plausible and intuitive coefficient estimates.    


```python
(np.exp(-0.21)-1)*100
```




    -18.941575402981293



[Previous](http://mehak-sachdeva.github.io/MGWR_workshop_book/Html/Interpretation)

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_workshop_book/)

***
