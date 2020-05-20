
# <center> Mapping Parameter Coefficients </center>

**Notebook Outline:**  
  
**An example of hedonic house price modeling using MGWR**
- [Code to Mask the Significant Coefficients](#Mask-the-Significant-Coefficients)
- [Spatial Join - Results to Shapefile](#Spatial-Join---Results-to-Shapefile)
- [Maps of all Parameter Coeffcients](#Maps-of-all-Parameter-Coefficients) 
- [Interpretation of Maps](#Interpretation-of-Maps)<br><br>

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_workshop_book/)


### If you want to follow along with the code, follow [this link](https://colab.research.google.com/drive/1oqnwg_HkY-L_MdRTT2qg5EL-LRrOzmbd?usp=sharing)

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

# Map of all Parameter Coefficients


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

***
