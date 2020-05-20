

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
df = pd.read_csv("example_dataset.csv")
```


```python
df.columns
```




    Index(['the_geom', 'longitude', 'the_geom_webmercator', 'avg_tech', 'latitude',
           'houses', 'avg_unemp', 'cartodb_id', 'avg_index', 'avg_price',
           'avg_basement', 'avg_sqft', 'avg_water_dist', 'avg_age', 'ind'],
          dtype='object')




```python
df['ln_price']=np.log(df['avg_price'])
```


```python
df['ln_price']
```




    0      12.611636
    1      12.312879
    2      12.563657
    3      12.715962
    4      12.619820
             ...    
    354    12.936781
    355    12.733797
    356    12.839368
    357    13.007454
    358    13.426880
    Name: ln_price, Length: 359, dtype: float64




```python
df=df.dropna()
```


```python
import statsmodels.api as statm
X=df[['avg_tech','avg_unemp','avg_index','avg_sqft','avg_basement','avg_water_dist','avg_age']].copy()
X_std = (X-X.mean(axis=0))/X.std(axis=0)
X_std=statm.add_constant(X_std)
y=df['ln_price']
y_std = (y-y.mean(axis=0))/y.std(axis=0)
model = statm.OLS(y_std,X_std).fit()
predictions=model.predict(X_std)

model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>ln_price</td>     <th>  R-squared:         </th> <td>   0.898</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.896</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   438.4</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 20 May 2020</td> <th>  Prob (F-statistic):</th> <td>1.15e-168</td>
</tr>
<tr>
  <th>Time:</th>                 <td>12:47:31</td>     <th>  Log-Likelihood:    </th> <td> -98.782</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   357</td>      <th>  AIC:               </th> <td>   213.6</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   349</td>      <th>  BIC:               </th> <td>   244.6</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>          <td> 6.721e-15</td> <td>    0.017</td> <td> 3.93e-13</td> <td> 1.000</td> <td>   -0.034</td> <td>    0.034</td>
</tr>
<tr>
  <th>avg_tech</th>       <td>    0.2186</td> <td>    0.029</td> <td>    7.433</td> <td> 0.000</td> <td>    0.161</td> <td>    0.276</td>
</tr>
<tr>
  <th>avg_unemp</th>      <td>   -0.2990</td> <td>    0.028</td> <td>  -10.802</td> <td> 0.000</td> <td>   -0.353</td> <td>   -0.245</td>
</tr>
<tr>
  <th>avg_index</th>      <td>   -0.1231</td> <td>    0.030</td> <td>   -4.115</td> <td> 0.000</td> <td>   -0.182</td> <td>   -0.064</td>
</tr>
<tr>
  <th>avg_sqft</th>       <td>    0.5865</td> <td>    0.023</td> <td>   25.043</td> <td> 0.000</td> <td>    0.540</td> <td>    0.633</td>
</tr>
<tr>
  <th>avg_basement</th>   <td>    0.0984</td> <td>    0.023</td> <td>    4.212</td> <td> 0.000</td> <td>    0.052</td> <td>    0.144</td>
</tr>
<tr>
  <th>avg_water_dist</th> <td>   -0.2106</td> <td>    0.034</td> <td>   -6.280</td> <td> 0.000</td> <td>   -0.277</td> <td>   -0.145</td>
</tr>
<tr>
  <th>avg_age</th>        <td>    0.1563</td> <td>    0.025</td> <td>    6.371</td> <td> 0.000</td> <td>    0.108</td> <td>    0.205</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>17.936</td> <th>  Durbin-Watson:     </th> <td>   1.793</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  44.569</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.139</td> <th>  Prob(JB):          </th> <td>2.10e-10</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.708</td> <th>  Cond. No.          </th> <td>    4.47</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
import libpysal as ps
from mgwr.gwr import GWR
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from spglm.family import Gaussian, Binomial, Poisson
import multiprocessing as mp
pool = mp.Pool()
```


```python
df.columns
```




    Index(['the_geom', 'longitude', 'the_geom_webmercator', 'avg_tech', 'latitude',
           'houses', 'avg_unemp', 'cartodb_id', 'avg_index', 'avg_price',
           'avg_basement', 'avg_sqft', 'avg_water_dist', 'avg_age', 'ind',
           'ln_price'],
          dtype='object')




```python
coords = np.array(list(zip(df['longitude'],df['latitude'])))
y = np.array(df['ln_price']).reshape((-1,1))
y_std = (y-y.mean(axis=0))/y.std(axis=0)
X=df[['avg_tech','avg_unemp','avg_index','avg_sqft','avg_basement','avg_water_dist','avg_age']].values
X_std=(X-X.mean(axis=0))/X.std(axis=0)
selector_gwr = Sel_BW(coords, y_std, X_std)
```


```python
selector_mgwr = Sel_BW(coords, y_std, X_std, multi=True)
```


```python
selector_mgwr.search(pool=pool)
```




    array([ 43., 164., 226., 191.,  45.,  81.,  45., 354.])




```python
%%time
model_mgwr = MGWR(coords,y_std,X_std,selector_mgwr,fixed=False,kernel='bisquare',sigma2_v1=True)
results_mgwr=model_mgwr.fit()
```


    HBox(children=(IntProgress(value=0, description='Inference', max=1), HTML(value='')))


    
    Wall time: 7.64 s
    


```python
results_mgwr.R2
```




    0.9775393617728864




```python
df2 = pd.read_csv("MGWR_session_results.csv")
```


```python
df2['w_43']=results_mgwr.W[0][171]
df2['w_164']=results_mgwr.W[1][171]
df2['w_226']=results_mgwr.W[2][171]
df2['w_81']=results_mgwr.W[5][171]
df2['w_354']=results_mgwr.W[7][171]
```


```python
df2.to_csv("MGWR_session_results.csv")
```
