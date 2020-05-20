

```python
%matplotlib inline
%config InlineBackend.figure_format='retina'
import pandas as pd
import numpy as np
import mgwr
import pickle
import matplotlib.pyplot as plt
from io import StringIO
```


```python
filename = '../bws_global_start.csv'

def plotBws(filename, title):
    with open(filename, 'r') as inf:
        lines = inf.readlines()
        soclines = [i.replace('Current iteration: ', '').replace('SOC: ', '') for i in lines[0::2]]
        bwlines = [i.replace('Bandwidths: ', '') for i in lines[1::2]]

        soc = pd.read_csv(StringIO('\n'.join(soclines)), header=None)
        bw = pd.read_csv(StringIO('\n'.join(bwlines)), header=None)

        soc.columns = ['Iteration', 'SOC']
        bw.columns = [
            'const',
     'avg_commute_km',
     # Alternate specification of commute distance
     #'pct_commute_less2km', 'pct_commute_2_10km', 'pct_commute_10_20km',
     'pctbachdeg',
     'pctnovehicle',
      # the original paper had the low-income cutoff at 20k but that's just really low
     'income0_35k',
     'income35_50k',
      #'income50_100k', # removed due to multicollinearity
     'pcthhkids',
     'pct_w_cbd', # NB high correlation with bachelor's degree
     'pctmultifamily',
     'pctrent',
     'luentropy',
     'access60', # lower cutoff metric might be preferable, access30 and access45 are also available
     'lnldist_km',
     'lnpopdens_km2',
     'lnjob_dens_500m_km2',
     'pct_w_chicago'
    ]
        f, ax1 = plt.subplots(figsize=(16, 12))

        for idx, col in enumerate(bw.columns):
            ax1.plot(bw.index, bw[col], label=col, ls=('--' if idx > 9 else None), lw=(3 if col == 'pctnovehicle' else 0.75))

        ax2 = ax1.twinx()
        ax2.semilogy()
        ax2.plot(soc.index, soc.SOC, label='SOC', ls=':', color='black')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Bandwidth')
        ax2.set_ylabel('SOC')

        plt.title(title)
        
        ax1.legend()
        ax2.legend()
        
        return bw, soc

    
    
    
```


```python
global_bws, global_soc = plotBws('../bws_global_start.csv', 'Starting value: all bandwidths global (1996)')
```


![png](Plot%20Bandwidths_files/Plot%20Bandwidths_2_0.png)



```python
local_bws, local_soc = plotBws('../bws_local_start.csv', 'Starting value: all bandwidths 30')
```


![png](Plot%20Bandwidths_files/Plot%20Bandwidths_3_0.png)



```python
gwr_bws, gwr_soc = plotBws('../bws_gwr_start.csv', 'Starting value: GWR bandwidth')
```


![png](Plot%20Bandwidths_files/Plot%20Bandwidths_4_0.png)



```python
# plot all triads
plt.subplots(figsize=(16, 12))
for i, col in enumerate(global_bws.columns):
    idx = i // 4 * 5 + i % 4 + 1
    plt.subplot(4, 5, idx)
    
    plt.title(col)
    plt.plot(global_bws.index / max(global_bws.index), global_bws[col])
    plt.plot(local_bws.index / max(local_bws.index), local_bws[col])
    plt.plot(gwr_bws.index / max(gwr_bws.index), gwr_bws[col])
    plt.xticks([])
    plt.yticks([])
    
plt.subplot(4, 5, 5)
plt.plot([1], [1], label='1996')
plt.plot([1], [1], label='30')
plt.plot([1], [1], label='GWR')
plt.legend(title='Starting bandwidth')
plt.xticks([])
plt.yticks([])
```




    ([], <a list of 0 Text yticklabel objects>)




![png](Plot%20Bandwidths_files/Plot%20Bandwidths_5_1.png)



```python
len(gwr_bws)
```




    85




```python
bws.columns = [
    'const',
 'avg_commute_km',
 # Alternate specification of commute distance
 #'pct_commute_less2km', 'pct_commute_2_10km', 'pct_commute_10_20km',
 'pctbachdeg',
 'pctnovehicle',
  # the original paper had the low-income cutoff at 20k but that's just really low
 'income35_50k',
  #'income50_100k', # removed due to multicollinearity
 'pct_w_cbd', # NB high correlation with bachelor's degree
 'pctrent',
 'access60', # lower cutoff metric might be preferable, access30 and access45 are also available
 'lnldist_km',
 'lnpopdens_km2',
 'lnjob_dens_500m_km2',
 'pct_w_chicago'
]
```


```python
plt.figure(figsize=(16, 12))
for idx, col in enumerate(bws.columns):
    plt.plot(bws.index, bws[col], label=col, ls=('--' if idx > 9 else None), lw=(3 if col == 'pctnovehicle' else 0.75))
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Bandwidth')
```




    Text(0, 0.5, 'Bandwidth')




![png](Plot%20Bandwidths_files/Plot%20Bandwidths_8_1.png)



```python
models['mgwr']['mod'].bws
```




    array([1996.,  450., 1996., 1996.,  966., 1986.,  192., 1996., 1996.,
           1819., 1986., 1996.,  821., 1976.,  240.,  664.])




```python

```
