
**Notebook Outline:**  
  
- [Setup with libraries](#Set-up-Cell)
- [List bandwidths from pickles](#List-bandwidths-from-pickles)
- [Parameter functions](#Parameter-functions)
- [GWR bandwidth](#GWR-bandwidth)
- [MGWR bandwidths](#MGWR-bandwidths)
- [AIC, AICc, BIC check](#AIC,-AICc,-BIC-check)
    - [AIC, AICc, BIC Boxplots for comparison](#AIC,-AICc,-BIC-Boxplots-for-comparison)
- [Parameter comparison from MGWR and GWR](#Parameter-comparison-from-MGWR-and-GWR)

Monte Carlo experiment code can be found in path mgwr/notebooks/Poisson_MC_script/

### Set up Cell


```python
import warnings
warnings.filterwarnings("ignore")
import pickle
import sys
import seaborn as sns
import numpy as np
sys.path.append("C:/Users/msachde1/Downloads/Research/Development/mgwr/notebooks/Binomial_MC_script_2/")
import model_mc
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

    C:\Users\msachde1\AppData\Local\Continuum\anaconda3\envs\gwrenv\lib\site-packages\libpysal\io\iohandlers\__init__.py:25: UserWarning: SQLAlchemy and Geomet not installed, database I/O disabled
      warnings.warn('SQLAlchemy and Geomet not installed, database I/O disabled')
    

### List bandwidths from pickles


```python
mgwr_bw0=[]
mgwr_bw1=[]
mgwr_bw2=[]
gwr_bw=[]
```


```python
for i in range(0,500,10):
    p1 = pickle.load( open( "C:/Users/msachde1/Downloads/Research/Development/mgwr/notebooks/Binomial_MC_script_we/pkls/results-{}-{}.pkl".format(str(i), str(i+10)),"rb") )
    for j in range(10):
        mgwr_bw0.append(p1[j].mgwr_bw[0][0])
        mgwr_bw1.append(p1[j].mgwr_bw[0][1])
        mgwr_bw2.append(p1[j].mgwr_bw[0][2])
        gwr_bw.append(p1[j].gwr_bw[0])
```

### Parameter functions


```python
def add(a,b):
    return 1+((1/120)*(a+b))

def con(u,v):
    return (0*(u)*(v))+0.3

def sp(u,v):
    return 1+1/3240*(36-(6-u/2)**2)*(36-(6-v/2)**2)

def med(u,v):
    B = np.zeros((25,25))
    for i in range(25):
        for j in range(25):
            
            if u[i][j]<=8:
                B[i][j]=0.2
            elif u[i][j]>17:
                B[i][j]=0.7
            else:
                B[i][j]=0.5
    return B
```


```python
x = np.linspace(0, 25, 25)
y = np.linspace(25, 0, 25)
X, Y = np.meshgrid(x, y)

B0=con(X,Y)
#B1=add(X,Y)
B1=sp(X,Y)
B2=med(X,Y)
```


```python
x = np.linspace(0, 25, 25)
y = np.linspace(25, 0, 25)
```


```python
x = np.linspace(0, 25, 25)
y = np.linspace(25, 0, 25)
```

### GWR bandwidth


```python
sns.distplot(gwr_bw)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e830ad0b8>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_13_1.png)



```python
np.mean(gwr_bw)
```




    243.354



### MGWR bandwidths


```python
plt.imshow(B0, extent=[0,10, 0, 10], origin='lower',cmap='Blues')
plt.colorbar()
plt.axis(aspect='image')
plt.xticks([])
plt.yticks([])
```




    ([], <a list of 0 Text yticklabel objects>)




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_16_1.png)



```python
sns.distplot(mgwr_bw0)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e837a1d30>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_17_1.png)



```python
np.mean(mgwr_bw0)
```




    532.386




```python
plt.imshow(B1, extent=[0,25, 0, 25], origin='lower',cmap='RdBu_r')
plt.colorbar()
plt.axis(aspect='image')
plt.xticks([])
plt.yticks([])
```




    ([], <a list of 0 Text yticklabel objects>)




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_19_1.png)



```python
np.mean(mgwr_bw1)
```




    234.168




```python
sns.distplot(mgwr_bw1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e83b4fc50>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_21_1.png)



```python
plt.imshow(B2, extent=[0,25, 0, 25], origin='lower',cmap='RdBu_r')
plt.colorbar()
plt.axis(aspect='image')
plt.xticks([])
plt.yticks([])
```




    ([], <a list of 0 Text yticklabel objects>)




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_22_1.png)



```python
np.mean(mgwr_bw2)
```




    502.406




```python
sns.distplot(mgwr_bw2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e834a5f60>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_24_1.png)



```python
np.mean(mgwr_bw0),np.mean(mgwr_bw1),np.mean(mgwr_bw2)
```




    (532.386, 234.168, 502.406)



### AIC, AICc, BIC check


```python
mgwr_aicc=[]
gwr_aicc=[]
mgwr_bic=[]
gwr_bic=[]
mgwr_aic=[]
gwr_aic=[]
mgwr_params=[]
gwr_params=[]
mgwr_predy=[]
gwr_predy=[]
```


```python
for i in range(0,500,10):
    p1 = pickle.load( open( "C:/Users/msachde1/Downloads/Research/Development/mgwr/notebooks/Binomial_MC_script_we/pkls/results-{}-{}.pkl".format(str(i), str(i+10)),"rb") )
    for j in range(10):
        mgwr_aicc.append(p1[j].mgwr_aicc[0])
        gwr_aicc.append(p1[j].gwr_aicc[0])
        
        mgwr_bic.append(p1[j].mgwr_bic[0])
        gwr_bic.append(p1[j].gwr_bic[0])
        
        mgwr_aic.append(p1[j].mgwr_aic[0])
        gwr_aic.append(p1[j].gwr_aic[0])
        
        mgwr_params.append(p1[j].mgwr_params[0])
        gwr_params.append(p1[j].gwr_params[0])
        
        mgwr_predy.append(p1[j].mgwr_predy[0])
        gwr_predy.append(p1[j].gwr_predy[0])
```


```python
sns.distplot(np.mean(gwr_predy,axis=0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e83099898>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_29_1.png)



```python
sns.distplot(np.mean(mgwr_predy,axis=0))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e81caef28>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_30_1.png)



```python
sns.distplot(y)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x24e81e7e4a8>




![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_31_1.png)



```python
f, axes = plt.subplots(1, 2, figsize=(4, 4), sharex=True)
sns.despine(left=True)
sns.distplot(mgwr_bic,ax=axes[0])
sns.distplot(gwr_bic,ax=axes[1])

plt.setp(axes, yticks=[])
plt.tight_layout()
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_32_0.png)



```python
f, axes = plt.subplots(1, 2, figsize=(4, 4), sharex=True)
sns.despine(left=True)
sns.distplot(mgwr_aic,ax=axes[0])
sns.distplot(gwr_aic,ax=axes[1])

plt.setp(axes, yticks=[])
plt.tight_layout()
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_33_0.png)



```python
f, axes = plt.subplots(1, 2, figsize=(4, 4), sharex=True)
sns.despine(left=True)
sns.distplot(mgwr_aicc,ax=axes[0])
sns.distplot(gwr_aicc,ax=axes[1])

plt.setp(axes, yticks=[])
plt.tight_layout()
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_34_0.png)



```python
np.mean(mgwr_aicc), np.mean(gwr_aicc)
```




    (573.0542142681326, 564.1109593727111)




```python
np.mean(mgwr_aic), np.mean(gwr_aic)
```




    (573.0155189048916, 562.4075174365614)




```python
np.mean(mgwr_bic), np.mean(gwr_bic)
```




    (586.3381072542409, 657.4481261493207)



#### AIC, AICc, BIC Boxplots for comparison


```python
model=[]
model = ['gwr']*500
model2 = ['mgwr']*500
model=model+model2
```


```python
aic=[]
aic=gwr_aic
aic=aic+mgwr_aic
```


```python
aicc=[]
aicc=gwr_aicc
aicc=aicc+mgwr_aicc
```


```python
bic=[]
bic=gwr_bic
bic=bic+mgwr_bic
```


```python
d = {'aic':aic,'bic':bic,'aicc':aicc,'model':model}
```


```python
df=pd.DataFrame(data=d)
```


```python
sns.set(style="whitegrid")
ax = sns.boxplot(y=df['aic'],x=df['model'])
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_45_0.png)



```python
sns.set(style="whitegrid")
ax = sns.boxplot(y=df['aicc'],x=df['model'])
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_46_0.png)



```python
sns.set(style="whitegrid")
ax = sns.boxplot(y=df['bic'],x=df['model'])
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_47_0.png)


### Parameter comparison from MGWR and GWR


```python
mgwr_params_mean=np.mean(mgwr_params,axis=0)
gwr_params_mean=np.mean(gwr_params,axis=0)
```


```python
gwr_params_mean
```




    array([[0.31469142, 1.61932981, 0.24680036],
           [0.31376851, 1.62072329, 0.25249583],
           [0.31251579, 1.62234972, 0.25957388],
           ...,
           [0.29442015, 1.71479871, 0.64257558],
           [0.2947018 , 1.67897208, 0.64781585],
           [0.29479386, 1.64721497, 0.65266   ]])




```python
B0_mgwr=np.hsplit(mgwr_params_mean,3)[0]
B1_mgwr=np.hsplit(mgwr_params_mean,3)[1]
B2_mgwr=np.hsplit(mgwr_params_mean,3)[2]
```


```python
B0_gwr=np.hsplit(gwr_params_mean,3)[0]
B1_gwr=np.hsplit(gwr_params_mean,3)[1]
B2_gwr=np.hsplit(gwr_params_mean,3)[2]
```


```python
B0_mgwr=B0_mgwr.reshape(25,25)
B1_mgwr=B1_mgwr.reshape(25,25)
B2_mgwr=B2_mgwr.reshape(25,25)
```


```python
np.mean(B0_mgwr),np.mean(B0_gwr),np.mean(B0)
```




    (0.3162876373767867, 0.3000642017402459, 0.3)




```python
B0_gwr=B0_gwr.reshape(25,25)
B1_gwr=B1_gwr.reshape(25,25)
B2_gwr=B2_gwr.reshape(25,25)
```


```python
fig, (ax, ax2,ax3, cax) = plt.subplots(ncols=4,figsize=(10,6), 
                  gridspec_kw={"width_ratios":[1,1,1, 0.1],"height_ratios":[1]})
fig.subplots_adjust(wspace=0.3)
im = ax.imshow(B0, extent=[0,10, 0, 10], origin='lower',cmap='Blues')
ax.text(3, -2, 'Original B0')
im2  = ax2.imshow(B0_mgwr, extent=[0,10, 0, 10], origin='lower',cmap='Blues')
ax2.text(3, -2, 'MGWR B0')
im3 = ax3.imshow(B0_gwr, extent=[0,10, 0, 10], origin='lower',cmap='Blues')
ax3.text(3, -2, 'GWR B0')

divider = make_axes_locatable(ax3)

fig.colorbar(im, cax=cax)

ax.set_xticks([])
ax.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xticks([])
ax3.set_yticks([])

plt.tight_layout()
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_56_0.png)



```python
fig, (ax, ax2,ax3, cax) = plt.subplots(ncols=4,figsize=(10,6), 
                  gridspec_kw={"width_ratios":[1,1,1, 0.1],"height_ratios":[1]})
fig.subplots_adjust(wspace=0.3)
im = ax.imshow(B1, extent=[0,10, 0, 10], origin='lower',cmap='RdBu_r')
ax.text(3, -2, 'Original B1')
im2  = ax2.imshow(B1_mgwr, extent=[0,10, 0, 10], origin='lower',cmap='RdBu_r')
ax2.text(3, -2, 'MGWR B1')
im3 = ax3.imshow(B1_gwr, extent=[0,10, 0, 10], origin='lower',cmap='RdBu_r')
ax3.text(3, -2, 'GWR B1')

divider = make_axes_locatable(ax3)

fig.colorbar(im, cax=cax)

ax.set_xticks([])
ax.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xticks([])
ax3.set_yticks([])

plt.tight_layout()
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_57_0.png)



```python
fig, (ax, ax2,ax3, cax) = plt.subplots(ncols=4,figsize=(10,6), 
                  gridspec_kw={"width_ratios":[1,1,1, 0.1],"height_ratios":[1]})
fig.subplots_adjust(wspace=0.3)
im = ax.imshow(B2, extent=[0,10, 0, 10], origin='lower',cmap='RdBu_r')
ax.text(3, -2, 'Original B2')
im2  = ax2.imshow(B2_mgwr, extent=[0,10, 0, 10], origin='lower',cmap='RdBu_r')
ax2.text(3, -2, 'MGWR B2')
im3 = ax3.imshow(B2_gwr, extent=[0,10, 0, 10], origin='lower',cmap='RdBu_r')
ax3.text(3, -2, 'GWR B2')

divider = make_axes_locatable(ax3)

fig.colorbar(im, cax=cax)

ax.set_xticks([])
ax.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])

ax3.set_xticks([])
ax3.set_yticks([])

plt.tight_layout()
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_58_0.png)


### Comparing parameters (MGWR and GWR)

$RMSE_j$ = $\sqrt{1/n \sum{(\beta_j (u_i, v_i) - \hat{\beta}(u_i, v_i))^2}}$


```python
B0_g=np.hsplit(gwr_params_mean,3)[0]
B1_g=np.hsplit(gwr_params_mean,3)[1]
B2_g=np.hsplit(gwr_params_mean,3)[2]
```


```python
B0_m=np.hsplit(mgwr_params_mean,3)[0]
B1_m=np.hsplit(mgwr_params_mean,3)[1]
B2_m=np.hsplit(mgwr_params_mean,3)[2]
```


```python
b0 = B0.reshape(-1,1)
b1 = B1.reshape(-1,1)
b2 = B2.reshape(-1,1)
```

### $B_0$


```python
rmse_b0_m=[]
for i in range(100):
    rmse_b0_m.append(np.sqrt((np.sum((b0 - (np.hsplit(mgwr_params[i],3)[0]))**2))/625))
    
rmse_b0_g=[]
for i in range(100):
    rmse_b0_g.append(np.sqrt((np.sum((b0 - (np.hsplit(gwr_params[i],3)[0]))**2))/625))
```


```python
model=[]
model = ['gwr']*100
model2 = ['mgwr']*100
model=model+model2

rmse_b0 = rmse_b0_g+rmse_b0_m
d = {"model":model,"rmse_b0":rmse_b0}
df = pd.DataFrame(data=d)
```


```python
sns.set(style="whitegrid")
ax = sns.boxplot(y=df['rmse_b0'],x=df['model'])
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_67_0.png)


### $B_1$


```python
rmse_b1_m=[]
for i in range(100):
    rmse_b1_m.append(np.sqrt((np.sum((b1 - (np.hsplit(mgwr_params[i],3)[1]))**2))/625))
```


```python
rmse_b1_g=[]
for i in range(100):
    rmse_b1_g.append(np.sqrt((np.sum((b1 - (np.hsplit(gwr_params[i],3)[1]))**2))/625))
```


```python
model=[]
model = ['gwr']*100
model2 = ['mgwr']*100
model=model+model2
rmse_b1=[]
rmse_b1 = rmse_b1_g+rmse_b1_m
d = {"model":model,"rmse_b1":rmse_b1}
df = pd.DataFrame(data=d)
```


```python
sns.set(style="whitegrid")
ax = sns.boxplot(x=df['model'],y=df['rmse_b1'])
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_72_0.png)



```python
np.sqrt((np.sum((b1 - (np.hsplit(mgwr_params_mean,3)[1]))**2))/625)
```




    1.8237125254924693




```python
np.sqrt((np.sum((b1 - (np.hsplit(gwr_params_mean,3)[1]))**2))/625)
```




    1.5244657525329972



### $B_2$


```python
rmse_b2_m=[]
for i in range(100):
    rmse_b2_m.append(np.sqrt((np.sum((b2 - np.hsplit(mgwr_params[i],3)[2])**2))/625))
    
rmse_b2_g=[]
for i in range(100):
    rmse_b2_g.append(np.sqrt((np.sum((b2 - np.hsplit(gwr_params[i],3)[2])**2))/625))
```


```python
model=[]
model = ['gwr']*100
model2 = ['mgwr']*100
model=model+model2
rmse_b2=[]
rmse_b2 = rmse_b2_g+rmse_b2_m
d = {"model":model,"rmse_b2":rmse_b2}
df = pd.DataFrame(data=d)
```


```python
sns.set(style="whitegrid")
ax = sns.boxplot(y=df['rmse_b2'],x=df['model'])
```


![png](Binomial_MGWR_MonteCarlo_Results-we_files/Binomial_MGWR_MonteCarlo_Results-we_78_0.png)



```python
np.sqrt((np.sum((b2 - (np.hsplit(mgwr_params_mean,3)[2]))**2))/625)
```




    0.10002993807213198




```python
np.sqrt((np.sum((b2 - (np.hsplit(gwr_params_mean,3)[2]))**2))/625)
```




    0.07791108751758576


