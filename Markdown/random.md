

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df=pd.read_csv("example_dataset.csv")
```


```python
plt.hist(df['avg_price'],color="orange")
```

    C:\Users\msachde1\AppData\Roaming\Python\Python36\site-packages\numpy\lib\histograms.py:829: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    C:\Users\msachde1\AppData\Roaming\Python\Python36\site-packages\numpy\lib\histograms.py:830: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)
    




    (array([184., 123.,  31.,  15.,   1.,   1.,   1.,   0.,   0.,   1.]),
     array([ 200442.86 ,  480398.574,  760354.288, 1040310.002, 1320265.716,
            1600221.43 , 1880177.144, 2160132.858, 2440088.572, 2720044.286,
            3000000.   ]),
     <a list of 10 Patch objects>)




![png](random_files/random_2_2.png)



```python

```
