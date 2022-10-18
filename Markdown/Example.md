
# <center> Hands-on Example with MGWR </center>

**Notebook Outline:**  
  
**An example of hedonic house price modeling using MGWR**
- [Introduction to the Dataset](#Introduction-to-the-Dataset)
- [Loading the dataset](#Loading-the-Dataset)
- [Spatial Weighting Kernels and other Options](#Spatial-Weighting-Kernels-and-other-Options) <br><br>

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_workshop_book/)


# Introduction to the Dataset

***

### Please use [this link](https://github.com/mehak-sachdeva/Nagoya_University_workshop_2022/blob/main/data/census_tracts_final.csv) to directly download a csv of the dataset

The raw data has been obtained from https://www.kaggle.com/harlfoxem/housesalesprediction.
The data are cleaned and aggregated to the census tract level for King county, WA to make it smaller in size and manageable for the workshop.

### Study area and aggregation


<img src="../images/aggregation.PNG">

### Aggregated dataset snapshot

<img src="../images/data_structure_1.PNG">

- **houses** - number of houses in the census tracts

**Dependent variables**

- **avg_price** - average house prices in the census tracts
        
        OR

- **ln_avg_price** - log-transformed average house prices in the census tracts

**Independent variables**

- **avg_tech** - average technology related jobs in the census tracts

- **avg_unemp** - average unemployment rate in the census tracts

- **avg_index** - average number of house with a view to the waterfront

- **avg_basement** - average basements in the houses in the census tracts

- **avg_water_dist** - average distance to nearest waterfronts from the houses in the census tracts

- **avg_sqft** - average square footage of living area in houses in the census tracts

- **avg_age** - average age of housing units in the census tracts

#### Dependent variable distribution

<img src="../images/data_hist.PNG">

#### Hence, we use the log-transformed dependent variable *ln_avg_price*

# Loading the Dataset

Open the [MGWR GUI software](https://sgsup.asu.edu/sparc/mgwr) on your desktop to follow along!



At this stage I am going to do the steps elaborated below live. The screenshots of the steps below will guide you if you need additional reference.

*Please note, the screenshots below are taken on a Windows system. If you are using the software on Mac, the interface might differ slightly.*

### 1. Loading the dataset and variables

<img src="../images/step_1.png" width="700">

### 2. Load the dependent and independent variables

<img src="../images/step_2.png" width="700">

# Spatial Weighting Kernels and other Options

<img src="../images/step_3.png" width="700">

<img src="../images/spatial_weights.PNG" width="700">

### Advanced options

<img src="../images/advanced.png" width="700">

[Next](http://mehak-sachdeva.github.io/MGWR_workshop_book/Html/Interpretation)

[Previous](http://mehak-sachdeva.github.io/MGWR_workshop_book/Html/Methodology)

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_workshop_book/)

***
