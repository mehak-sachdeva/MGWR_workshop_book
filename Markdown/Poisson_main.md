
# <center>Multiscale Geographically Weighted Regression - Poisson dependent variable</center>


The model has been explored and tested for multiple parameters on real and simulated datasets. The research includes the following outline with separate notebooks for each part.


**Notebook Outline:**  
  
**Introduction (current)**
- [Introduction](#Introduction)
 - [Introduction to the problem](#Introduction-to-the-project)
 - [Statistical Equations](#Statistical-Equations) 
- [Approaches Explored](#Approaches-Explored)
- [Other notebooks](#Other-notebooks)
- [References](#References)

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_book/)

---

# Introduction

***

## Introduction to the problem

A recent addition to the local statistical models in PySAL is the implementation of Multiscale Geographically Weighted Regression (MGWR) model, a multiscale extension to the widely used approach for modeling process spatial heterogeneity - Geographically Weighted Regression (GWR). GWR is a local spatial multivariate statistical modeling technique embedded within the regression framework that is calibrated and estimates covariate parameters at each location using borrowed data from neighboring observations. The extent of neighboring observations used for calibration is interpreted as the indicator of scale for the spatial processes and is assumed to be constant across covariates in GWR. MGWR, using a back-fitting algorithm relaxes the assumption that all processes being modeled operate at the same spatial scale and estimates a unique indicator of scale for each process.
The GWR model in PySAL can currently estimate Gaussian, Poisson and Logistic models though the MGWR model is currently limited to only Gaussian models. This project aims to expand the MGWR model to nonlinear local spatial regression modeling techniques where the response outcomes may be discrete (following a Poisson distribution). This will enable a richer and holistic local statistical modeling framework to model multi-scale process heterogeneity for the open source community.

## Statistical Equations

A conventional Poisson regression model is written as:

\begin{align}
O_i {\sim} Poisson[E_i exp ({\sum} {\beta} & _k x _{k,i})] \\
\end{align}

where  $x_{k,1}$ is the kth explanatory variable in place i and the ${\beta}_ks$ are the parameters and Poisson indicates a Poisson distribution with mean $\lambda$.

Nakaya et.al. (2005) introduced the concept of allowing parameter values to vary with geographical location ($u_i$), which is a vector of two dimensional co-ordinates describing the location *i*. The Poisson model for geographically varying parameters can be written as:

\begin{align}
O_i {\sim} Poisson[E_i exp ({\sum} {\beta} & _k (u_i) x _{k,i})] \\
\end{align}

The Geographically Weighted Poisson Regression model (GWPR) is estimated using a modified local Fisher scoring procedure, a form of iteratively reweighted least squares (IRLS). In this procedure, the following matrix computation of weighted least squares should be repeated to update parameter estimates until they converge (Nakaya et.al., 2005):

\begin{align}
\beta^{(l+1)} (u_i) = (X^{t} W (u_i) A(u_i)^{(l)} X)^{-1} X^{t} W (u_i) A (u_i) ^{(l)} z (u_i){(l)} \\
\end{align}

# Approaches Explored

**Expected theoretical model calibration:**<br><br>
The calibration methodology for modeling response variables with a Poisson distribution in MGWR, through references from Geographically Weighted Poisson Regression (GWPR) (Nakaya, et al., 2005) and literature on Generalized Additive Models (Hastie & Tibshirani, 1986), is expected to be as follows:<br>


1. Initialize using GWPR estimates $𝛽_𝑘(𝑢_𝑖)^0: f^1_0,f^2_0, … , f^𝐾_0, $ 
    where $f_𝑘^0 = (𝑥_{1 𝑘}𝛽_𝑘(𝑢_1)^0 … 𝑥_{𝑁𝑘}𝛽_𝑘(𝑢_𝑁)^0)$<br><br>
2. Update for each location ($𝑢_𝑖$) an adjusted dependent variable: <br>
    $z(𝑢_𝑖)^{(𝑙)} = (z_1(𝑢_𝑖)^{(𝑙)}, z_2(𝑢_𝑖)^{(𝑙)}, … ,z_𝑁(𝑢_𝑖)^{(𝑙)})^𝑡$<br> 
    $z_𝑗(𝑢_𝑖)^{(𝑙)} = Σ𝛽_𝑘(𝑢_𝑖)^{(𝑙)} 𝑥_{𝑗,𝑘} + O_𝑗 − Ô_𝑗(𝛽(𝑢_𝑖)^{(𝑙)}) / Ô_𝑗(𝛽(𝑢_𝑖)^{(𝑙)})$<br><br>
3. Construct weights as follows:
    $A(𝑢_𝑖)^{(𝑙)}$ which is a diagonal matrix with values ($Ô_1(𝛽(
𝑢_𝑖)^{(𝑙)}), Ô_2(𝛽(
𝑢_𝑖)^{(𝑙)}), ... ,Ô_𝑁(𝛽(𝑢_𝑖)^{(𝑙)}))$<br><br>
4. Fit an MGWR model to $z(𝑢_𝑖)^{(𝑙)}$ to update $𝛽_𝑘(𝑢_𝑖)$ and $z(𝑢_𝑖)^{(𝑙)}$, using the new weight matrix:<br>
$𝑊_𝑘∗(𝑢_𝑖)^{(𝑙)}=𝑊_𝑘(𝑢_𝑖)^{(𝑙)} A(𝑢_𝑖)^{(𝑙)}$<br><br>
5. Repeat steps (2) through (4) until convergence.

# Other notebooks

Below are links to the tests and exploration for the finalized MGWR model for Poisson dependent variables:

## Tests with real and simulated data
***

**[Initial module changes and univariate model check ](http://mehak-sachdeva.github.io/MGWR_book/Html/Poisson_MGWR_univariate_check)**
- Setup with libraries
- Fundamental equations for Poisson MGWR
- Example Dataset
- Helper functions
- Univariate example
    - Parameter check
    - Bandwidths check

**[Simulated Data example](http://mehak-sachdeva.github.io/MGWR_book/Html/Simulated_data_example_Poisson-MGWR)**
- Setup with libraries
- Create Simulated Dataset
    - Forming independent variables
    - Creating y variable with Poisson distribution
- Univariate example
    - Bandwidth: Random initialization check
    - Parameters check
- Multivariate example
    - Bandwidths: Random initialization check
    - Parameters check
- Global model parameter check
 
**[Real Data example](http://mehak-sachdeva.github.io/MGWR_book/Html/Real_data_example_Poisson-MGWR)**

- Setup with libraries
- Tokyo Mortality Dataset
- Univariate example
    - Bandwidth: Random initialization check
    - Parameter check
- Multivariate example
    Bandwidths: Random initialization check
- MGWR bandwidths
- AIC, AICc, BIC check


## Monte Carlo Test
***


**[Monte Carlo Simulation Visualization](http://mehak-sachdeva.github.io/MGWR_book/Html/Poisson_MGWR_MonteCarlo_Results)**
 
- Setup with libraries
- List bandwidths from pickles
- Parameter functions
- GWR bandwidth
- MGWR bandwidths
- AIC, AICc, BIC check
    - AIC, AICc, BIC Boxplots for comparison
- Parameter comparison from MGWR and GWR

# References

1. Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale Geographically Weighted Regression (MGWR). Annals of the American Association of Geographers, 107(6), 1247–1265. https://doi.org/10.1080/24694452.2017.1352480


2. Nakaya, T., Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2005). Geographically weighted Poisson regression for disease association mapping. Statistics in Medicine, 24(17), 2695–2717. https://doi.org/10.1002/sim.2129


3. Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W., & Wolf, L. J. (2019). Inference in Multiscale Geographically Weighted Regression. Geographical Analysis, gean.12189. https://doi.org/10.1111/gean.12189


4. Hastie, T., & Tibshirani, R. (1986). Generalized Additive Models. Statistical Science, 1(3), 297–310. https://doi.org/10.1214/ss/1177013604


5. Wood, S. N. (2006). Generalized additive models : an introduction with R. Chapman & Hall/CRC.

[Back to the main page](https://mehak-sachdeva.github.io/MGWR_book/)
