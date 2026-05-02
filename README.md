## Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
## Date: 02.05.2026
## NAME: MOHAMED NIZAMUDDIN A
## REG NO: 212224040194


## AIM:
To implement ARMA model in python.
## TOOLS USED: 
 GOOGLE COLAB SOFTWARE
## ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load YOUR dataset
data = pd.read_csv("DailyDelhiClimateTrain.csv")

# Use humidity column
X = data['humidity']

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Original Data
plt.plot(X)
plt.title('Original Humidity Data')
plt.show()

# ACF & PACF of original data
plt.subplot(2,1,1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2,1,2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1)
arma11_model = ARIMA(X, order=(1,0,1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# ARMA(2,2)
arma22_model = ARIMA(X, order=(2,0,2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()




```
## OUTPUT:
## ORIGINAL DATA :
<img width="986" height="528" alt="image" src="https://github.com/user-attachments/assets/a3fdcc6e-3144-4754-bd27-91430e71ef95" />
## ORIGINAL DATA ACF:
<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/c0893b40-cb9c-416d-bfd1-9b9be764c1a6" />
## ORIGINAL DATA PACF:
<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/43818968-a485-494a-acbe-559b2484f131" />

## SIMULATED ARMA(1,1) PROCESS:
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/96c6754f-0276-4c15-b467-5301e7c25fc4" />



## Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/6551d863-6952-448a-95ef-a143f5a06f18" />

## Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/e238fbd6-6cf3-47a3-a3eb-69cda2efb375" />



## SIMULATED ARMA(2,2) PROCESS:
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/1d4882af-fe89-417f-8741-ab0fabbafed6" />

 ## Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/e5ca6e43-b7dd-4f4f-add4-848a56e3e6a6" />


## Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/5f7947b3-3a55-47bf-888e-e4e579e558e5" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
