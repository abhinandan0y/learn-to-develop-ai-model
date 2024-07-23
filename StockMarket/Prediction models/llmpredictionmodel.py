#Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Enable inline plotting in Jupyter Notebooks
%matplotlib inline


# Enable inline plotting in Jupyter Notebooks
%matplotlib inline

#Get the Dataset
df=pd.read_csv(“MicrosoftStockData.csv”,na_values=[‘null’],index_col=’Date’,parse_dates=True,infer_datetime_format=True)
df.head()
#Print the shape of Dataframe  and Check for Null Values
print(“Dataframe Shape: “, df. shape)
print(“Null Value Present: “, df.IsNull().values.any())
Output:
#>> Dataframe Shape: (7334, 6)
#>>Null Value Present: False

#Plot the True Adj Close Value
df[‘Adj Close’].plot()























