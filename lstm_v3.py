import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset = pd.read_csv('./AAPL.csv',index_col="Date")

sc = MinMaxScaler(feature_range = (0, 1))
training_set=dataset[:2900]
test_set=dataset[2900:]
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

window_size=60

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(window_size, len(training_set)):
    X_train.append(training_set_scaled[i-window_size:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)




# Creating a data structure with 60 timesteps and 1 output
X_test = []
y_test = []
for i in range(window_size, len(test_set)):
    X_test.append(test_set_scaled[i-window_size:i, 0])
    y_test.append(test_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(np.repeat(predicted_stock_price,5,axis=1))[:,0]
np.savetxt('prediction.csv', predicted_stock_price, delimiter=',')
np.savetxt('real.csv', test_set.values[window_size:,0], delimiter=',')

print(np.mean(np.abs(predicted_stock_price-test_set.values[window_size:,0])/np.mean(test_set.values[window_size:,0])))



