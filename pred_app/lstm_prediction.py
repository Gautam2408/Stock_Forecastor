
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import datetime


def lstm_prediction(se, stock_symbol):
	
	def fetch_stock_data(se, stock_symbol):
		"""fetch stock data"""
		if se == 'NSE': stock_symbol += ".NS" 
		return yf.download(stock_symbol, period="5y")


	"""LSTM model development"""
	
	og_df = fetch_stock_data(se, stock_symbol)
	todataframe = og_df.reset_index(inplace=False)['Close']
	print(len(todataframe))	
#----------------------------------minmax scaler--------------------------------------------
	scaler=MinMaxScaler(feature_range=(0,1))
	todataframe=scaler.fit_transform(np.array(todataframe).reshape(-1,1))
#--------------------------------to print the info of the http://127.0.0.1:8000/OG dataset------------------------------

	print("\n<-------------------------------------------------->")
	#print(todataframe)
	print("<-------------------------------------------------------------------->\n")
#--------------------------- split dataset into train and test---------------------------------

	training_size=int(len(todataframe)*0.75)
	test_size=len(todataframe)-training_size
	train_data,test_data=todataframe[0:training_size,:],todataframe[training_size:len(todataframe),:1]
#--------------convert data into timestep---------------------------------------
	def create_dataset(dataset, time_step=1):
		dataX, dataY = [], []
		for i in range(len(dataset)-time_step-1):
			a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
			dataX.append(a)
			dataY.append(dataset[i + time_step, 0])
		return np.array(dataX), np.array(dataY)
	# reshape into X=t,t+1,t+2,t+3 and Y=t+4
	time_step = 100
	X_train, y_train = create_dataset(train_data, time_step)
	X_test, ytest = create_dataset(test_data, time_step)
	
	
#----------------- reshape input to be [samples, time steps, features] which is required for LSTM------------
	X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#-------------------create stacked lstm model---------------------	
	model=Sequential()
	model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
	model.add(LSTM(50,return_sequences=True))
	model.add(LSTM(50))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error',optimizer='adam')
	print(model.summary())
	model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=64,verbose=1)
	
#------------------------- Lets Do the prediction and check performance metrics--------------------------
	train_predict=model.predict(X_train)
	test_predict=model.predict(X_test)
	
#---------------------------------Test & Train Data RMSE------------------------------
	print(math.sqrt(mean_squared_error(y_train,train_predict)))
	print(math.sqrt(mean_squared_error(ytest,test_predict)))
	
#------------------------------- to predict data for next 100 days
	x_input=test_data[len(test_data)-100:].reshape(1,-1)
	temp_input=list(x_input)
	temp_input=temp_input[0].tolist()
	# demonstrate prediction for next 100 days
	lst_output=[]
	n_steps=100
	i=0
	while(i<100):
		if(len(temp_input)>100):
			#print(temp_input)
			x_input=np.array(temp_input[1:])
			x_input=x_input.reshape(1,-1)
			x_input = x_input.reshape((1, n_steps, 1))
			#print(x_input)
			yhat = model.predict(x_input, verbose=0)
			print("{} day output {}".format(i,yhat))
			temp_input.extend(yhat[0].tolist())
			temp_input=temp_input[1:]
			#print(temp_input)
			lst_output.extend(yhat.tolist())
			i=i+1
		else:
			x_input = x_input.reshape((1, n_steps,1))
			yhat = model.predict(x_input, verbose=0)
			print(yhat[0])
			temp_input.extend(yhat[0].tolist())
			print(len(temp_input))
			lst_output.extend(yhat.tolist())
			i=i+1
	lst_output = scaler.inverse_transform(lst_output)

#--------------------------Combining og and predicted dataset for end result.---------------------
	datelist = pd.date_range(datetime.now().date(), periods=101)[1:]
	predicted_df = pd.DataFrame(lst_output, columns=['Close'], index=datelist)
	result_df = pd.concat([og_df, predicted_df])[['Close']]
	result_df = result_df.reset_index(inplace=False)
	
	result_df.columns = ['Date', 'Close']
	print(result_df)
	#to print the info of the END RESULT dataset
	print("\n<----------------------Info of the RESULT dataset---------------------->")
	print(result_df.info())
	print("<-------------------------77----------------------------------------------->\n")
	
	
	def get_json(df):
		""" Small function to serialise DataFrame dates as 'YYYY-MM-DD' in JSON """
		import json
		import datetime
		def convert_timestamp(item_date_object):
			if isinstance(item_date_object, (datetime.date, datetime.datetime)):
				return item_date_object.strftime("%Y-%m-%d")
		
		dict_ = df.to_dict(orient='records')

		return json.dumps(dict_, default=convert_timestamp)

	return get_json(result_df)
