from django.shortcuts import render, redirect
from django.contrib import messages, auth
from pred_app.lstm_prediction import *

# ------------------ MAIN WEB PAGES ---------------------------
def redirect_root(request):
    return redirect('/pred_app/index')

def index(request):
	return render(request, 'pred_app/index.html') 

def pred(request):
    return render(request, 'pred_app/prediction.html')

def contact(request):
	return render(request, 'pred_app/contact.html')

def chart(request):
	import yfinance as yf
	import pandas as pd
	volume=[]
	goog=[]
	Dates=[]
	close=[]	
	nyse_tickers=['TSLA','AMZN','FB','MSFT','AAPL']
	nse_tickers=['RELIANCE.NS','TATAMOTORS.NS','INFY.NS']
	column=['Adj Close','Open','Close','High','Low']
	df_nyse=yf.download(nyse_tickers,period='1y')
	df_nyse.reset_index(inplace=True)
	df_nse=yf.download(nse_tickers,period='1y')
	df_nse.reset_index(inplace=True)
	df_goog=yf.download('GOOG',period='3mo')
	df_goog.reset_index(inplace=True)
	for col in column :
		df_nyse[col]=df_nyse[col].applymap('{:.4f}'.format)
	for col in column :
		df_nse[col]=df_nse[col].applymap('{:.4f}'.format)	
	for col in column :	
		df_goog[col]=df_goog[col].apply('{:.5f}'.format)
	df_nse.to_csv('df_nse.csv')
	df_goog.to_csv('df_goog.csv')
	df1=pd.read_csv('df_nse.csv')
	df2=pd.read_csv('df_goog.csv')
	Dates.append(df1['Date'].tolist())
	Dates.append(df2['Date'].tolist())
	Dates[0]=Dates[0][1:]
	start_date=Dates[1][0]
	last_date=Dates[1][-1]
	max_goog=df_goog['High'].max()
	min_goog=df_goog['Low'].min()
	for ticker in nyse_tickers :
		volume.append(format(df_nyse['Volume'][ticker].sum()/len(df_nyse['Volume'][ticker]),'.2f'))
	close.append(df_nse['Close']['RELIANCE.NS'].tolist())
	close.append(df_nse['Close']['TATAMOTORS.NS'].tolist())
	close.append(df_nse['Close']['INFY.NS'].tolist())
	#candlestick
	goog.append(df_goog['Close'].tolist())
	goog.append(df_goog['Open'].tolist())
	goog.append(df_goog['High'].tolist())
	goog.append(df_goog['Low'].tolist())
		
	return render(request, 'pred_app/chart.html',{
		'volume' 	   :	volume,
		'labels'	   :	nyse_tickers,
		'close_rel'    :	close[0],
		'close_tata'   :	close[1],
		'close_infy'   :	close[2],
		'labels_line'  :	Dates[0],
		'goog_close'   :	goog[0],
		'goog_open'    :	goog[1],
		'goog_high'    : 	goog[2],
		'goog_low'     :  	goog[3],
		'label_candle' : 	Dates[1],
		'start_date'   :	start_date,
		'last_date'    :	last_date,
		'max'		   :	max_goog,
		'min'	       :	min_goog
		})


def search(request, se, stock_symbol):
	import json
	print("hiii",se,stock_symbol)
	predicted_result_df = lstm_prediction(se, stock_symbol)
	if se=="NYSE" :
		se1="NASDAQ"
	else :
		se1=se
	return render(request, 'pred_app/search.html', {
		"predicted_result_df": predicted_result_df,
		"se"			: se1,
		"stock_symbol"	: stock_symbol
	})
# -----------------------------------------------------------
def search1(request, se, stock_symbol):
	if se=="NYSE" :
		se="NASDAQ"
	return render(request, 'pred_app/prediction.html', {
		"se"			: se,
		"stock_symbol"	: stock_symbol
	})
# -----------------------------------------------------------