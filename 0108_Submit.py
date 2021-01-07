import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

def IsCanceledPredict():
	
	# num_features = ["lead_time", "total_of_special_requests", "previous_cancellations", "agent"]
	# cat_features = ["customer_type"]

	num_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests']
	cat_features = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent', 'company', 'customer_type']

	features = num_features + cat_features
	X = x_train_all.drop(["is_canceled"], axis=1)[features]
	y = x_train_all["is_canceled"]
	test_x = x_test_all[features]
	test_y = x_test_all

	num_transformer = SimpleImputer(strategy="constant")

	cat_transformer = Pipeline(steps=[
	    # ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
	    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

	preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
	                                               ("cat", cat_transformer, cat_features)])

	# clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=2000, random_state=42,n_jobs=-1))])
	clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(use_label_encoder=False))])

	clf.fit(X, y)

	x_test_all['is_canceled'] = clf.predict(test_x)

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

def AdrPredict():

	global x_train_all

	x_train_all['arrival_date'] = x_train_all['arrival_date_year'].astype(str) + '-' + pd.to_datetime(x_train_all['arrival_date_month'], format='%B').dt.month.astype(str).apply(lambda x: x.zfill(2)) + '-' + x_train_all['arrival_date_day_of_month'].astype(str).apply(lambda x: x.zfill(2))

	# noencode_features = ['lead_time']
	# encode_features = ['hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
	# 				   'stays_in_weekend_nights', 'stays_in_week_nights', 
 #                       'adults', 'children', 'babies', 'meal', 
 #                       'country', 'market_segment', 'distribution_channel', 'is_repeated_guest',
 #                       'previous_cancellations', 'previous_bookings_not_canceled', 
 #                       'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
 #                       'agent', 'company', 'days_in_waiting_list', 'customer_type', 'required_car_parking_spaces', 
 #                       'total_of_special_requests']

	noencode_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests']
	encode_features = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent', 'company', 'customer_type']

	features = noencode_features + encode_features

	x_train = x_train_all[features].copy()
	adr = x_train_all['adr']

	num_transformer = SimpleImputer(strategy="constant")
	cat_transformer = OneHotEncoder(handle_unknown='ignore')
	preprocessor = ColumnTransformer(transformers=[("num", num_transformer, noencode_features),
	                                               ("cat", cat_transformer, encode_features)])

	x_train = preprocessor.fit_transform(x_train)

	model = XGBRegressor()
	model.fit(x_train, adr)

	x_test = x_test_all[features].copy()
	x_test = preprocessor.transform(x_test)
	x_test_all['predict_adr'] = model.predict(x_test)

import numpy as np

def LabelPredict():

	x_test_all['arrival_date'] = x_test_all['arrival_date_year'].astype(str) + '-' + pd.to_datetime(x_test_all['arrival_date_month'], format='%B').dt.month.astype(str).apply(lambda x: x.zfill(2)) + '-' + x_test_all['arrival_date_day_of_month'].astype(str).apply(lambda x: x.zfill(2))

	x_test_not_canceled = x_test_all[x_test_all['is_canceled'] == False].copy()
	x_test_not_canceled['revenue'] = x_test_not_canceled['predict_adr']*(x_test_not_canceled['stays_in_week_nights']+x_test_not_canceled['stays_in_weekend_nights'])
	day_revenue = np.floor((x_test_not_canceled.groupby('arrival_date')['revenue'].sum())/10000)

	y_test = pd.read_csv("./test_nolabel.csv")
	y_test = y_test.merge(day_revenue, on='arrival_date')
	y_test = y_test.rename(columns={"revenue": "label"})
	y_test.to_csv('baseline_0108_01.csv', index=False)

if __name__ == "__main__":

	x_train_all = pd.read_csv("./train.csv")
	x_test_all = pd.read_csv("./test.csv")

	IsCanceledPredict()
	AdrPredict()
	LabelPredict()
