
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.metrics


dataset=pd.read_csv('sme_loan.csv')


dataset['loan_date']=pd.to_datetime(dataset['loan_date'])


dataset['month']=dataset.loan_date.dt.month


dataset['year']=dataset.loan_date.dt.year


featureSet=dataset.copy()
target=dataset['default_prob']
del featureSet['default_prob']
del featureSet['business_name']
del featureSet['loan_date']
del featureSet['Unnamed: 0']


#LabelEncoding
encoder=LabelEncoder()
featureSet.industry=encoder.fit_transform(featureSet.industry)
featureSet.funding_type=encoder.fit_transform(featureSet.funding_type)
featureSet.state=encoder.fit_transform(featureSet.state)


#Splitting the dataset into training and testing set
train_x,test_x,train_y,test_y=train_test_split(featureSet,target,test_size=0.25,random_state=46)


#Lasso
lasso_model = make_pipeline(StandardScaler(with_mean=False), Lasso())
lasso_model.fit(train_x,train_y)
lasso_predicted_y=lasso_model.predict(test_x)

#Random_Forest model
rf_model=RandomForestRegressor(n_estimators=1000,random_state=42)
rf_model.fit(train_x,train_y)
rf_predicted_y=rf_model.predict(test_x)


#Lasso MSE
sklearn.metrics.mean_squared_error(test_y,lasso_predicted_y)


#Random_Forest MSE
sklearn.metrics.mean_squared_error(test_y,rf_predicted_y)
