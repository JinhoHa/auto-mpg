import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

################################
#         Read dataset         #
################################
dataset_path = 'auto-mpg.data'

column_names=["mpg","cylinders","displacement","horsepower","weight",
       "acceleration","model_year","origin"]
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

# horsepower has 6 missing values. To keep the problem simple, drop those rows.
raw_dataset = raw_dataset.dropna()

############################################
#        Estimation of Raw Dataset         #
############################################
# Split dataset into train and test set
features = raw_dataset.copy()
targets = features.pop('mpg')
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 2017147594)

# Create Regression Model
lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
lr_rsme = np.sqrt(np.mean((y_test - predictions)**2))
print('LinearRegression rsme: ', lr_rsme)
lr_score = lr.score(X_test, y_test)
print('Linear R2 Squared: ', lr_score)
print()

##########################################
#          Feature Engineering           #
##########################################

dataset = raw_dataset.copy()
my_rsme = lr_rsme
my_score = lr_score

############ Step 1 ################
# remove cylinders column from dataset
# add cylinder_n columns to dataset

print('Step1: cylinders')

cylinders = dataset.pop('cylinders')
dataset['cylinder_3'] = (cylinders == 3)*1.0
dataset['cylinder_4'] = (cylinders == 4)*1.0
dataset['cylinder_5'] = (cylinders == 5)*1.0
dataset['cylinder_6'] = (cylinders == 6)*1.0
dataset['cylinder_8'] = (cylinders == 8)*1.0

features = dataset.copy()
targets = features.pop('mpg')
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 2017147594)

previous_rsme = my_rsme
previous_score = my_score

lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
my_rsme = np.sqrt(np.mean((y_test - predictions)**2))
print('Linear rsme: ', lr_rsme)
print('My RSME: ', my_rsme)
print('RSME Compared to original: ', my_rsme - lr_rsme)
print('RSME Compared to previous: ', my_rsme - previous_rsme)
print()

my_score = lr.score(X_test, y_test)
print('Linear R2: ', lr_score)
print('My R2: ', my_score)
print('R2 Compared to original: ', my_score - lr_score)
print('R2 Compared to previous: ', my_score - previous_score)
print()

############# Step2 ##################
print('Step2: log-transformation')
# Add new columns with log-transformed values
for col in dataset.columns:
  if col == 'displacement' or col == 'horsepower' or col == 'weight' or col == 'acceleration':
    dataset['log_' + col] = np.log1p(dataset[col])
  else:
    next

features = dataset.copy()
targets = features.pop('mpg')
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 2017147594)

previous_rsme = my_rsme
previous_score = my_score

lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
my_rsme = np.sqrt(np.mean((y_test - predictions)**2))
print('Linear rsme: ', lr_rsme)
print('My RSME: ', my_rsme)
print('RSME Compared to original: ', my_rsme - lr_rsme)
print('RSME Compared to previous: ', my_rsme - previous_rsme)
print()

my_score = lr.score(X_test, y_test)
print('Linear R2: ', lr_score)
print('My R2: ', my_score)
print('R2 Compared to original: ', my_score - lr_score)
print('R2 Compared to previous: ', my_score - previous_score)
print()

############## Step 3 #################
print('Step3: sqrt-transformation')

# add new columns with sqaure root transformation
for col in dataset.columns:
  if col == 'displacement' or col == 'horsepower' or col == 'weight' or col == 'acceleration':
    dataset['sqrt_' + col] = np.sqrt(dataset[col])
  else:
    next

features = dataset.copy()
targets = features.pop('mpg')
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 2017147594)

previous_rsme = my_rsme
previous_score = my_score

lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
my_rsme = np.sqrt(np.mean((y_test - predictions)**2))
print('Linear rsme: ', lr_rsme)
print('My RSME: ', my_rsme)
print('RSME Compared to original: ', my_rsme - lr_rsme)
print('RSME Compared to previous: ', my_rsme - previous_rsme)
print()

my_score = lr.score(X_test, y_test)
print('Linear R2: ', lr_score)
print('My R2: ', my_score)
print('R2 Compared to original: ', my_score - lr_score)
print('R2 Compared to previous: ', my_score - previous_score)
print()

########### Step 4 ###############
print('Step 4: Group model_year')

# remove model_year column
model_year = dataset.pop('model_year')
# add model_year_n columns to dataset
dataset['model_year_1'] = (model_year < 73)*1.0
dataset['model_year_2'] = (model_year < 73)*(-1.0) + (model_year < 76)*1.0
dataset['model_year_3'] = (model_year < 76)*(-1.0) + (model_year < 79)*1.0
dataset['model_year_4'] = (model_year >= 79)*1.0

features = dataset.copy()
targets = features.pop('mpg')
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 2017147594)

previous_rsme = my_rsme
previous_score = my_score

lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
my_rsme = np.sqrt(np.mean((y_test - predictions)**2))
print('Linear rsme: ', lr_rsme)
print('My RSME: ', my_rsme)
print('RSME Compared to original: ', my_rsme - lr_rsme)
print('RSME Compared to previous: ', my_rsme - previous_rsme)
print()

my_score = lr.score(X_test, y_test)
print('Linear R2: ', lr_score)
print('My R2: ', my_score)
print('R2 Compared to original: ', my_score - lr_score)
print('R2 Compared to previous: ', my_score - previous_score)
print()

#################### Step 5 ##################3
print('Step 5: Standardization')

features = dataset.copy()
targets = features.pop('mpg')
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 2017147594)

scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

previous_rsme = my_rsme
previous_score = my_score

lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
my_rsme = np.sqrt(np.mean((y_test - predictions)**2))
print('Linear rsme: ', lr_rsme)
print('My RSME: ', my_rsme)
print('RSME Compared to original: ', my_rsme - lr_rsme)
print('RSME Compared to previous: ', my_rsme - previous_rsme)
print()

my_score = lr.score(X_test, y_test)
print('Linear R2: ', lr_score)
print('My R2: ', my_score)
print('R2 Compared to original: ', my_score - lr_score)
print('R2 Compared to previous: ', my_score - previous_score)