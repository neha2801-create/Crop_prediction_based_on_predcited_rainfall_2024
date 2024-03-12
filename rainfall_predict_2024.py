#############       PREDICTING 2024 RAINFALL LEVELS     ##########################
## All important packages given below############################################
import pandas as pd
from sklearn.model.selection import train_test_split, GridSearchCV      #  Model 1 for rainfall prediction
from sklearn.ensemble import RandomForestRegressor                      #  Model used for enhancing performance of Model 1
from sklearn.preprocessing import StandardScaler                        #  For scaling data used
from sklearn.model_selection import train_test_split                    #  Spliting training and test data
from sklearn.ensemble import RandomForestClassifier                     #  Model 2 for crop label prediction
from sklearn.metrics import accuracy_score                              #  For getting accuracy score
import matplotlib.pyplot as plt                                         #  For comparison plot for all input years and predicted year rainfall

###################################      Loading 1st csv file ########################################
################ Useful Parameters in input csv file (column names) : datetime, precip ( rainfall level in mm for eevryday) ###############
dataset_file = 'India.csv' 
# Now converting it in dataframe
dataset_df = pd.read_csv(dataset_file)

#################################    Data exploration Begins  ###################################################
#  Converting 'datetime' column to datetime type data
dataset_df['datetime'] = pd.to_datetime(dataset_df['datetime'])

# Extracting year value from datetime type of data
dataset_df['year'] = dataset_df['datetime'].dt.year

print('Exploratory Data Analysis')
print('Precipitation over various months through the years 2018 - 2023')

# Creating respective year precipitation/raainfall level subplots subplots
plt.figure(figsize=(22, 8))
for year in dataset_df['year'].unique():
    plt.subplot(2, 3, year-2018+1)  # Assuming 2018-2023
    subset = dataset_df[dataset_df['year'] == year]
    plt.plot(subset['datetime'], subset['precip'])
    plt.title(f'Precipitation in {year}')
    plt.xlabel('Datetime')
    plt.ylabel('Precipitation')

plt.tight_layout()
plt.show()

# Computing correlation matrix for demonstrating heat map for comparitive study among all column values/ parameter/ feature values
correlation_matrix = dataset_df.corr(numeric_only=True)

# Ploting heat map using correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Plotting Pair wise plot ( displaying relationship among column values/ parameter/ feature values)
pair_plot_data = dataset_df[['precip', 'name', 'datetime', 'precipprob']]  # different column names as per the input India.csv file
# Create further pair plots
sns.pairplot(pair_plot_data)
plt.show()

# Plot histogram of target variable : Precipitation/ Rainfall levels (in mm)
plt.figure(figsize=(8, 6))
sns.histplot(dataset_df['precip'], kde=True)
plt.title('Distribution of Precipitation (Histogram)')
plt.xlabel('Precipitation')
plt.ylabel('Frequency')
plt.show()

#################################    Data exploration Ends  ##################################################################

####################################    Filtering data based on column names further getting India specific data   ############

india_data = dataset_df[dataset_df['name'] == 'India']

################################        Converting datetime column value into datetime type value ###########################
india_data['datetime'] = pd.to_datetime(india_data['datetime'])

##############################   Extracting feature and target variables################################################
X_train = india_data[['datetime']]
y_train = india_data['precip']

#############################  Extract relevant features from the date using .loc since ML model cannot understand datetime in raw format  ###########
X_train = X_train.copy()
X_train.loc[:, 'day_of_year'] = X_train['datetime'].dt.dayofyear

# Droping the original datetime column since we already have understandable format of date for ML model to interpret
X_train = X_train.drop(['datetime'], axis=1)

############################   Training model : Random Forest Regressor on given data ######################################################
regression_model = RandomForestRegressor(random_state=42)

#  Feature Engineering - Scaling numerical features for using data inside model ( though scaling in not required in Random Forest Regressor) for better results
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

############################  Using grid_search_regression for tuning Hyperparameters of Model : Random Forest Regressor for better results #################

# Following 3 parameters of regression_model is tuned using grid_search_regression for getting accurate results since it uses grid search method for better selection
# Performing a grid search for hyperparameter tuning
param_grid_regression = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_regression = GridSearchCV(regression_model, param_grid_regression, cv=5)
grid_search_regression.fit(X_train_scaled, y_train)

best_regression_model = grid_search_regression.best_estimator_

#########################  Predicting 2024 precipitation/rainfall levels using best_regression_model obtaibed above ###################################################
# Preparing a new dataset for India in 2024 with the entire calendar year
india_2024_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
india_2024_data = pd.DataFrame({'datetime': india_2024_dates})
india_2024_data['day_of_year'] = india_2024_data['datetime'].dt.dayofyear
india_2024_data = india_2024_data.drop(['datetime'], axis=1)

# Using the scaler to transform the features
india_2024_data_scaled = scaler.transform(india_2024_data)

# Now predicting the precipitation for India in 2024
predicted_precipitation_2024 = best_regression_model.predict(india_2024_data_scaled)

# Creating a DataFrame with the predicted precipitation values and corresponding dates for 2024 year
predicted_precipitation_df = pd.DataFrame({
    'datetime': india_2024_dates,
    'predicted_precipitation': predicted_precipitation_2024
})

# Filtering rows where 'predicted_precipitation' is greater than 0.0 for 2024 
filtered_precipitation_df = predicted_precipitation_df[predicted_precipitation_df['predicted_precipitation'] > 0.0]

print("Filtered Predicted Precipitation for India in 2024 (where precipitation > 0.0):")
print(filtered_precipitation_df)

####################### Prediction for 2024 ends ########################################################################

######## Since we have to test on future dates and actual data cannot be obtained so we are using 2023 data with actual and predicted data for comparison ##################

#####################  2023 data analysis begins for rating model efficiency ##################################################
india_2023_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
india_2023_data_x = pd.DataFrame({'datetime': india_2023_dates})
india_2023_data_x['day_of_year'] = india_2023_data_x['datetime'].dt.dayofyear
india_2023_data_x = india_2023_data_x.drop(['datetime'], axis=1)

# Use the scaler to transform the features
india_2023_data_x_scaled = scaler.transform(india_2023_data_x)

# Assuming 'precip_class' is a binary column indicating precipitation occurrence (1 for precipitation, 0 for no precipitation)
# If your dataset has a different column for precipitation occurrence, adjust accordingly
india_data['precip_class'] = (india_data['precip'] > 0).astype(int)

# Extract features and target variable for regression
X_train_regression = india_data[['datetime']]
y_train_regression = india_data['precip']

# Extracting relevant features from the date using .loc
X_train_regression = X_train_regression.copy()
X_train_regression.loc[:, 'day_of_year'] = X_train_regression['datetime'].dt.dayofyear

# Droping the original datetime column for regression since we already extracted day from datetime for making ML model understand values in column
X_train_regression = X_train_regression.drop(['datetime'], axis=1)

# Training a RandomForestRegressor for rainfall prediction
regression_model_2023 = RandomForestRegressor(random_state=42)

# Feature Engineering - Scaling numerical features for regression
scaler_regression = StandardScaler()
#X_train_regression_scaled = scaler_regression.fit_transform(X_train_regression)

# Performing a grid search for hyperparameter tuning for regression
param_grid_regression = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search_regression_2023 = GridSearchCV(regression_model_2023, param_grid_regression, cv=5)
grid_search_regression_2023.fit(X_train_regression.values, y_train_regression)
best_regression_model_2023 = grid_search_regression.best_estimator_

# using train_test_split
X_test_class_2023, _,y_test_class_2023, _ = train_test_split(X_train_regression, y_train_regression, test_size=0.2, random_state=42)

# Predicting precipitation occurrence for the test set
y_pred_class_2023 = best_regression_model_2023.predict(X_test_class_2023)

# Evaluating regression model performance
mae = mean_absolute_error(y_test_class_2023, y_pred_class_2023)
mse = mean_squared_error(y_test_class_2023, y_pred_class_2023)
r2 = r2_score(y_test_class_2023, y_pred_class_2023)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2) Score: {r2}')

#################################### Model evaluation ends using 2023 data ###################################################################################

###################################  Displaying combined plot for all years from 2018 to 2023 and predicted 2024 precipiation/ rainfall level #################

# Plotting the predicted precipitation for each year
plt.figure(figsize=(30, 10))

# 2018 year plot 
india_data_2018 = india_data[india_data['datetime'].dt.year == 2018]
# Plot for 2018
plt.plot(india_data_2018['datetime'], india_data_2018['precip'], label='Actual Rainfall (2018)')

# 2019 year plot
india_data_2019 = india_data[india_data['datetime'].dt.year == 2019]
# Plot for 2019
plt.plot(india_data_2019['datetime'], india_data_2019['precip'], label='Actual Rainfall (2019)')

# 2020 year plot 
india_data_2020 = india_data[india_data['datetime'].dt.year == 2020]
# Plot for 2020
plt.plot(india_data_2020['datetime'], india_data_2020['precip'], label='Actual Rainfall (2020)')

# 2021 year plot
india_data_2021 = india_data[india_data['datetime'].dt.year == 2021]
# Plot for 2021
plt.plot(india_data_2021['datetime'], india_data_2021['precip'], label='Actual Rainfall (2021)')

# 2022 year plot
india_data_2022 = india_data[india_data['datetime'].dt.year == 2022]
# Plot for 2022
plt.plot(india_data_2022['datetime'], india_data_2022['precip'], label='Actual Rainfall (2022)')

# 2023 year plot
india_data_2023 = india_data[india_data['datetime'].dt.year == 2023]
# Plot for 2022
plt.plot(india_data_2023['datetime'], india_data_2023['precip'], label='Actual Rainfall (2023)')

# 2024 predicted data plot
plt.plot(predicted_precipitation_df['datetime'], predicted_precipitation_df['predicted_precipitation'], label='Predicted Rainfall (2024)')

plt.title('Predicted Precipitation for India')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)
plt.show()

######################## Combining entire year data and segregating them into 4 chuncks of 4 season since here data is India specific ############
# Converting 'Date' column to datetime format
predicted_precipitation_df['datetime'] = pd.to_datetime(predicted_precipitation_df['datetime'])

# Extracting month from the date
predicted_precipitation_df['Month'] = predicted_precipitation_df['datetime'].dt.month

# Maping month numbers to seasons
seasons_mapping = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

# Adding a new column 'Season' based on the 'Month'
predicted_precipitation_df['Season'] = predicted_precipitation_df['Month'].apply(lambda month: next(season for season, months in seasons_mapping.items() if month in months))

# Grouping by 'Season' and aggregate the rainfall
seasonal_rainfall = predicted_precipitation_df.groupby('Season')['predicted_precipitation'].sum().reset_index()

# Displaying the original DataFrame
print("Original DataFrame:")
print(predicted_precipitation_df)

# Displaying the seasonal rainfall
print("\nSeasonal Rainfall:")
print(seasonal_rainfall)

# Updating the DataFrame with the seasonal rainfall values
predicted_precipitation_df = pd.merge(predicted_precipitation_df, seasonal_rainfall, on='Season', suffixes=('', '_Seasonal'))

# Displaying the updated DataFrame
print("\nUpdated DataFrame:")
print(predicted_precipitation_df)
print("########################## PRINTING FINAL PREDICTED RAINFALL SEASON WISE FOR NEXT YEAR 2024  $$$$$$$$$$$$$$$$$$$$$$$")
print(seasonal_rainfall)

############################## Ends extracting data season wise ##############################################################

################################  Saving final predicted 2024 rainfall combined season wise in Csv file ######################

# Saving the DataFrame containing predicted rainfall level to CSV file named Predicted_2024.csv
print("Saving the DataFrame containing predicted rainfall level to Excel file named Predicted_2024.csv")
seasonal_rainfall.to_csv('Predicted_2024.csv', index=False)

#################################### MODEL 1 ENDS : With predicted rainfall levels for 4 seasons which is input for next model : MODEL 2 ###############################
''' Model 2 : Random Forest Classifier : Gets inputtesting data from Model 1 , predicted rainfall level for 2024 (segregated season wise) (THIS IS TEST DATA)
                                         Provides output : Predicting crop levels for entire year/ 4 respective seasons based on training data in csv format
                                         Training data used : Crop.csv which contains 2 columns : Rainfall level and Crop labels
                                         '''
#  Reading csv format file as input file
crop_df =pd.read_csv('crop_data.csv')

################################## Feature extraction for Model 2 ##################################################################
# label : Column in crop.csv contains names of all types of crop used as training data for model 2

X= crop_df.drop('label',axis = 1) 
y = crop_df['label']

# Spliting data into training and testing sets (80-20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

################################ Training Model 2 : Random Forest Classifier ######################################################

# Creating and training a Random Forest Classifier 
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting features in model Random Forest Classifier
clf.fit(X_train, y_train)

############################# Predicting labels for TESTING DATA : obtained spliting data above ######################################
# Predict the labels for the testing set
y_pred = clf.predict(X_test)

# Calculating accuracy on the testing set used above
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Testing Set:", accuracy)

############################# Predicting 2024 crop labels for its predicted rainfall labels for 4 seasons ###############################

# Replacing 'testing_crop_labels_dataset.csv' with the actual file path and name of your testing crop labels dataset

testing_crop_labels_df = pd.read_csv('Predicted_2024.csv')
new_Column_name = {'predicted_precipitation':"rainfall"}
testing_crop_labels_df.rename(columns = new_Column_name, inplace = True)


# Getting test data from 2024 year 
X_test = testing_crop_labels_df.drop('Season',axis = 1)

# Predicting the labels for new dataset OF 2024 RAINFALL labels
y_pred_crop_label = clf.predict(X_test)

testing_crop_labels_df['Predicted crop label'] = y_pred_crop_label
print('Prediction for new dataset')
print(testing_crop_labels_df)

############################## Crop labels predicted for 2024 year above ####################################################################
print("Model 2 ends with predicted result above")


                                         
                                         







