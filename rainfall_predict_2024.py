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

#################################    Data exploration  ###################################################
#  Converting 'datetime' column to datetime type data
dataset_df['datetime'] = pd.to_datetime(dataset_df['datetime'])

# Extracting year value from datetime type of data
dataset_df['year'] = dataset_df['datetime'].dt.year

print(f'Exploratory Data Analysis')
print(f'Precipitation over various months through the years 2018 - 2023')

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


####################################    Filtering data based on column names further getting India specific data############
india_data = dataset_df[dataset_df['name'] == 'India']

################################        Converting datetime column value into datetime type value ###########################
india_data['datetime'] = pd.to_datetime(india_data['datetime'])

##############################   Extracting feature and target variables################################################
X_train = india_data[['datetime']]
y_train = india_data['precip']

#############################




