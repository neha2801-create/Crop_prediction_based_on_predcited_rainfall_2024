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

####################################    Filtering data based on column names further getting India specific data############
india_data = dataset_df[dataset_df['name'] == 'India']

################################        Converting datetime column value into datetime type value ###########################
india_data['datetime'] = pd.to_datetime(india_data['datetime'])

##############################   Extracting feature and target variables################################################
X_train = india_data[['datetime']]
y_train = india_data['precip']

#############################   




