# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the CSV into a DataFrame
csv_path = "./datasets/telecom_churn_data.csv"
telecom_df = pd.read_csv(csv_path)

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(telecom_df.head())

# Display summary information about the DataFrame
telecom_df.info(verbose=1, null_counts=True, memory_usage=True)

# Display the shape of the DataFrame
print("DataFrame shape:", telecom_df.shape)


# Display the initial number of rows and columns in the dataframe
init_rows, init_cols = telecom_df.shape
print("Number of rows:", init_rows)
print("Number of columns:", init_cols)

# Calculate the percentage of missing values for each feature
missing_percent = (100 * (telecom_df.isnull().sum() / len(telecom_df))).round(2)
missing_percent = missing_percent.reset_index().rename(columns={'index': 'features', 0: 'Percentage'})

# Display the count of features with missing percentages within certain ranges
print("Missing percentage < 10%:", missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].shape[0])
print("Missing percentage 10-60%:", missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].shape[0])
print("Missing percentage >= 60%:", missing_percent[(missing_percent['Percentage'] >= 60.00)].shape[0])

# Display attributes with more than 60% missing value
print("Attributes with more than 60% missing value:")
print(missing_percent[missing_percent['Percentage'] > 60.00].features.tolist())

# Display the value counts for a specific column
print("Value counts for fb_user_8:")
print(telecom_df.fb_user_8.value_counts().to_string(index=False))

# List attributes with more than 60% missing value
no_data_rech_cols = missing_percent[missing_percent['Percentage'] > 60.00].features.tolist()
print("Attributes with more than 60% missing value:", no_data_rech_cols)


# Replace missing values with 0 for specified columns
telecom_df[no_data_rech_cols] = telecom_df[no_data_rech_cols].fillna(value=0)

# Resetting the 'missing_percent' DataFrame with updated 'telecom_df' data
missing_percent = pd.DataFrame(round(100 * (telecom_df.isnull().sum() / len(telecom_df)), 2), columns=['Percentage'])
missing_percent = missing_percent.reset_index().rename(columns={'index': 'features'})

# Display the count of features with missing percentages above 60%
print("Number of features with missing percentage > 60%:", missing_percent[missing_percent['Percentage'] > 60.00].shape[0])

# Display attributes with missing percentages less than 10%
print("Attributes with missing percentage 0-10%:")
print(missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].to_string(index=False))

# Validate data in 'loc_og_t2o_mou' attribute
print("Value counts for loc_og_t2o_mou:")
print(telecom_df['loc_og_t2o_mou'].value_counts().to_string(index=False))

# Validate data in 'std_og_t2o_mou' attribute
print("Value counts for std_og_t2o_mou:")
print(telecom_df['std_og_t2o_mou'].value_counts().to_string(index=False))

# Validate data in 'loc_ic_t2o_mou' attribute
print("Value counts for loc_ic_t2o_mou:")
print(telecom_df['loc_ic_t2o_mou'].value_counts().to_string(index=False))

# Drop attributes with mostly zero values and 1% missing values
attributes_to_drop = ['std_og_t2o_mou', 'loc_og_t2o_mou', 'loc_ic_t2o_mou']
telecom_df.drop(attributes_to_drop, axis=1, inplace=True)
print("Attributes dropped:", attributes_to_drop)
print(telecom_df.head())

# Filling last date of month columns with last date
last_month_columns = ['last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9']
for col in last_month_columns:
    telecom_df[col] = telecom_df[col].fillna(value=f'{col[-1]}/30/2014')

# Verifying recharge date distribution for each month
recharge_columns = ['date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9']
for col in recharge_columns:
    print(f"Value counts for {col}:")
    print(telecom_df[col].value_counts().to_string(index=False))
    print()

# Filling missing data in last date of recharge columns with mode
for col in recharge_columns:
    telecom_df[col] = telecom_df[col].fillna(value=telecom_df[col].mode()[0])

# Resetting the 'missing_percent' DataFrame with updated 'telecom_df' data
missing_percent = pd.DataFrame(round(100 * (telecom_df.isnull().sum() / len(telecom_df)), 2), columns=['Percentage'])
missing_percent = missing_percent.reset_index().rename(columns={'index': 'features'})

# Display the count of features with missing percentages within certain ranges
print("Missing percentage < 10%:", missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].shape[0])
print("Missing percentage 10-60%:", missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].shape[0])
print("Missing percentage >= 60%:", missing_percent[(missing_percent['Percentage'] >= 60.00)].shape[0])

# Display attributes with missing percentages 0-10%
print("Attributes with missing percentage 0-10%:")
print(missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].to_string(index=False))

# List of columns with similar missing percentages
similar_missing_cols = missing_percent[
    (missing_percent['Percentage'] == 3.94) | (missing_percent['Percentage'] == 3.86) |
    (missing_percent['Percentage'] == 5.38) | (missing_percent['Percentage'] == 7.75)].features

# Fill null columns with zero for specific months
telecom_df[similar_missing_cols] = telecom_df[similar_missing_cols].fillna(value=0)


# Checking the percentage of missing values and displaying counts based on percentages
missing_percent = (telecom_df.isnull().sum() / len(telecom_df.index)) * 100
missing_percent_df = pd.DataFrame({'features': missing_percent.index, 'Percentage': missing_percent.values})
missing_percent_df['Percentage'] = missing_percent_df['Percentage'].round(2)

print("Missing percentage < 10%:", missing_percent_df[(missing_percent_df['Percentage'] > 0.00) & (missing_percent_df['Percentage'] < 10.00)].shape[0])
print("Missing percentage 10-60%:", missing_percent_df[(missing_percent_df['Percentage'] >= 10.00) & (missing_percent_df['Percentage'] < 60.00)].shape[0])
print("Missing percentage >= 60%:", missing_percent_df[(missing_percent_df['Percentage'] >= 60.00)].shape[0])

# Filtering high value customer data set from telecom data
telecom_df['av_total_rech_goodPhase'] = (telecom_df['total_rech_amt_6'] + telecom_df['total_rech_amt_7']) / (
    telecom_df['total_rech_num_6'] + telecom_df['total_rech_num_7'])

recharge_quantile = 0.7
recharge_threshold = telecom_df['av_total_rech_goodPhase'].quantile(recharge_quantile)
print(f"{recharge_quantile * 100}% of average Recharge Amount of month 6 and 7:", recharge_threshold)

HV_Cust_df = telecom_df[telecom_df.av_total_rech_goodPhase >= recharge_threshold]
HV_Cust_df = HV_Cust_df.drop(['av_total_rech_goodPhase'], axis=1)

# Tag Churn/Non-Churn based on ninth month attributes
churn_attributes = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']
churn_info = HV_Cust_df[churn_attributes]

def churn_detect(row):
    if all([row[attr] == 0.0 for attr in churn_attributes]):
        return 1
    else:
        return 0

HV_Cust_df['Churn'] = churn_info.apply(churn_detect, axis=1)

# Display the Churn column along with 9th month attributes
churn_info['Churn'] = HV_Cust_df['Churn']
print(churn_info.head(10))


# Removing all attributes corresponding to Churn phase
non_churn_phase_cols = [col for col in HV_Cust_df.columns if "_9" not in col]
num_non_churn_cols = len(non_churn_phase_cols)
print("Number of attributes in non-churn phase:", num_non_churn_cols)

# Keeping the dataset with non-churn phase attributes only
HVC_telecom = HV_Cust_df[non_churn_phase_cols]

# Displaying the first few rows of the new dataset
print("Head of the dataset with non-churn phase attributes:")
print(HVC_telecom.head())

# Displaying the shape of the new dataset
print("Shape of the dataset with non-churn phase attributes:", HVC_telecom.shape)

# Displaying information about missing values in the new dataset
print("Information about missing values:")
print(HVC_telecom.info(verbose=True, null_counts=True))


# Columns to drop
drop_cols = ['std_ic_t2o_mou_6', 'std_ic_t2o_mou_7', 'std_ic_t2o_mou_8', 'std_og_t2c_mou_6', 'std_og_t2c_mou_7', 'std_og_t2c_mou_8']
HVC_telecom.drop(columns=drop_cols, axis=1, inplace=True)

# Dropping the phone number attribute as it's not useful for the model
HVC_telecom.drop(['mobile_number'], axis=1, inplace=True)

# Displaying the first few rows of the modified dataset
print("Head of the dataset after dropping columns:")
print(HVC_telecom.head())

# Checking the distribution of circle_id attribute
print("Circle ID distribution:")
print(HVC_telecom.circle_id.value_counts())

# Dropping the circle_id attribute as it has only one value for the whole dataset
HVC_telecom.drop(['circle_id'], axis=1, inplace=True)

# Columns related to date attributes
date_cols = [col for col in HVC_telecom.columns if "date" in col]

# Converting date attributes to datetime type
HVC_telecom[date_cols] = HVC_telecom[date_cols].apply(pd.to_datetime)

# Function to find the last recharge data/call for each month in days for each user
def find_last_rech(df, month_num_str):
    diff_date = df['last_date_of_month_' + month_num_str] - df['date_of_last_rech_' + month_num_str]
    diff_data = df['last_date_of_month_' + month_num_str] - df['date_of_last_rech_data_' + month_num_str]
    return (diff_date.dt.days if (diff_date.dt.days <= diff_data.dt.days).any() else diff_data.dt.days)

# Applying the function for all months
for month_num_str in ["6", "7", "8"]:
    HVC_telecom['last_rech_day_' + month_num_str] = find_last_rech(HVC_telecom, month_num_str)

# Dropping the original date columns
HVC_telecom.drop(columns=date_cols, axis=1, inplace=True)
HVC_telecom.head()


# Displaying information about the dataset including memory usage and null counts
print("Dataset Information:")
HVC_telecom.info(verbose=True, memory_usage=True, null_counts=True)

# Converting specific attributes from float64 to int64
cols_int = [col for col in HVC_telecom.columns if ("count" in col) or ("fb" in col)]
cols_int.extend(['total_rech_num_6', 'total_rech_num_7', 'total_rech_num_8', 'total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8'])

print("\nColumns to convert to int64:", cols_int)

# Converting selected columns to int64
HVC_telecom[cols_int] = HVC_telecom[cols_int].astype('int64')

# Displaying dataset information after data type conversion
print("\nDataset Information after type conversion:")
HVC_telecom.info(verbose=True, memory_usage=True, null_counts=True)

# Displaying the first few rows of the modified dataset
print("\nHead of the dataset after type conversion:")
print(HVC_telecom.head(10))

# Calculating and displaying class imbalance percentage
class_imbalance_percentage = 100 * (HVC_telecom['Churn'].value_counts(normalize=True))
print("\nClass Imbalance in the dataset:")
print(class_imbalance_percentage)

# Describing Minute of Usage related columns
mou_cols = [col for col in HVC_telecom.columns if "mou" in col]
print("\nTotal MOU attributes:", len(mou_cols))
print("\nDescription of MOU attributes:")
print(HVC_telecom[mou_cols].describe())

# Describing maximum recharge data and average recharge data attributes
rech_data_cols = ['max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 
                  'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']
rech_data_desc = HVC_telecom[rech_data_cols].describe(percentiles=[.25, .5, .75, .90, .95, .99])
print("\nDescription of Recharge Data attributes:")
print(rech_data_desc)

# Boxplot for maximum recharge data attributes
sns.boxplot(data=HVC_telecom, y='max_rech_data_8')
plt.title("Boxplot of max_rech_data_8")

# Filtering rows based on specific conditions for further analysis
filtered_rows = HVC_telecom[(HVC_telecom['max_rech_data_6'] > 1200) & 
                            (HVC_telecom['max_rech_data_7'] > 1200) & 
                            (HVC_telecom['max_rech_data_8'] > 1200)]

# Displaying selected columns for the filtered rows
cols_to_display = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 
                   'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 
                   'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']
filtered_data = filtered_rows[cols_to_display]

print("\nFiltered Data based on max_rech_data conditions:")
print(filtered_data)


# Filtering out rows based on maximum recharge data conditions
HVC_telecom = HVC_telecom[(HVC_telecom['max_rech_data_6'] < 1200) & 
                          (HVC_telecom['max_rech_data_7'] < 1200) & 
                          (HVC_telecom['max_rech_data_8'] < 1200)]

# Boxplot for maximum recharge data attribute
sns.boxplot(data=HVC_telecom, y='max_rech_data_8')
plt.title("Boxplot of max_rech_data_8")

# Function to identify and fix wrongly calculated average recharge amounts
def fix_wrong_avg_rech(df, month_col, max_rech_col, avg_rech_col):
    return df[(df[month_col] > 1) & 
              (df[month_col] * df[max_rech_col] == df[avg_rech_col])]

# Identifying and fixing wrongly calculated average recharge amounts for each month
wrong_cal_6 = fix_wrong_avg_rech(HVC_telecom, 'total_rech_data_6', 'max_rech_data_6', 'av_rech_amt_data_6')
wrong_cal_7 = fix_wrong_avg_rech(HVC_telecom, 'total_rech_data_7', 'max_rech_data_7', 'av_rech_amt_data_7')
wrong_cal_8 = fix_wrong_avg_rech(HVC_telecom, 'total_rech_data_8', 'max_rech_data_8', 'av_rech_amt_data_8')

# Dropping rows with wrongly calculated average recharge amounts
wrong_cal_indices = wrong_cal_6.index.union(wrong_cal_7.index).union(wrong_cal_8.index)
HVC_telecom.drop(wrong_cal_indices, inplace=True)

# Displaying Churn count after dropping wrongly calculated rows
print("\nChurn Count after Dropping Wrongly Calculated Rows:")
print(HVC_telecom.Churn.sum())

# Displaying Churn count for individual months with wrongly calculated average values
print("\nChurn Count for Individual Months with Wrongly Calculated Averages:")
print("Month 6 Churn Count:", wrong_cal_6.Churn.sum())
print("Month 7 Churn Count:", wrong_cal_7.Churn.sum())
print("Month 8 Churn Count:", wrong_cal_8.Churn.sum())


# User-defined function to calculate average recharge data
def cal_av_rech_data(df, col):
    if (df['total_rech_data_' + col] > 1) & (
            df['total_rech_data_' + col] * df['max_rech_data_' + col] == df['av_rech_amt_data_' + col]).all():
        return df['max_rech_data_' + col] / df['total_rech_data_' + col]
    else:
        return df['av_rech_amt_data_' + col]

# Applying the function for all months
for col in ['6', '7', '8']:
    HVC_telecom['av_rech_amt_data_' + col] = cal_av_rech_data(HVC_telecom, col)

# Dropping the total recharge data attributes
HVC_telecom.drop(columns=['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8'], axis=1, inplace=True)

# Merging month 6 and 7 attributes
cols_6_7 = [col for col in HVC_telecom.columns if ("_6" in col) or ("_7" in col)]

for i, v in enumerate(cols_6_7):
    if "_6" in v:
        col = "av_" + v.split("6")[0] + "good"
        HVC_telecom[col] = (HVC_telecom[v] + HVC_telecom[cols_6_7[i + 1]]) / 2

# Dropping the previous 6 and 7 attributes
HVC_telecom.drop(columns=cols_6_7, axis=1, inplace=True)

# Rechecking the percentage of missing values
missing_percent = pd.DataFrame(round(100 * (HVC_telecom.isnull().sum() / len(HVC_telecom.index)), 2),
                                columns=['Percentage'])
missing_percent.reset_index(inplace=True)
missing_percent.rename(columns={'index': 'features'}, inplace=True)

print("missing percentage < 10%",
      missing_percent[(missing_percent['Percentage'] > 0.00) & (missing_percent['Percentage'] < 10.00)].count())
print("missing percentage 10-60%",
      missing_percent[(missing_percent['Percentage'] >= 10.00) & (missing_percent['Percentage'] < 60.00)].count())
print("missing percentage >= 60%", missing_percent[(missing_percent['Percentage'] >= 60.00)].count())

# Saving the cleaned data to a CSV file
HVC_telecom.to_csv("./datasets/HVC_telecom_cleaned.csv", index=False)

# Loading the cleaned data from the CSV file
HVC_telecom = pd.read_csv("./datasets/HVC_telecom_cleaned.csv")

# Divide the dataset into X & y
X = HVC_telecom.drop('Churn', axis=1)
y = HVC_telecom['Churn']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)


# Applying scaling on the train and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using PCA for dimensionality reduction
pca = PCA(svd_solver='randomized', random_state=50)
pca.fit(X_train_scaled)

# Plotting the cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(color='grey', linestyle='--', linewidth=0.5)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Choosing 50 components as they explain 95% variance
n_components = 50
pca_final = IncrementalPCA(n_components=n_components)

# Transforming the train and test data using PCA
df_train_pca = pca_final.fit_transform(X_train_scaled)
df_test_pca = pca_final.transform(X_test_scaled)

# Validating correlations between components
corrmat = np.corrcoef(df_train_pca.transpose())
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("Max Correlation:", corrmat_nodiag.max(), ", Min Correlation:", corrmat_nodiag.min())

# Plotting the first two principal components
plt.scatter(df_train_pca[:, 0], df_train_pca[:, 1], c=y_train.map({0: 'green', 1: 'red'}))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()

# Logistic Regression with PCA
logReg = LogisticRegression(class_weight='balanced')
model_logReg = logReg.fit(df_train_pca, y_train)
y_pred_logReg = model_logReg.predict(df_test_pca)

print("Logistic Regression with PCA:")
print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_logReg))
print("Sensitivity/Recall:", metrics.recall_score(y_test, y_pred_logReg))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_logReg))

# Advanced Regression with GridSearchCV and PCA
logR = LogisticRegression(class_weight='balanced')
params = {'penalty': ['l1', 'l2'],
          'C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1]}

grid = GridSearchCV(estimator=logR, param_grid=params, cv=5, verbose=1, scoring='recall', return_train_score=True)
grid.fit(df_train_pca, y_train)

print("Best Train Score:", grid.best_score_)
print("Best Parameters:", grid.best_params_)

logReg = LogisticRegression(penalty=grid.best_params_['penalty'], 
                            C=grid.best_params_['C'], class_weight='balanced')
model_advanced_reg = logReg.fit(df_train_pca, y_train)
y_pred_advanced_reg = model_advanced_reg.predict(df_test_pca)

print("Advanced Regression with PCA:")
print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_advanced_reg))
print("Sensitivity/Recall:", metrics.recall_score(y_test, y_pred_advanced_reg))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_advanced_reg))

# RandomForestClassifier with PCA
rfc = RandomForestClassifier(class_weight='balanced')
model_rfc = rfc.fit(df_train_pca, y_train)
y_pred_rfc = model_rfc.predict(df_test_pca)

print("RandomForestClassifier with PCA:")
print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_rfc))
print("Sensitivity/Recall:", metrics.recall_score(y_test, y_pred_rfc))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_rfc))

# Advanced Regression with Elastic Net and PCA
logR = LogisticRegression(class_weight='balanced')
params = {'penalty': ['l1', 'l2'],
          'C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1],
          'l1_ratio': [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]}

grid = GridSearchCV(estimator=logR, param_grid=params, cv=5, verbose=1, scoring='recall', return_train_score=True)
grid.fit(df_train_pca, y_train)

print("Elastic Net Train score:", grid.best_score_)
print("Elastic Net Parameters:", grid.best_params_)

logReg = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'],
                            class_weight='balanced', l1_ratio=grid.best_params_['l1_ratio'])
model_elastic_net = logReg.fit(df_train_pca, y_train)
y_pred_elastic_net = model_elastic_net.predict(df_test_pca)

print("Advanced Regression with Elastic Net and PCA:")
print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_elastic_net))
print("Sensitivity/Recall:", metrics.recall_score(y_test, y_pred_elastic_net))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_elastic_net))

# Final model for Churn/Non-Churn prediction
logReg = LogisticRegression(penalty='l2', C=0.0001, class_weight='balanced')
final_classifier = logReg.fit(df_train_pca, y_train)

# Predicting Churn/Non-Churn using the model
y_pred_final = final_classifier.predict(df_test_pca)

# Displaying the Accuracy, Recall, and Confusion matrix
print("Final Model - Churn/Non-Churn Prediction:")
print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred_final))
print("Sensitivity/Recall:", metrics.recall_score(y_test, y_pred_final))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_final))

# Saving the model to a pkl file
with open('./model/final_model.pkl', 'wb') as model_file:
    pickle.dump(final_classifier, model_file)


