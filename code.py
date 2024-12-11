# %% [markdown]
# 
# ### Import Libraries and Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# %%
filename=('heart_attack_prediction_dataset.csv')
df=pd.read_csv(filename)
df = pd.DataFrame(df)
df

# %% [markdown]
# ## Data Cleaning

# %%
# Checking the data shape
df.shape

# %%
## Check for null values
for col in df.columns:
    num_nulls = df[col].isnull().sum()
    num_not_null = df[col].notnull().sum()
    pct_nulls = (num_nulls / len(df)) * 100
    print(f"Column: {col}\n Number null: {num_nulls}\n Number not null: {num_not_null}\n Proportion null: {pct_nulls:.2f}%")

# %%
# Check for duplicate records
df.duplicated().sum()


# %%
df.columns

# %%
df=df.drop(columns={'Patient ID'})

# %% [markdown]
# ## Visualizing the data

# %% [markdown]
# ### Distribution of Risk
# 

# %%
df['Heart Attack Risk'] = df['Heart Attack Risk'].map({0: 'Low', 1: 'High'})

df['Heart Attack Risk'] = df['Heart Attack Risk'].astype('category')

plt.figure(figsize=(7,5))
df['Heart Attack Risk'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.title('Distribution for Heart Attack Classification')

# %% [markdown]
# ### Categorical Variables

# %%
categorical_columns = [
    'Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Diet', 
    'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Continent', 'Country', 'Hemisphere',
]

# Plot histograms for each categorical variable with respect to Heart Risk
for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column, hue='Heart Attack Risk')
    plt.title(f'{column}')
    plt.ylabel('Count')
    plt.legend(title='Heart Attack Risk')
    plt.xticks(rotation=90)  # Rotate the x-axis labels by 90 degrees
    plt.xlabel('')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Numerical Variables

# %%
# Split the 'Blood Pressure' column into 'Systolic_BP' and 'Diastolic_BP' columns
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)

# Convert the new columns to integers
df['Systolic_BP'] = df['Systolic_BP'].astype(int)
df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)

# Drop the original 'Blood Pressure' column
df = df.drop(columns=['Blood Pressure'])

# %%
# Plot box plots for each numeric variable with respect to Heart Attack Risk

numerical_columns = [
    'Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 'Sedentary Hours Per Day', 
    'Income', 'BMI', 'Triglycerides', 'Systolic_BP', 'Diastolic_BP', 'Physical Activity Days Per Week', 'Sleep Hours Per Day'
]

for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Heart Attack Risk', y=column, data=df)
    plt.title(f'{column} by Heart Attack Risk')
    plt.xlabel('Heart Attack Risk')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## FEATURE SELECTION

# %% [markdown]
# ### CHI SQUARED TEST ON CATEGORICAL VARIABLES

# %%

# Factoring Categorical variables 
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Healthy': 1, 'Average': 0.5})

continent_mapping = {'Asia': 0, 'Europe': 1, 'North America': 2, 'South America': 3, 'Africa': 4, 'Australia': 5}
df['Continent'] = df['Continent'].map(continent_mapping)

hemisphere_mapping = {'Northern Hemisphere': 0, 'Southern Hemisphere': 1}
df['Hemisphere'] = df['Hemisphere'].map(hemisphere_mapping)

df['Heart Attack Risk'] = df['Heart Attack Risk'].map({'High':1,'Low':0})

#df['Heart Attack Risk'] = df['Heart Attack Risk'].astype('category')

df1 =df


# %%
from scipy.stats import chi2_contingency


# Select only the categorical columns
categorical_columns

# Perform chi-squared test for each categorical variable
chi2_results = {}
for column in categorical_columns:
    contingency_table = pd.crosstab(df['Heart Attack Risk'], df[column])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    chi2_results[column] = {'chi2': chi2, 'p-value': p, 'dof': dof}

# Display the results
chi2_results_df = pd.DataFrame(chi2_results).T
chi2_results_df

# %%
# Select the columns with p-value < 0.5
significant_columns = chi2_results_df[chi2_results_df['p-value'] < 0.5].index

print(f"The columns with p-value < 0.5 are: {', '.join(significant_columns)}")

# %% [markdown]
# ## Correlation Matrix of Numeric Variables

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Create a new DataFrame with only numerical columns and the Heart Risk column
matrix_df = df[numerical_columns]

# Create the correlation matrix
corr_matrix = matrix_df.corr() 

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 10))

# Create the correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd', vmin=-1, vmax=1, center=0, linewidths=.5, ax=ax)

# Set the title and axis labels
ax.set_title('Correlation Matrix', fontsize=20)

# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=90)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()

# %% [markdown]
# Important numeric variables:
# BMI,
# Heart Rate,
# Exercise Hours Per Week,
# Sedentary Hours Per Day,
# Cholesterol,
# Diastolic_BP,
# Physical Activity Days Per Week.

# %% [markdown]
# # Model Selection Before Feature Selection

# %% [markdown]
# ### Feature enginerring 

# %%
from sklearn.preprocessing import OneHotEncoder

# Initialize OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical columns
encoded_array = encoder.fit_transform(df[categorical_columns]).toarray()

# Create a DataFrame with the one-hot encoded columns
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the original DataFrame with the one-hot encoded DataFrame
df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# Print the final DataFrame
df

# %% [markdown]
# ### Train Test Split 

# %%
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

# %%
X = df.drop('Heart Attack Risk', axis=1)
y = df["Heart Attack Risk"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

X_train

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(
    X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(
    X_test), columns=X_train.columns, index=X_test.index)

X_train_scaled.head()

# %% [markdown]
# ## Logistic Regression

# %% [markdown]
# #### Training and Testing on Unbalanced Data

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')

folds = KFold(n_splits=5, shuffle=True, random_state=7)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=folds)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)

print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train, y_pred_train),
      f1_score(y_train, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test, y_pred_test),
      f1_score(y_test, y_pred_test, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred_test,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred_test))

# %% [markdown]
# ### Resampling

# %%
# Install the imbalanced-learn library
#!pip install imbalanced-learn

# Import the necessary modules
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

smote = SMOTE()
rus = RandomUnderSampler()
smoteenn = SMOTEENN()
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)
X_test_resampled, y_test_resampled = rus.fit_resample(X_test_scaled, y_test)



# %% [markdown]
# #### Training on Balanced Data and Testing on unbalanced data

# %%
# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')

folds = KFold(n_splits=5, shuffle=True, random_state=7)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=folds)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_resampled)

print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test, y_pred_test),
      f1_score(y_test, y_pred_test, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred_test,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred_test))

# %% [markdown]
# ### Making synthentic predictions

# %%
from sklearn.datasets import make_classification

# Generate 1000 synthetic samples with the same number of features as the original data
X_synth, y_synth = make_classification(n_samples=10, n_features=X_train.shape[1], random_state=42)

y_synth_pred = best_model.predict(X_synth)

# %%
# Print the individual predictions
print("Actual Labels:", y_synth)
print("Predicted Labels:", y_synth_pred)

# %%
print("Accuracy:", accuracy_score(y_synth, y_synth_pred))


# %% [markdown]
# #### Training and Testing on Balanced Data

# %%
# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')

folds = KFold(n_splits=5, shuffle=True, random_state=7)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=folds)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test_resampled)
y_pred_train = best_model.predict(X_train_resampled)

print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test_resampled, y_pred_test),
      f1_score(y_test_resampled, y_pred_test, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test_resampled,
                                                y_pred_test,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test_resampled,y_pred_test))

# %% [markdown]
# ## SVM

# %%
from sklearn.model_selection import GridSearchCV, KFold

#K fold cross validation and grid search

folds = KFold(n_splits=5, shuffle=True, random_state=7)
model = SVC()


params = {'C': [0.1, 1, 10, 100, 1000, 2000],
          'gamma': [1, 0.1, 0.01]}

c_opt = GridSearchCV(estimator=model, param_grid=params,
                     scoring='f1_macro', cv=folds, n_jobs=-1,
                     verbose=1, return_train_score=True)

c_opt.fit(X_train_scaled, y_train)
c_results = pd.DataFrame(c_opt.cv_results_)
c_results

# %%
#best score and best parameters
best_score = c_opt.best_score_
best_params = c_opt.best_params_

print(f'Best Score: {best_score}')
print(f'Best Parameters: {best_params}')

# %% [markdown]
# #### Training and Testing on unbalanced data

# %%
model_scaled = SVC(gamma=0.01, C=100)
model_scaled.fit(X_train_scaled, y_train)


y_pred_train = model_scaled.predict(X_train_scaled)
y_pred = model_scaled.predict(X_test_scaled)


print('Training Accuracy and Training F1 (macro)')
print(metrics.accuracy_score(y_train, y_pred_train),
      metrics.f1_score(y_train, y_pred_train, average='macro'))

print('\nTesting Accuracy and Testing F1 (macro)')
print(metrics.accuracy_score(y_test, y_pred),
      metrics.f1_score(y_test, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred))

# %% [markdown]
# #### Training and Testing on balanced data

# %%
model_scaled = SVC(gamma=0.1, C=100)
model_scaled.fit(X_train_resampled, y_train_resampled)


y_pred_train = model_scaled.predict(X_train_resampled)
y_pred = model_scaled.predict(X_test_resampled)


print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test_resampled, y_pred),
      f1_score(y_test_resampled, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test_resampled,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test_resampled,y_pred))

# %% [markdown]
# #### Training on balanced data and Testing on unbalanced data

# %%
model_scaled = SVC(gamma=0.1, C=100)
model_scaled.fit(X_train_resampled, y_train_resampled)


y_pred_train = model_scaled.predict(X_train_resampled)
y_pred = model_scaled.predict(X_test_scaled)


print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred))

# %% [markdown]
# ### Making Synthetic Predictions 

# %%
from sklearn.datasets import make_classification

# Generate 1000 synthetic samples with the same number of features as the original data
X_synth, y_synth = make_classification(n_samples=10, n_features=X_train.shape[1], random_state=42)

y_synth_pred = model_scaled.predict(X_synth)

# %%
# Print the individual predictions
print("Actual Labels:", y_synth)
print("Predicted Labels:", y_synth_pred)

# %%
print("Accuracy:", accuracy_score(y_synth, y_synth_pred))

# %% [markdown]
# # Model Selection After Feature Selection

# %%

selected_categorical = ['Diabetes', 'Obesity', 'Alcohol Consumption', 'Hemisphere']

selected_numerical = ['Cholesterol', 'BMI', 'Exercise Hours Per Week', 'Heart Rate', 'Physical Activity Days Per Week', 
                      'Sedentary Hours Per Day', 'Diastolic_BP']

selected_variables = ['Diabetes', 'Obesity', 'Alcohol Consumption', 'Hemisphere', 'Cholesterol', 'BMI', 
                      'Exercise Hours Per Week', 'Heart Rate', 'Physical Activity Days Per Week', 'Sedentary Hours Per Day', 'Diastolic_BP', 'Heart Attack Risk']

data = df1[selected_variables]
data

# %% [markdown]
# ### Feature Engineering

# %%

from sklearn.preprocessing import OneHotEncoder

# Initialize OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical columns
encoded_array = encoder.fit_transform(data[selected_categorical]).toarray()

# Create a DataFrame with the one-hot encoded columns
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(selected_categorical))

# Concatenate the original DataFrame with the one-hot encoded DataFrame
data = pd.concat([data.drop(columns=selected_categorical), encoded_df], axis=1)

#data['Heart Attack Risk'] = data['Heart Attack Risk'].astype('int')

# Print the final DataFrame
data

# %% [markdown]
# ### Train Test Split

# %%
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

# %%
X = data.drop('Heart Attack Risk', axis=1)
y = data["Heart Attack Risk"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

X_train

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(
    X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(
    X_test), columns=X_train.columns, index=X_test.index)

X_train_scaled.head()

# %% [markdown]
# ## Logistic Regression

# %% [markdown]
# #### Training and Testing on Unbalanced Data

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')

folds = KFold(n_splits=5, shuffle=True, random_state=7)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=folds)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)

print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train, y_pred_train),
      f1_score(y_train, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test, y_pred_test),
      f1_score(y_test, y_pred_test, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred_test,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred_test))

# %% [markdown]
# ## Resampling

# %%
# Install the imbalanced-learn library
#!pip install imbalanced-learn

# Import the necessary modules
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

smote = SMOTE()
rus = RandomUnderSampler()
smoteenn = SMOTEENN()
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)
X_test_resampled, y_test_resampled = rus.fit_resample(X_test_scaled, y_test)

# %% [markdown]
# #### Training on Balanced Data and Testing on unbalanced data

# %%
# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')

folds = KFold(n_splits=5, shuffle=True, random_state=7)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=folds)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_resampled)

print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test, y_pred_test),
      f1_score(y_test, y_pred_test, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred_test,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred_test))

# %% [markdown]
# #### Training and Testing on Balanced Data

# %%
# Define the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

# Create the logistic regression model
model = LogisticRegression(solver='liblinear')

folds = KFold(n_splits=5, shuffle=True, random_state=7)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=folds)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test_resampled)
y_pred_train = best_model.predict(X_train_resampled)

print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test_resampled, y_pred_test),
      f1_score(y_test_resampled, y_pred_test, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test_resampled,
                                                y_pred_test,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test_resampled,y_pred_test))

# %% [markdown]
# ## SVM

# %%
from sklearn.model_selection import GridSearchCV, KFold

#K fold cross validation and grid search

folds = KFold(n_splits=5, shuffle=True, random_state=7)
model = SVC()


params = {'C': [0.1, 1, 10, 100, 1000, 2000],
          'gamma': [1, 0.1, 0.01]}

c_opt = GridSearchCV(estimator=model, param_grid=params,
                     scoring='f1_macro', cv=folds, n_jobs=-1,
                     verbose=1, return_train_score=True)

c_opt.fit(X_train_scaled, y_train)
c_results = pd.DataFrame(c_opt.cv_results_)
c_results

# %%
#best score and best parameters
best_score = c_opt.best_score_
best_params = c_opt.best_params_

print(f'Best Score: {best_score}')
print(f'Best Parameters: {best_params}')

# %% [markdown]
# #### SVM Model based on best params

# %% [markdown]
# ##### Training on balaced data and testing on unbalanced data

# %%

model_scaled = SVC(gamma=0.1, C=100)
model_scaled.fit(X_train_resampled, y_train_resampled)


y_pred_train = model_scaled.predict(X_train_resampled)
y_pred = model_scaled.predict(X_test_scaled)


print('Training accuracy score and Training F1 score:')
print(accuracy_score(y_train_resampled, y_pred_train),
      f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting accuracy score and Testing F1 score:')
print(accuracy_score(y_test, y_pred),
      f1_score(y_test, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred))

# %% [markdown]
# ##### Training and testing on unbalanced data

# %%
model_scaled = SVC(gamma=0.1, C=100)
model_scaled.fit(X_train_scaled, y_train)


y_pred_train = model_scaled.predict(X_train_scaled)
y_pred = model_scaled.predict(X_test_scaled)


print('Training Accuracy and Training F1 (macro)')
print(metrics.accuracy_score(y_train, y_pred_train),
      metrics.f1_score(y_train, y_pred_train, average='macro'))

print('\nTesting Accuracy and Testing F1 (macro)')
print(metrics.accuracy_score(y_test, y_pred),
      metrics.f1_score(y_test, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred))

# %% [markdown]
# ##### Training and testing on balanced data

# %%
model_scaled = SVC(gamma=0.1, C=100)
model_scaled.fit(X_train_resampled, y_train_resampled)


y_pred_train = model_scaled.predict(X_train_resampled)
y_pred = model_scaled.predict(X_test_resampled)


print('Training Accuracy and Training F1 (macro)')
print(metrics.accuracy_score(y_train_resampled, y_pred_train),
      metrics.f1_score(y_train_resampled, y_pred_train, average='macro'))

print('\nTesting Accuracy and Testing F1 (macro)')
print(metrics.accuracy_score(y_test_resampled, y_pred),
      metrics.f1_score(y_test_resampled, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test_resampled,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test_resampled,y_pred))

# %% [markdown]
# ## Decision Trees

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Create a NumPy array from 1 to 49 with a step of 1
max_depths = np.arange(1, 15, 1)

# Lists to store the training and testing F1 scores
train_f1_scores = []
test_f1_scores = []

# Loop through the max_depth values and fit the models
for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    train_f1_scores.append(train_f1)

    y_test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_f1_scores.append(test_f1)

# %%
# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_f1_scores, label='Training F1 Score')
plt.plot(max_depths, test_f1_scores, label='Testing F1 Score')
plt.xlabel('Max Depth')
plt.ylabel('F1 Score')
plt.title('Training and Testing F1 Scores')
plt.legend()
plt.show()

# %% [markdown]
# The best max_depth to choose in this case would be a lower value, 5, where the training and testing F1 scores are closer together and the model is not overfitting.

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train the model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Convert class names to strings
class_names = list(map(str, model.classes_))

# Plot the decision tree
plt.figure(figsize=(25, 15))
_ = plot_tree(model, 
              feature_names=X_train.columns,  
              class_names=class_names,
              filled=True)
plt.show()


# %%
text_representation = tree.export_text(model)
print(text_representation)

# %%
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)


print('Training Accuracy and Training F1 (macro)')
print(metrics.accuracy_score(y_train, y_pred_train),
      metrics.f1_score(y_train, y_pred_train, average='macro'))

print('\nTesting Accuracy and Testing F1 (macro)')
print(metrics.accuracy_score(y_test, y_pred),
      metrics.f1_score(y_test, y_pred, average='macro'))

# %%
#Confusion Matrix 
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred,
                                                xticks_rotation='vertical')
plt.show()

# %%
print(metrics.classification_report(y_test,y_pred))

# %% [markdown]
# ### Making Synthetic Predictions

# %%
from sklearn.datasets import make_classification

# Generate 1000 synthetic samples with the same number of features as the original data
X_synth, y_synth = make_classification(n_samples=10, n_features=X_train.shape[1], random_state=42)

y_synth_pred = model.predict(X_synth)

# %%
# Print the individual predictions
print("Actual Labels:", y_synth)
print("Predicted Labels:", y_synth_pred)

# %%
print("Accuracy:", accuracy_score(y_synth, y_synth_pred))


