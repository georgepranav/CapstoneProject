# Exploratory Data Analysis and Visualization:
# First, let's come up with five questions we can explore during our EDA:
#
# a. What is the distribution of wine quality in the dataset?
# b. Are there any correlations between the wine's physicochemical attributes and
# its quality?
# c. How do different acidity levels (pH) impact the quality of white wine?
# d. What is the relationship between alcohol content and wine quality?
# e. Are there any outliers or interesting patterns in the data that could
# impact our predictions?
#
# To perform EDA and visualization, use the following steps:
#
# 1.1. Import necessary libraries and load the dataset:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# 1.2. Get a basic overclearview of the dataset:
def dataset_Overview_head(data, no_of_rows: int = 10):
    """Display the first few rows of the dataset"""
    print("First few rows of the dataset:")
    return data.head(no_of_rows)


def dataset_Overview_describe(data):
    """Display the dataset's statistical summary"""
    print("\nStatistical summary of the dataset:")
    return data.describe()


def dataset_Overview_info(data):
    """Display the dataset's info"""
    print("\nDataset info:")
    return data.info()


# 1.3. Check for missing values:
def dataset_missing_values(data):
    # 1.3. Check for missing values:
    print("\nCheck for missing values:")
    return data.isnull().sum()


# 1.4. Visualize the distribution of wine quality:
def visualize_distribution(data):
    """Visualize the distribution of wine quality"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='quality', data=data)
    return plt.gcf()


# 1.5. Explore correlations between variables:
def explore_correlations(data):
    """Explore correlations between variables"""
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    return plt.gcf()


# 1.6. Visualize other relationships (e.g., pH vs. quality, alcohol vs. quality):
def visualize_relationships(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='quality', y='pH', data=data)
    ph = plt.gcf()  # Display the current Matplotlib figure

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='quality', y='alcohol', data=data)
    alcohol = plt.gcf()  # Display the current Matplotlib figure
    return ph, alcohol


# Classification (Predictive Analytics):

# 2.1. Prepare the dataset for classification:

def __dataset_Classification(data):
    # Create a binary target variable for classification (e.g., good quality: 1, bad quality: 0)
    data['good_quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

    # Split the dataset into features (X) and target (y)
    X = data.drop(['quality', 'good_quality'], axis=1)
    y = data['good_quality']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test,y_train,y_test

# Function to convert classification report to DataFrame
def __classification_report_to_df(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    # Remove empty key-value pairs
    report = {key: value for key, value in report.items() if key.strip()}
    report_df = pd.DataFrame(report)
    report_df = report_df.reset_index().rename(columns={'index': 'class'})
    return report_df

# 2.2. Train a classifier (e.g., using RandomForestClassifier):
def random_forest_classifier(data):

    """Create a RandomForestClassifier model"""
    X_train,X_test,y_train,y_test = __dataset_Classification(data)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #report = __classification_report_to_df(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy,report

