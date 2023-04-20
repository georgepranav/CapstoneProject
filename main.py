# sample commit for one of my students
import streamlit as st
import pandas as pd
import wine
# Load the dataset from a local file
# Replace 'path/to/your/local/file.csv' with the actual path to the downloaded dataset
data = pd.read_csv('WhiteWineQuality.csv',delimiter=";")


# Create a Streamlit app to display the dataset's first few rows
st.title('White Wine Quality Dataset')
st.header('First few rows of the dataset:')
st.dataframe(wine.dataset_Overview_head(data))

# Create a dataset's statistical summary
st.header("Statistical summary of the dataset:")
st.dataframe(wine.dataset_Overview_describe(data))

# Create the dataset's info
st.header("Display the dataset's info")
st.write(data.info())

# Check for missing values:
st.header("Check for missing values: ")
st.dataframe(wine.dataset_missing_values(data))

# Display a Seaborn countplot of the 'quality' column
st.header("Countplot of 'quality' column:")
st.pyplot(wine.visualize_distribution(data))

# Display a Seaborn heatmap of the correlation matrix
st.header("Heatmap of correlation matrix:")
st.pyplot(wine.explore_correlations(data))

# Display Seaborn boxplots for 'pH' and 'alcohol' columns
st.header("Boxplots for 'pH' and 'alcohol' columns:")
ph, alcohol = wine.visualize_relationships(data)
st.pyplot(ph)
st.pyplot(alcohol)

# Create a RandomForestClassifier model
st.header("Random Forest Classifier Results")
accuracy, report = wine.random_forest_classifier(data)
st.write("Accuracy: ", accuracy)
st.write("Classification Report:")
st.text('Model Report:\n    ' + report)