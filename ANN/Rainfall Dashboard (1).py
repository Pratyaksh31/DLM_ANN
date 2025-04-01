import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.express as px

st.set_page_config(page_title='Rainfall Prediction Dashboard', layout='wide')

st.title('Rainfall Prediction Dashboard')

# Load the dataset
@st.cache_data  # Cache data to improve performance
def load_data(file_path):
    df = pd.read_csv(file_path, na_filter=False)
    return df

import pandas as pd
import streamlit as st

# Correct raw URL of the CSV file from GitHub
file_url = 'https://raw.githubusercontent.com/Pratyaksh31/DLM_ANN/main/ANN/weatherAUS.csv'

# Load the CSV data into a DataFrame
df = pd.read_csv(file_url)

# Display the first few rows of the dataframe to verify
st.write(df.head())


# Preprocessing function (handle missing values and categorical features)
@st.cache_data
def preprocess_data(df):
    # Replace 'NA' strings with np.nan
    df = df.replace('NA', np.nan)

    # Convert Date to datetime and extract year, month, day
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    # Handle missing values using median for numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Convert RainToday and RainTomorrow to numerical (0 and 1)
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).fillna(0)

    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

# Load data
file_path = "weatherAUS.csv"  # Use the local file path
df = load_data(file_path)

# Preprocess data
df = preprocess_data(df.copy())

# Separate features and target variable
X = df.drop(['RainTomorrow', 'Date'], axis=1)
y = df['RainTomorrow']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Streamlit app
st.title("Rainfall Prediction Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Location filter
locations = df['Location'].unique()
selected_locations = st.sidebar.multiselect("Select Locations", locations, default=locations)
filtered_df = df[df['Location'].isin(selected_locations)]

# Date range filter
min_date = filtered_df['Date'].min().date()
max_date = filtered_df['Date'].max().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Convert selected dates to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter by date
filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

# Numerical feature filters
st.sidebar.header("Numerical Filters")
min_temp_threshold = st.sidebar.slider("Minimum Temperature Threshold", float(df['MinTemp'].min()), float(df['MinTemp'].max()), float(df['MinTemp'].min()))
max_temp_threshold = st.sidebar.slider("Maximum Temperature Threshold", float(df['MaxTemp'].min()), float(df['MaxTemp'].max()), float(df['MaxTemp'].max()))
filtered_df = filtered_df[(filtered_df['MinTemp'] >= min_temp_threshold) & (filtered_df['MaxTemp'] <= max_temp_threshold)]

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Average rainfall by location
avg_rainfall = filtered_df.groupby('Location')['Rainfall'].mean().reset_index()

# Visualizations
st.subheader("Visualizations")

# Rainfall distribution
st.subheader("Rainfall Distribution")
fig_rainfall = px.histogram(filtered_df, x="Rainfall", title="Rainfall Distribution")
st.plotly_chart(fig_rainfall)

fig_avg_rainfall = px.bar(avg_rainfall, x='Location', y='Rainfall', title='Average Rainfall by Location')
st.plotly_chart(fig_avg_rainfall)

# Hyperparameter tuning
st.sidebar.header('Hyperparameter Tuning')
pc31_learning_rate = st.sidebar.slider('Learning Rate', 0.0001, 0.01, 0.001, step=0.0001, format='%.4f')
pc31_batch_size = st.sidebar.selectbox('Batch Size', [32, 64, 128, 256, 512])
pc31_epochs = st.sidebar.selectbox('Epochs', [pc31_i * 10 for pc31_i in range(1, 11)])
pc31_num_layers = st.sidebar.slider('Number of Hidden Layers', 1, 10, 3)
pc31_neurons_per_layer = [st.sidebar.selectbox(f'Neurons in Layer {pc31_i+1}', [2**pc31_j for pc31_j in range(4, 10)]) for pc31_i in range(pc31_num_layers)]
pc31_dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.2, step=0.05)

# Custom Model Building
def build_custom_model():
    pc31_model = Sequential()
    pc31_model.add(Dense(pc31_neurons_per_layer[0], activation='relu', input_shape=(X_train.shape[1],)))
    for pc31_i in range(1, pc31_num_layers):
        pc31_model.add(Dense(pc31_neurons_per_layer[pc31_i], activation='relu'))
        pc31_model.add(Dropout(pc31_dropout_rate))
    pc31_model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    pc31_optimizer = tf.keras.optimizers.Adam(learning_rate=pc31_learning_rate)
    pc31_model.compile(optimizer=pc31_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return pc31_model

# Display model summary
if st.sidebar.button('Show Model Summary'):
    pc31_model = build_custom_model()
    pc31_model.summary(print_fn=lambda x: st.text(x))

# Accuracy and Loss Plot
def plot_metrics(pc31_history):
    pc31_fig, pc31_ax = plt.subplots(1, 2, figsize=(12, 5))
    pd.DataFrame(pc31_history.history)[['accuracy', 'val_accuracy']].plot(ax=pc31_ax[0])
    pd.DataFrame(pc31_history.history)[['loss', 'val_loss']].plot(ax=pc31_ax[1])
    pc31_ax[0].set_title('Accuracy')
    pc31_ax[1].set_title('Loss')
    st.pyplot(pc31_fig)

# Confusion Matrix Plot
def plot_confusion_matrix(pc31_y_true, pc31_y_pred, pc31_title):
    pc31_cm = confusion_matrix(pc31_y_true, pc31_y_pred)
    pc31_fig, pc31_ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(pc31_cm, annot=True, fmt='d', cmap='Blues', ax=pc31_ax)
    plt.title(pc31_title)
    st.pyplot(pc31_fig)

# Precision, Recall, and F1-Score Plot
def plot_classification_report(pc31_y_true, pc31_y_pred):
    pc31_report = classification_report(pc31_y_true, pc31_y_pred, output_dict=True)
    pc31_df_report = pd.DataFrame(pc31_report).transpose().iloc[:-3, :3]
    pc31_fig, pc31_ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=pc31_df_report, x=pc31_df_report.index, y='f1-score', color='skyblue', label='F1-Score')
    sns.barplot(data=pc31_df_report, x=pc31_df_report.index, y='precision', color='lightgreen', label='Precision')
    sns.barplot(data=pc31_df_report, x=pc31_df_report.index, y='recall', color='salmon', label='Recall')
    plt.title('Precision, Recall, and F1-Score')
    plt.legend()
    st.pyplot(pc31_fig)

# Class Distribution Plot
def plot_class_distribution(pc31_y):
    pc31_fig, pc31_ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=pc31_y)
    plt.title('Class Distribution')
    st.pyplot(pc31_fig)

# Model training and evaluation
if st.button('Train and Evaluate Model'):
    pc31_model = build_custom_model()

    # Train the model
    pc31_history = pc31_model.fit(X_train, y_train, epochs=pc31_epochs, batch_size=pc31_batch_size, validation_data=(X_val, y_val))

    # Plot metrics
    plot_metrics(pc31_history)

    # Predict and evaluate
    pc31_y_pred = (pc31_model.predict(X_val) > 0.5).astype('int32')
    accuracy = accuracy_score(y_val, pc31_y_pred)
    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix and Classification Report
    plot_confusion_matrix(y_val, pc31_y_pred, 'Confusion Matrix')
    plot_classification_report(y_val, pc31_y_pred)

    # Class distribution
    plot_class_distribution(y_val)
