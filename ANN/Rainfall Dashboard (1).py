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
import plotly.graph_objects as go

st.set_page_config(page_title='Rainfall Prediction Dashboard', layout='wide')

st.title('Rainfall Prediction Dashboard')

# Load the dataset
@st.cache_data  # Cache data to improve performance
def load_data(file_path):
    df = pd.read_csv(file_path, na_filter=False)
    return df

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

# location_coords dictionary (MAKE SURE IT'S COMPLETE)
location_coords = {
    'Albury': (-36.0737, 146.9135),
    'BadgerysCreek': (-33.8800, 150.7440),
    'Cobar': (-31.4941, 145.8353),
    'CoffsHarbour': (-30.2964, 153.1142),
    'Moree': (-29.4635, 149.8449),
    'NorfolkIsland': (-29.0341, 167.9547),
    'Penrith': (-33.7557, 150.6723),
    'Richmond': (-33.6007, 150.7497),
    'Sydney': (-33.8688, 151.2093),
    'SydneyAirport': (-33.9461, 151.1772),
    'SydneyOlympicPark': (-33.8447, 151.0694),
    'Williamtown': (-32.8150, 151.8428),
    'Wollongong': (-34.4278, 150.8931),
    'Canberra': (-35.2809, 149.1300),
    'Tuggeranong': (-35.4333, 149.0667),
    'MountGinini': (-35.5217, 148.7758),
    'Ballarat': (-37.5622, 143.8503),
    'Bendigo': (-36.7578, 144.2809),
    'Sale': (-38.1067, 147.0656),
    'MelbourneAirport': (-37.6733, 144.8433),
    'Melbourne': (-37.8136, 144.9631),
    'MelbourneCBD': (-37.8178, 144.9659),
    'Mildura': (-34.1872, 142.1578),
    'Nhil': (-36.6487, 141.6511),
    'Portland': (-38.3433, 141.6033),
    'Watsonia': (-37.7167, 145.0833),
    'Dartmoor': (-37.8333, 141.2333),
    'Brisbane': (-27.4698, 153.0251),
    'Cairns': (-16.9186, 145.7781),
    'GoldCoast': (-28.0167, 153.4000),
    'Townsville': (-19.2589, 146.8169),
    'Adelaide': (-34.9285, 138.6007),
    'MountGambier': (-37.8274, 140.7817),
    'Nuriootpa': (-34.4703, 138.9919),
    'Woomera': (-31.1667, 136.8167),
    'Albany': (-35.0275, 117.8836),
    'Witchcliffe': (-34.1167, 115.0500),
    'PearceRAAF': (-31.6667, 116.0333),
    'PerthAirport': (-31.9403, 115.9669),
    'Perth': (-31.9505, 115.8605),
    'SalmonGums': (-32.9833, 121.7833),
    'Walpole': (-34.9778, 116.7333),
    'Hobart': (-42.8821, 147.3272),
    'Launceston': (-41.4419, 147.1450),
    'AliceSprings': (-23.6980, 133.8807),
    'Darwin': (-12.4634, 130.8456),
    'Katherine': (-14.4667, 132.2667),
    'Uluru': (-25.3444, 131.0369)
}

# Correct way to create map_data
map_data = pd.DataFrame(list(location_coords.items()), columns=['Location', 'Coordinates'])
map_data[['Latitude', 'Longitude']] = pd.DataFrame(map_data['Coordinates'].tolist(), index=map_data.index)
map_data.drop('Coordinates', axis=1, inplace=True)

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

# Australia Map
st.subheader("Rainfall Map of Australia")

# Merge with average rainfall data
map_data = pd.merge(map_data, avg_rainfall, on='Location', how='left')
map_data['Rainfall'].fillna(0, inplace=True)  # Fill NaN rainfall values with 0

# Create the map
fig_map = go.Figure(go.Scattergeo(
    lon=map_data['Longitude'],
    lat=map_data['Latitude'],
    text=map_data['Location'] + ': ' + map_data['Rainfall'].astype(str) + ' mm',
    mode='markers',
    marker=dict(
        size=map_data['Rainfall'] * 5,  # Adjust size multiplier as needed
        color=map_data['Rainfall'],
        colorscale='Viridis',
        colorbar_title="Average Rainfall (mm)"
    )
))

fig_map.update_layout(
    title_text='Average Rainfall by Location',
    geo=dict(
        scope='australia',
        landcolor='lightgreen',
        showocean=True,
        oceancolor="lightblue",
        projection_type='natural earth'
    )
)  # ✅ Fixed


st.plotly_chart(fig_map)

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
if st.button('Train Model'):
    with st.spinner('Training...'):
        pc31_model = build_custom_model()
        pc31_history = pc31_model.fit(
            x=X_train, y=y_train,
            validation_data=(X_val, y_val),
            batch_size=pc31_batch_size, epochs=pc31_epochs, verbose=0)
        st.success('Model trained successfully!')
        plot_metrics(pc31_history)

        # Predictions and Confusion Matrices
        pc31_y_train_pred = (pc31_model.predict(X_train) > 0.5).astype("int32")
        pc31_y_val_pred = (pc31_model.predict(X_val) > 0.5).astype("int32")

        pc31_col1, pc31_col2 = st.columns(2)
        with pc31_col1:
            plot_confusion_matrix(y_train, pc31_y_train_pred, 'Training Set Confusion Matrix')
        with pc31_col2:
            plot_confusion_matrix(y_val, pc31_y_val_pred, 'Validation Set Confusion Matrix')

        plot_classification_report(y_val, pc31_y_val_pred)
        plot_class_distribution(y_train)
