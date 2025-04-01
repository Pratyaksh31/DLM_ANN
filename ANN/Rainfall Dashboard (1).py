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

# Set page config
st.set_page_config(page_title='Rainfall Prediction Dashboard', layout='wide')

# Title
st.title('Rainfall Prediction Dashboard')

# Correct raw URL of the CSV file from GitHub
file_url = 'https://raw.githubusercontent.com/Pratyaksh31/DLM_ANN/main/ANN/weatherAUS.csv'

# Load the dataset (directly from URL)
@st.cache_data  # Cache data to improve performance
def load_data(file_url):
    df = pd.read_csv(file_url, na_filter=False)
    return df

# Load the CSV data into a DataFrame
df = load_data(file_url)

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

# Location coordinates dictionary
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

# Creating map_data DataFrame
map_data = pd.DataFrame(list(location_coords.items()), columns=['Location', 'Coordinates'])
map_data[['Latitude', 'Longitude']] = pd.DataFrame(map_data['Coordinates'].tolist(), index=map_data.index)
map_data.drop('Coordinates', axis=1, inplace=True)

# Sidebar Filters
st.sidebar.header("Filters")
locations = df['Location'].unique()
selected_locations = st.sidebar.multiselect("Select Locations", locations, default=locations)
filtered_df = df[df['Location'].isin(selected_locations)]

# Date Range Filter
min_date = filtered_df['Date'].min().date()
max_date = filtered_df['Date'].max().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

# Numerical Filters
st.sidebar.header("Numerical Filters")
min_temp_threshold = st.sidebar.slider("Min Temp Threshold", float(df['MinTemp'].min()), float(df['MinTemp'].max()), float(df['MinTemp'].min()))
max_temp_threshold = st.sidebar.slider("Max Temp Threshold", float(df['MaxTemp'].min()), float(df['MaxTemp'].max()), float(df['MaxTemp'].max()))
filtered_df = filtered_df[(filtered_df['MinTemp'] >= min_temp_threshold) & (filtered_df['MaxTemp'] <= max_temp_threshold)]

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Visualizations
avg_rainfall = filtered_df.groupby('Location')['Rainfall'].mean().reset_index()

# Rainfall Distribution
st.subheader("Rainfall Distribution")
fig_rainfall = px.histogram(filtered_df, x="Rainfall", title="Rainfall Distribution")
st.plotly_chart(fig_rainfall)

fig_avg_rainfall = px.bar(avg_rainfall, x='Location', y='Rainfall', title='Average Rainfall by Location')
st.plotly_chart(fig_avg_rainfall)

# Australia Map
st.subheader("Rainfall Map of Australia")
map_data = pd.merge(map_data, avg_rainfall, on='Location', how='left')
map_data['Rainfall'].fillna(0, inplace=True)

fig_map = go.Figure(go.Scattergeo(
    lon=map_data['Longitude'],
    lat=map_data['Latitude'],
    text=map_data['Location'] + ': ' + map_data['Rainfall'].astype(str) + ' mm',
    mode='markers',
    marker=dict(
        size=map_data['Rainfall'] * 5,  
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
)

st.plotly_chart(fig_map)
