import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Function to create synthetic OEE data
def create_synthetic_oee_data(num_records):
    np.random.seed(42)
    
    data = {
        'date': pd.date_range(start='2023-01-01', periods=num_records, freq='D'),
        'productivity': np.random.uniform(60, 100, num_records),  # In percentage
        'efficiency': np.random.uniform(70, 100, num_records),  # In percentage
        'quality': np.random.uniform(80, 100, num_records),  # In percentage
        'cycle_time': np.random.uniform(30, 120, num_records),  # In seconds
        'throughput': np.random.randint(100, 500, num_records),  # Number of units
        'yield': np.random.uniform(80, 100, num_records),  # In percentage
        'scrap_rate': np.random.uniform(0, 5, num_records),  # In percentage
        'downtime': np.random.uniform(0, 10, num_records),  # In hours
    }
    
    df = pd.DataFrame(data)
    return df

# Load and cache example data
@st.cache_data
def load_example_data():
    return create_synthetic_oee_data(365)

# Load uploaded data
def load_uploaded_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to calculate average OEE
def calculate_average_oee(data):
    oee_columns = ['productivity', 'efficiency', 'quality']
    data['oee_average'] = data[oee_columns].mean(axis=1)
    return data['oee_average'].mean()

# Function to display home page
def display_home():
    st.title("Welcome to the OEE Improvement App")
    st.write("""
    This app is designed to help you analyze and improve Overall Equipment Effectiveness (OEE) metrics. OEE is a crucial 
    metric for measuring the efficiency of manufacturing processes. It combines three key factors: Productivity, 
    Efficiency, and Quality.

    **Features:**
    - Use example data to explore the app's functionalities.
    - Upload your own CSV file containing OEE data.
    - View data summaries and statistics.
    - Generate interactive visualizations to identify trends and patterns.

    **How to Use:**
    - Navigate to the "Use Example Data" tab to see pre-loaded synthetic data.
    - Or, go to the "Upload Your Own Data" tab to upload a CSV file with your data.
    - Use the sidebar to switch between data preview and visualizations.

    **Data Requirements:**
    - The data should include the following columns: `date`, `productivity`, `efficiency`, `quality`, `cycle_time`, `throughput`, `yield`, `scrap_rate`, `downtime`.
    """)

    
    # Calculate and display average OEE
    example_data = load_example_data()
    avg_oee = calculate_average_oee(example_data)
    st.write(f"### Average OEE Value (Example Data): {avg_oee:.2f}%")

# Function to display data page
def display_data(data):
    st.write("## Data Preview")
    st.write(data.head())
    
    st.write("## Data Summary")
    st.write(data.describe())

# Function to display visualizations page
def display_visualizations(data):
    st.write("# Data Analysis")
    st.write("## Average OEE")
    st.write("The average OEE of the data is: ", calculate_average_oee(data))
    st.write("## Data Visualizations")
    # Custom layout for Plotly
    layout = go.Layout(
        template="plotly_dark",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        margin=dict(l=40, r=40, b=40, t=40),
        hovermode="closest"
    )
    
    # Productivity Over Time
    fig = go.Figure(data=go.Scatter(x=data['date'], y=data['productivity'], mode='lines+markers', name='Productivity'), layout=layout)
    fig.update_layout(title='Productivity Over Time')
    st.plotly_chart(fig)
    
    # Efficiency Over Time
    fig = go.Figure(data=go.Scatter(x=data['date'], y=data['efficiency'], mode='lines+markers', name='Efficiency'), layout=layout)
    fig.update_layout(title='Efficiency Over Time')
    st.plotly_chart(fig)
    
    # Quality Over Time
    fig = go.Figure(data=go.Scatter(x=data['date'], y=data['quality'], mode='lines+markers', name='Quality'), layout=layout)
    fig.update_layout(title='Quality Over Time')
    st.plotly_chart(fig)
    
    # Scatter plot for cycle time vs throughput
    fig = px.scatter(data, x='cycle_time', y='throughput', title='Cycle Time vs Throughput', template='plotly_dark')
    st.plotly_chart(fig)

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Home", "Use Example Data", "Upload Your Own Data"])
    
    if option == "Home":
        display_home()
    else:
        if option == "Use Example Data":
            data = load_example_data()
        else:
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file is not None:
                data = load_uploaded_data(uploaded_file)
            else:
                st.write("Please upload a CSV file to proceed.")
                return
        
        st.sidebar.title("Data Analysis")
        analysis_option = st.sidebar.selectbox("Choose Analysis", ["Data Preview", "Data Insights and Visualizations"])
        
        if analysis_option == "Data Preview":
            display_data(data)
        else:
            display_visualizations(data)

if __name__ == "__main__":
    main()
