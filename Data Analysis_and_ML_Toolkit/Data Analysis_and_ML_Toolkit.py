import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Function to create synthetic data (sample data for testing)
def create_synthetic_data(num_records):
    np.random.seed(42)
    data = {
        'feature_1': np.random.randint(1, 100, num_records),
        'feature_2': np.random.uniform(1.0, 10.0, num_records),
        'feature_3': np.random.normal(0.0, 1.0, num_records),
        'target': np.random.choice([0, 1], num_records)
    }
    df = pd.DataFrame(data)
    return df

# Load and cache example data
@st.cache_data
def load_example_data():
    return create_synthetic_data(100)

# Load uploaded data
def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    
    return df

# Function to display home page with acknowledgements and developer info
def display_home():
    st.title("Welcome to the General Data Analysis App")
    st.write("""
    This app allows you to analyze various datasets using different models and visualize the results.
    You can use the example data provided or upload your own dataset.
    """)
    st.write("""
    ### How to Use
    
    1. **Navigation:**
       - Use the sidebar on the left to navigate between different sections: Home, Use Example Data, and Upload Your Own Data.
    
    2. **Data Analysis:**
       - After selecting a dataset, choose from options like Data Preview, Data Visualizations, Supervised Model, or Unsupervised Model in the sidebar to analyze the data.
    
    3. **Model Options:**
       - Adjust model parameters such as Test Size, Max Depth, Number of Estimators, etc., based on your analysis needs.
    
    4. **Acknowledgements:**
       - The app uses Streamlit, Pandas, NumPy, Plotly, Matplotlib, Seaborn, and Scikit-learn libraries for data analysis and visualization.
       
    5. **Developer Info:**
       - This app was developed by Ahmad Bin Sadiq and Muhammad Bin Sadiq, two brothers passionate about data science and machine learning.
       - For more details and to contribute, visit the [GitHub repository](https://github.com/ahmadbinsadiq/Streamlit-Apps.git).
    """)

# Function to display data preview with info
def display_data(data):
    st.write("## Data Preview")
    st.write(data.head())
    st.write("## Data Info")
    st.write(data.info())
    st.write("## Data Summary")
    st.write(data.describe())

# Function to display visualizations page
def display_visualizations(data):
    st.write("## Data Visualizations")
    
    # Custom colors for plots
    palette = sns.color_palette("husl", len(data.columns))
    
    # Correlation Matrix
    corr_matrix = data.corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
    st.write("### Correlation Matrix")
    st.pyplot(fig_corr)

    # Pairplot for numerical columns
    st.write("### Pair Plot")
    hue_col = st.selectbox("Select Hue Column for Pair Plot", data.columns)
    fig_pairplot = sns.pairplot(data, hue=hue_col, diag_kind="kde", plot_kws={"alpha": 0.6})
    st.pyplot(fig_pairplot.fig)

    # Histograms for each column
    st.write("### Histograms")
    fig_hist, axes_hist = plt.subplots(len(data.columns)//4 + 1, 4, figsize=(20, 15))
    axes_hist = axes_hist.flatten()
    for i, col in enumerate(data.columns):
        sns.histplot(data[col], bins=10, kde=True, color=palette[i], edgecolor='black', ax=axes_hist[i])
        axes_hist[i].set_title(f'Histogram of {col}')
        axes_hist[i].set_xlabel(col)
        axes_hist[i].set_ylabel('Frequency')
    fig_hist.tight_layout()
    st.pyplot(fig_hist)

    # Boxplot for each column
    st.write("### Boxplots")
    fig_box, axes_box = plt.subplots(len(data.columns)//4 + 1, 4, figsize=(20, 15))
    axes_box = axes_box.flatten()
    for i, col in enumerate(data.columns):
        sns.boxplot(x=data[col], ax=axes_box[i])
        axes_box[i].set_title(f'Boxplot of {col}')
        axes_box[i].set_xlabel(col)
    fig_box.tight_layout()
    st.pyplot(fig_box)

# Function to train and display results of a supervised model
def display_supervised_model(data, model_type, target_col, test_size, **params):
    st.write(f"## {model_type} Model Results")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if model_type == 'Linear Regression':
        model = LinearRegression(**params)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(**params)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(**params)
    elif model_type == 'SVM':
        model = SVC(**params)
    elif model_type == 'Decision Tree Classifier':
        model = DecisionTreeClassifier(**params)
    elif model_type == 'Decision Tree Regressor':
        model = DecisionTreeRegressor(**params)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if model_type in ['Linear Regression', 'Gradient Boosting', 'Decision Tree Regressor']:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R² Score: {r2}")
    else:
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred), language='text')
        
    # Input data columns for prediction
    st.write("### Predict on New Data")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Input {col}", value=float(X[col].mean()))
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"Prediction for input data: {prediction[0]}")

# Function to train and display results of an unsupervised model
def display_unsupervised_model(data, model_type, **params):
    st.write(f"## {model_type} Model Results")
    
    if model_type == 'KMeans':
        model = KMeans(**params)
    elif model_type == 'DBSCAN':
        model = DBSCAN(**params)
    elif model_type == 'Agglomerative Clustering':
        model = AgglomerativeClustering(**params)
    elif model_type == 'PCA':
        model = PCA(**params)
    elif model_type == 'TSNE':
        model = TSNE(**params)
    
    if model_type in ['PCA', 'TSNE']:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        components = model.fit_transform(scaled_data)
        fig = px.scatter(components, x=0, y=1, title=f'{model_type} Components', template='plotly_dark')
        st.plotly_chart(fig)
    else:
        labels = model.fit_predict(data)
        if model_type == 'KMeans':
            st.write("Cluster Centers:")
            st.write(model.cluster_centers_)
        fig = px.scatter_matrix(data, dimensions=data.columns, color=labels, title=f'{model_type} Clustering', template='plotly_dark')
        st.plotly_chart(fig)

# Function to get parameter ranges based on data size
def get_param_ranges(data):
    param_ranges = {
        'max_depth': (1, min(20, len(data))),
        'n_estimators': (10, min(100, len(data))),
        'n_clusters': (2, min(10, len(data))),
        'eps': (0.1, 1.0),
        'min_samples': (1, 10),
        'n_components': (2, min(len(data.columns), 10)),
        'perplexity': (5, 50)
    }
    return param_ranges

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
                if data is None:
                    return
            else:
                st.write("Please upload a CSV file to proceed.")
                return
        
        st.sidebar.title("Data Analysis")
        analysis_option = st.sidebar.selectbox("Choose Analysis", ["Data Preview", "Data Visualizations", "Supervised Model", "Unsupervised Model"])
        
        if analysis_option == "Data Preview":
            display_data(data)
        elif analysis_option == "Data Visualizations":
            display_visualizations(data)
        elif analysis_option == "Supervised Model":
            st.sidebar.title("Supervised Model Options")
            model_type = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest", "Gradient Boosting", "SVM", "Decision Tree Classifier", "Decision Tree Regressor"])
            target_col = st.sidebar.selectbox("Choose Target Column", data.columns)
            test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
            
            param_ranges = get_param_ranges(data)
            if model_type == "Random Forest":
                n_estimators = st.sidebar.slider("Number of Estimators", *param_ranges['n_estimators'])
                max_depth = st.sidebar.slider("Max Depth", *param_ranges['max_depth'])
                params = {"n_estimators": n_estimators, "max_depth": max_depth}
            else:
                params = {}
            
            display_supervised_model(data, model_type, target_col, test_size, **params)
        elif analysis_option == "Unsupervised Model":
            st.sidebar.title("Unsupervised Model Options")
            model_type = st.sidebar.selectbox("Choose Model", ["KMeans", "DBSCAN", "Agglomerative Clustering", "PCA", "TSNE"])
            
            param_ranges = get_param_ranges(data)
            if model_type == "KMeans":
                n_clusters = st.sidebar.slider("Number of Clusters", *param_ranges['n_clusters'])
                params = {"n_clusters": n_clusters}
            elif model_type == "DBSCAN":
                eps = st.sidebar.slider("Epsilon (eps)", *param_ranges['eps'])
                min_samples = st.sidebar.slider("Minimum Samples", *param_ranges['min_samples'])
                params = {"eps": eps, "min_samples": min_samples}
            elif model_type == "Agglomerative Clustering":
                n_clusters = st.sidebar.slider("Number of Clusters", *param_ranges['n_clusters'])
                params = {"n_clusters": n_clusters}
            elif model_type == "PCA":
                n_components = st.sidebar.slider("Number of Components", *param_ranges['n_components'])
                params = {"n_components": n_components}
            elif model_type == "TSNE":
                n_components = st.sidebar.slider("Number of Components", *param_ranges['n_components'])
                perplexity = st.sidebar.slider("Perplexity", *param_ranges['perplexity'])
                params = {"n_components": n_components, "perplexity": perplexity}
        
            display_unsupervised_model(data, model_type, **params)

if __name__ == "__main__":
    main()
