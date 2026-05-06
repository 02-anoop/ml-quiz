import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="ML Studio", layout="wide")

# Basic styling for tabs to fix visibility in dark mode, keeping it simple and minimal
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #2b313e;
        color: #e2e8f0;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("Machine Learning Studio")
st.write("A simple dashboard for data analysis and modeling.")

tab_data, tab_model, tab_results = st.tabs(["1. Data & EDA", "2. Model Config", "3. Results"])

if 'df' not in st.session_state:
    st.session_state.df = None
if 'task' not in st.session_state:
    st.session_state.task = "Classification"
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'target_names' not in st.session_state:
    st.session_state.target_names = None

@st.cache_data
def load_builtin_data(name):
    if name == "Iris":
        data = datasets.load_iris()
        task = "Classification"
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
        task = "Classification"
    elif name == "Wine":
        data = datasets.load_wine()
        task = "Classification"
    elif name == "Digits":
        data = datasets.load_digits()
        task = "Classification"
    elif name == "Moons":
        X, y = datasets.make_moons(n_samples=500, noise=0.2, random_state=42)
        df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
        df['Target'] = y
        return df, ['Feature_1', 'Feature_2'], "Classification", "Target", None
    elif name == "Circles":
        X, y = datasets.make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
        df['Target'] = y
        return df, ['Feature_1', 'Feature_2'], "Classification", "Target", None
    elif name == "Diabetes":
        data = datasets.load_diabetes()
        task = "Regression"
    else:
        data = datasets.fetch_california_housing()
        task = "Regression"
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target
    return df, data.feature_names, task, "Target", getattr(data, 'target_names', None)

with tab_data:
    st.header("Data Loading")
    col1, col2 = st.columns(2)
    
    with col1:
        data_source = st.radio("Source", ["Built-in", "Upload CSV"], horizontal=True)
        if data_source == "Built-in":
            dataset_name = st.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine", "Digits", "Moons", "Circles", "Diabetes", "California Housing"])
            df, feature_cols, task, target_col, target_names = load_builtin_data(dataset_name)
            st.session_state.df = df
            st.session_state.feature_cols = feature_cols
            st.session_state.task = task
            st.session_state.target_col = target_col
            st.session_state.target_names = target_names
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
            else:
                st.info("Upload a CSV to begin")
                st.session_state.df = None

    with col2:
        if st.session_state.df is not None and data_source == "Upload CSV":
            df = st.session_state.df
            all_cols = df.columns.tolist()
            target_col = st.selectbox("Target Column", all_cols, index=len(all_cols)-1)
            task = st.radio("Task Type", ["Classification", "Regression"], horizontal=True)
            st.session_state.target_col = target_col
            st.session_state.feature_cols = [c for c in all_cols if c != target_col]
            st.session_state.task = task
            st.session_state.target_names = None

    if st.session_state.df is not None:
        df = st.session_state.df
        feature_cols = st.session_state.feature_cols
        target_col = st.session_state.target_col
        task = st.session_state.task

        st.divider()
        st.header("EDA (Exploratory Data Analysis)")
        
        st.write("Data Preview")
        st.dataframe(df.head(10))
        
        st.write("Visualizations")
        
        # Row 1: Target Distribution and Correlation
        c1, c2 = st.columns(2)
        with c1:
            st.write("Target Variable Distribution")
            if task == "Classification":
                target_counts = df[target_col].value_counts()
                st.bar_chart(target_counts)
            else:
                st.bar_chart(np.histogram(df[target_col], bins=20)[0])
                
        with c2:
            st.write("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig_corr = px.imshow(numeric_df.corr(), aspect="auto", color_continuous_scale='Blues')
                st.plotly_chart(fig_corr, use_container_width=True)

        # Row 2: Histogram and Boxplot
        c3, c4 = st.columns(2)
        with c3:
            st.write("Feature Histogram")
            feat_hist = st.selectbox("Feature for Histogram", feature_cols)
            fig_hist = px.histogram(df, x=feat_hist, color=target_col if task=='Classification' else None)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c4:
            st.write("Feature Box Plot")
            feat_box = st.selectbox("Feature for Box Plot", feature_cols, index=min(1, len(feature_cols)-1))
            fig_box = px.box(df, y=feat_box, x=target_col if task=='Classification' else None)
            st.plotly_chart(fig_box, use_container_width=True)

        # Row 3: Scatter Plot
        st.write("Scatter Plot")
        if len(feature_cols) >= 2:
            fx = st.selectbox("X Axis", feature_cols, index=0)
            fy = st.selectbox("Y Axis", feature_cols, index=1)
        else:
            fx = feature_cols[0]
            fy = target_col
        fig_scatter = px.scatter(df, x=fx, y=fy, color=target_col if task=='Classification' else None)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab_model:
    if st.session_state.df is None:
        st.warning("Load data first.")
    else:
        st.header("Model Setup")
        task = st.session_state.task
        
        if task == "Classification":
            algos = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
        else:
            algos = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]
            
        c1, c2 = st.columns(2)
        with c1:
            algorithm = st.selectbox("Algorithm", algos)
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            
        with c2:
            model = None
            if algorithm == "Logistic Regression":
                C = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, max_iter=200)
            elif algorithm == "KNN":
                k = st.slider("K Neighbors", 1, 20, 5)
                model = KNeighborsClassifier(n_neighbors=k) if task=="Classification" else KNeighborsRegressor(n_neighbors=k)
            elif algorithm == "Decision Tree":
                md = st.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=md) if task=="Classification" else DecisionTreeRegressor(max_depth=md)
            elif algorithm == "Random Forest":
                n_est = st.slider("Estimators", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_est) if task=="Classification" else RandomForestRegressor(n_estimators=n_est)
            elif algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm in ["Ridge", "Lasso"]:
                alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                model = Ridge(alpha=alpha) if algorithm=="Ridge" else Lasso(alpha=alpha)

        st.session_state.model = model
        st.session_state.algorithm = algorithm
        st.session_state.test_size = test_size

with tab_results:
    if st.session_state.df is None or 'model' not in st.session_state:
        st.warning("Configure data and model first.")
    else:
        if st.button("Train Model", type="primary"):
            df = st.session_state.df.dropna()
            X = df[st.session_state.feature_cols]
            y = df[st.session_state.target_col]
            task = st.session_state.task
            
            if task == "Classification" and y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y)
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = st.session_state.model
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            st.divider()
            st.header("Results")
            
            c1, c2 = st.columns(2)
            
            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                with c1:
                    st.write(f"**Accuracy:** {acc:.4f}")
                    st.write("Confusion Matrix:")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with c2:
                    st.write("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                    
                st.write("Prediction Confidence Distribution")
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test_s)
                    # Histogram of max probabilities (confidence)
                    conf = np.max(probs, axis=1)
                    fig_conf = px.histogram(x=conf, nbins=20, labels={'x': 'Prediction Confidence (Probability)', 'y': 'Count'})
                    st.plotly_chart(fig_conf, use_container_width=True)
                else:
                    st.info("This model does not output probabilities.")
                    
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                res = y_test - y_pred
                
                with c1:
                    st.write(f"**MSE:** {mse:.4f} | **R2 Score:** {r2:.4f}")
                    st.write("Residual Distribution (Error)")
                    fig_res_dist = px.histogram(x=res, nbins=30, labels={'x': 'Residual Error', 'y': 'Count'})
                    st.plotly_chart(fig_res_dist, use_container_width=True)
                    
                with c2:
                    st.write("Actual vs Predicted")
                    fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                    min_val = min(np.min(y_test), np.min(y_pred))
                    max_val = max(np.max(y_test), np.max(y_pred))
                    fig_scatter.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='red', dash='dash'))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.write("Residuals vs Predicted")
                fig_res_scatter = px.scatter(x=y_pred, y=res, labels={'x': 'Predicted Value', 'y': 'Residual Error'})
                fig_res_scatter.add_shape(type='line', x0=np.min(y_pred), y0=0, x1=np.max(y_pred), y1=0, line=dict(color='red', dash='dash'))
                st.plotly_chart(fig_res_scatter, use_container_width=True)
                    
            st.write("Feature Importances / Coefficients")
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({'Feature': st.session_state.feature_cols, 'Importance': model.feature_importances_})
                st.bar_chart(imp.set_index('Feature'))
            elif hasattr(model, 'coef_'):
                # Handle multi-dimensional coefficients (e.g. multi-class Logistic Regression)
                coef_vals = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                coef = pd.DataFrame({'Feature': st.session_state.feature_cols, 'Coef': coef_vals})
                st.bar_chart(coef.set_index('Feature'))
