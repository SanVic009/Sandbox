import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
from scipy import stats
import hashlib
import json
import time
import base64
from io import BytesIO
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- Advanced App Configuration ---
st.set_page_config(
    page_title="🌾 AISACROP Pro - Advanced Agricultural Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/aisacrop',
        'Report a bug': "https://github.com/aisacrop/issues",
        'About': "# AISACROP Pro\n Advanced Agricultural Intelligence Platform v2.0"
    }
)

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Custom card styling */
    .custom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
    }
    
    /* Alert styling */
    .alert-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .alert-critical {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
    }
    
    .alert-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    
    .alert-info {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px 8px 0px 0px;
        padding: 0px 24px;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Enhanced Session State ---
def init_session_state():
    """Initialize all session state variables"""
    
    # User authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
    
    # Data management
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
        st.session_state.processed_data = None
        st.session_state.ml_models = {}
    
    # User preferences
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'light',
            'language': 'en',
            'notifications': True,
            'auto_save': True,
            'dashboard_layout': 'default',
            'chart_preferences': {
                'color_scheme': 'viridis',
                'animation': True,
                '3d_charts': False
            }
        }
    
    # Analytics tracking
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'session_start': datetime.now(),
            'actions_count': 0,
            'queries_history': [],
            'exports_count': 0
        }
    
    # Collaboration features
    if 'collaboration' not in st.session_state:
        st.session_state.collaboration = {
            'shared_dashboards': [],
            'comments': [],
            'annotations': {}
        }
    
    # Alert system
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
        st.session_state.alert_rules = []
    
    # Report generation
    if 'reports' not in st.session_state:
        st.session_state.reports = []

init_session_state()

# --- Authentication System ---
class AuthenticationSystem:
    """Handle user authentication and authorization"""
    
    @staticmethod
    def hash_password(password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_user(username, password):
        """Verify user credentials"""
        # Simulated user database
        users = {
            'admin': {
                'password': AuthenticationSystem.hash_password('admin123'),
                'role': 'administrator',
                'permissions': ['read', 'write', 'delete', 'admin']
            },
            'analyst': {
                'password': AuthenticationSystem.hash_password('analyst123'),
                'role': 'analyst',
                'permissions': ['read', 'write']
            },
            'viewer': {
                'password': AuthenticationSystem.hash_password('viewer123'),
                'role': 'viewer',
                'permissions': ['read']
            },
            'demo': {
                'password': AuthenticationSystem.hash_password('demo'),
                'role': 'demo_user',
                'permissions': ['read', 'write']
            }
        }
        
        if username in users:
            if users[username]['password'] == AuthenticationSystem.hash_password(password):
                return True, users[username]['role'], users[username]['permissions']
        return False, None, None
    
    @staticmethod
    def login_form():
        """Display login form"""
        st.markdown("""
        <div style='max-width: 400px; margin: auto; padding: 2rem;'>
            <h2 style='text-align: center; color: #667eea;'>🌾 AISACROP Pro Login</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.markdown("### Welcome Back!")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("Remember me for 30 days")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    submit = st.form_submit_button("🔐 Login", use_container_width=True)
                with col_b:
                    demo_login = st.form_submit_button("🎮 Demo Access", use_container_width=True)
                
                if submit:
                    if username and password:
                        success, role, permissions = AuthenticationSystem.verify_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_role = role
                            st.session_state.user_permissions = permissions
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.warning("Please enter both username and password")
                
                if demo_login:
                    st.session_state.authenticated = True
                    st.session_state.username = "demo"
                    st.session_state.user_role = "demo_user"
                    st.session_state.user_permissions = ['read', 'write']
                    st.success("✅ Demo access granted!")
                    time.sleep(1)
                    st.rerun()
            
            st.markdown("""
            ---
            **Demo Credentials:**
            - Username: `demo` | Password: `demo`
            - Username: `admin` | Password: `admin123`
            - Username: `analyst` | Password: `analyst123`
            """)

# --- Data Generation & Processing ---
class DataGenerator:
    """Generate synthetic agricultural data for testing"""
    
    @staticmethod
    def generate_sample_data(num_records=1000):
        """Generate synthetic crop yield data"""
        np.random.seed(42)
        
        states = ['Maharashtra', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Gujarat', 
                  'Rajasthan', 'Madhya Pradesh', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh']
        crops = ['Wheat', 'Rice', 'Cotton', 'Sugarcane', 'Maize', 'Soybean', 
                 'Groundnut', 'Pulses', 'Barley', 'Millet']
        
        data = []
        for _ in range(num_records):
            state = np.random.choice(states)
            crop = np.random.choice(crops)
            year = np.random.randint(2010, 2024)
            
            # Base yield with seasonal variation
            base_yield = np.random.uniform(1, 5)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * year / 10)
            weather_impact = np.random.uniform(0.8, 1.2)
            
            yield_value = base_yield * seasonal_factor * weather_impact
            
            # Additional features
            rainfall = np.random.uniform(500, 1500)
            temperature = np.random.uniform(20, 35)
            soil_quality = np.random.uniform(0.5, 1.0)
            fertilizer_usage = np.random.uniform(50, 200)
            irrigation_percentage = np.random.uniform(0.3, 1.0)
            
            data.append({
                'State': state,
                'Crop': crop,
                'Crop_Year': year,
                'Yield': yield_value,
                'Rainfall_mm': rainfall,
                'Avg_Temperature': temperature,
                'Soil_Quality_Index': soil_quality,
                'Fertilizer_kg_per_ha': fertilizer_usage,
                'Irrigation_Coverage': irrigation_percentage,
                'Season': 'Kharif' if np.random.random() > 0.5 else 'Rabi',
                'District': f"{state}_District_{np.random.randint(1, 6)}"
            })
        
        return pd.DataFrame(data)

class AdvancedDataProcessor:
    """Advanced data processing and feature engineering"""
    
    @staticmethod
    def process_data(df):
        """Process and enhance data with additional features"""
        df = df.copy()
        
        # Sort by year
        df = df.sort_values(['State', 'Crop', 'Crop_Year'])
        
        # Calculate rolling averages
        df['Yield_MA_3'] = df.groupby(['State', 'Crop'])['Yield'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Calculate year-over-year growth
        df['Yield_Growth'] = df.groupby(['State', 'Crop'])['Yield'].pct_change()
        
        # Calculate z-scores for anomaly detection
        df['Yield_Zscore'] = df.groupby(['State', 'Crop'])['Yield'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Create productivity index
        if 'Fertilizer_kg_per_ha' in df.columns:
            df['Productivity_Index'] = df['Yield'] / (df['Fertilizer_kg_per_ha'] / 100)
        
        # Add categorical encoding for seasons
        if 'Season' in df.columns:
            df['Is_Kharif'] = (df['Season'] == 'Kharif').astype(int)
        
        return df
    
    @staticmethod
    def create_features_for_ml(df, target_col='Yield'):
        """Create features for machine learning"""
        feature_cols = []
        
        # Numerical features
        num_cols = ['Rainfall_mm', 'Avg_Temperature', 'Soil_Quality_Index', 
                    'Fertilizer_kg_per_ha', 'Irrigation_Coverage']
        for col in num_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # One-hot encode categorical variables
        if 'State' in df.columns:
            state_dummies = pd.get_dummies(df['State'], prefix='State')
            df = pd.concat([df, state_dummies], axis=1)
            feature_cols.extend(state_dummies.columns.tolist())
        
        if 'Crop' in df.columns:
            crop_dummies = pd.get_dummies(df['Crop'], prefix='Crop')
            df = pd.concat([df, crop_dummies], axis=1)
            feature_cols.extend(crop_dummies.columns.tolist())
        
        # Time features
        if 'Crop_Year' in df.columns:
            df['Year_Normalized'] = (df['Crop_Year'] - df['Crop_Year'].min()) / (df['Crop_Year'].max() - df['Crop_Year'].min())
            feature_cols.append('Year_Normalized')
        
        return df, feature_cols

# --- Machine Learning Models ---
class MLPipeline:
    """Advanced machine learning pipeline"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.scalers = {}
        self.performance_metrics = {}
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and compare performance"""
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            results[name] = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test)
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            results[name]['cv_score_mean'] = cv_scores.mean()
            results[name]['cv_score_std'] = cv_scores.std()
        
        self.performance_metrics = results
        return results
    
    def predict_future(self, model_name, X_future):
        """Make predictions using specified model"""
        if model_name in self.trained_models:
            return self.trained_models[model_name].predict(X_future)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        model = self.trained_models.get(model_name)
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None

# --- Visualization Components ---
class AdvancedVisualizations:
    """Create advanced interactive visualizations"""
    
    @staticmethod
    def create_3d_surface_plot(df, x_col, y_col, z_col, title="3D Surface Plot"):
        """Create 3D surface plot"""
        fig = go.Figure(data=[go.Surface(
            x=df[x_col].unique(),
            y=df[y_col].unique(),
            z=df.pivot_table(values=z_col, index=y_col, columns=x_col).values,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
            ),
            height=600
        )
        return fig
    
    @staticmethod
    def create_animated_timeline(df, value_col, category_col, time_col):
        """Create animated timeline chart"""
        fig = px.bar(df, x=time_col, y=value_col, color=category_col,
                     animation_frame=time_col,
                     range_y=[0, df[value_col].max() * 1.1])
        
        fig.update_layout(
            title="Animated Timeline Analysis",
            xaxis_title=time_col,
            yaxis_title=value_col,
            height=500
        )
        return fig
    
    @staticmethod
    def create_correlation_network(df, threshold=0.5):
        """Create correlation network diagram"""
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        # Create edges for correlation above threshold
        edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    edges.append({
                        'source': corr_matrix.columns[i],
                        'target': corr_matrix.columns[j],
                        'value': corr_matrix.iloc[i, j]
                    })
        
        return edges
    
    @staticmethod
    def create_dashboard_grid(metrics_dict):
        """Create a grid of metric cards"""
        cols = st.columns(4)
        for idx, (key, value) in enumerate(metrics_dict.items()):
            with cols[idx % 4]:
                st.metric(label=key, value=value['value'], delta=value.get('delta'))

# --- Report Generator ---
class ReportGenerator:
    """Generate comprehensive reports"""
    
    @staticmethod
    def generate_pdf_report(data, analysis_results, user_info):
        """Generate PDF report (simulated as markdown for now)"""
        report = f"""
# 🌾 AISACROP Agricultural Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Generated by:** {user_info['username']}
**Report ID:** {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}

---

## Executive Summary

This report provides comprehensive analysis of agricultural yield data across multiple regions and crops.

## Key Metrics

- **Total Records Analyzed:** {len(data)}
- **Time Period:** {data['Crop_Year'].min()} - {data['Crop_Year'].max()}
- **Regions Covered:** {data['State'].nunique()}
- **Crops Analyzed:** {data['Crop'].nunique()}

## Statistical Summary

### Overall Yield Statistics
- **Average Yield:** {data['Yield'].mean():.2f} tonnes/hectare
- **Maximum Yield:** {data['Yield'].max():.2f} tonnes/hectare
- **Minimum Yield:** {data['Yield'].min():.2f} tonnes/hectare
- **Standard Deviation:** {data['Yield'].std():.2f}

## Top Performing Regions

{data.groupby('State')['Yield'].mean().sort_values(ascending=False).head().to_string()}

## Machine Learning Model Performance

{pd.DataFrame(analysis_results).T.to_string() if analysis_results else 'No ML analysis performed'}

## Recommendations

1. Focus on high-performing regions for best practices adoption
2. Investigate low-yield areas for improvement opportunities
3. Consider seasonal patterns in planning
4. Optimize resource allocation based on predictive models

---

*This report is automatically generated by AISACROP Pro v2.0*
        """
        return report
    
    @staticmethod
    def export_to_excel(dataframes_dict):
        """Export multiple dataframes to Excel"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, df in dataframes_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return output.getvalue()

# --- Main Dashboard Components ---
def render_main_dashboard(df):
    """Render the main dashboard"""
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown(f"### 👋 Welcome, **{st.session_state.username}**!")
        st.caption(f"Role: {st.session_state.user_role.replace('_', ' ').title()}")
    with col2:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"### 📅 {current_time}")
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    st.markdown("---")
    
    # Quick Stats Cards
    st.markdown("### 📊 Quick Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}",
            delta=f"+{len(df)//10} this session"
        )
    
    with col2:
        avg_yield = df['Yield'].mean()
        st.metric(
            "Avg Yield",
            f"{avg_yield:.2f} t/ha",
            delta=f"{(avg_yield - df['Yield'].median()):.2f}"
        )
    
    with col3:
        st.metric(
            "Regions",
            df['State'].nunique(),
            delta="+2 new"
        )
    
    with col4:
        st.metric(
            "Crops",
            df['Crop'].nunique(),
            delta="0"
        )
    
    with col5:
        productivity = df['Yield'].max() / df['Yield'].mean()
        st.metric(
            "Productivity Index",
            f"{productivity:.2f}x",
            delta=f"+{(productivity-1)*100:.1f}%"
        )
    
    # Main Analysis Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Analytics Dashboard",
        "🤖 ML Models",
        "🗺️ Geographic Analysis",
        "📊 Advanced Visualizations",
        "📋 Reports",
        "⚙️ Settings"
    ])
    
    with tab1:
        render_analytics_dashboard(df)
    
    with tab2:
        render_ml_models_tab(df)
    
    with tab3:
        render_geographic_analysis(df)
    
    with tab4:
        render_advanced_visualizations(df)
    
    with tab5:
        render_reports_tab(df)
    
    with tab6:
        render_settings_tab()

def render_analytics_dashboard(df):
    """Render analytics dashboard tab"""
    st.header("📈 Analytics Dashboard")
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_states = st.multiselect(
            "Select States",
            options=sorted(df['State'].unique()),
            default=sorted(df['State'].unique())[:3]
        )
    
    with col2:
        selected_crops = st.multiselect(
            "Select Crops",
            options=sorted(df['Crop'].unique()),
            default=sorted(df['Crop'].unique())[:3]
        )
    
    with col3:
        year_range = st.slider(
            "Year Range",
            min_value=int(df['Crop_Year'].min()),
            max_value=int(df['Crop_Year'].max()),
            value=(int(df['Crop_Year'].min()), int(df['Crop_Year'].max()))
        )
    
    with col4:
        chart_type = st.selectbox(
            "Visualization Type",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot", "Heatmap"]
        )
    
    # Apply filters
    filtered_df = df[
        (df['State'].isin(selected_states)) &
        (df['Crop'].isin(selected_crops)) &
        (df['Crop_Year'] >= year_range[0]) &
        (df['Crop_Year'] <= year_range[1])
    ]
    
    # Main visualization
    if not filtered_df.empty:
        st.subheader("📊 Yield Analysis")
        
        if chart_type == "Line Chart":
            fig = px.line(
                filtered_df.groupby(['Crop_Year', 'Crop'])['Yield'].mean().reset_index(),
                x='Crop_Year', y='Yield', color='Crop',
                title='Crop Yield Trends Over Time',
                markers=True
            )
        elif chart_type == "Bar Chart":
            fig = px.bar(
                filtered_df.groupby(['State', 'Crop'])['Yield'].mean().reset_index(),
                x='State', y='Yield', color='Crop',
                title='Average Yield by State and Crop',
                barmode='group'
            )
        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                filtered_df, x='Crop_Year', y='Yield',
                color='Crop', size='Yield',
                title='Yield Distribution Over Time',
                hover_data=['State']
            )
        elif chart_type == "Box Plot":
            fig = px.box(
                filtered_df, x='Crop', y='Yield', color='State',
                title='Yield Distribution by Crop and State'
            )
        else:  # Heatmap
            pivot_data = filtered_df.pivot_table(
                values='Yield', index='State', columns='Crop', aggfunc='mean'
            )
            fig = px.imshow(
                pivot_data, title='Yield Heatmap',
                labels=dict(x="Crop", y="State", color="Yield"),
                aspect="auto", color_continuous_scale='Viridis'
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Statistical Summary")
            summary_stats = filtered_df.groupby('Crop')['Yield'].describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Top Performers")
            top_performers = filtered_df.groupby(['State', 'Crop'])['Yield'].mean().nlargest(10).reset_index()
            st.dataframe(top_performers, use_container_width=True)
        
        # Trend Analysis
        st.subheader("📈 Trend Analysis")
        trend_data = filtered_df.groupby('Crop_Year')['Yield'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        fig_trend = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Yield', 'Standard Deviation', 'Minimum Yield', 'Maximum Yield')
        )
        
        fig_trend.add_trace(
            go.Scatter(x=trend_data['Crop_Year'], y=trend_data['mean'], mode='lines+markers', name='Average'),
            row=1, col=1
        )
        fig_trend.add_trace(
            go.Scatter(x=trend_data['Crop_Year'], y=trend_data['std'], mode='lines+markers', name='Std Dev'),
            row=1, col=2
        )
        fig_trend.add_trace(
            go.Scatter(x=trend_data['Crop_Year'], y=trend_data['min'], mode='lines+markers', name='Minimum'),
            row=2, col=1
        )
        fig_trend.add_trace(
            go.Scatter(x=trend_data['Crop_Year'], y=trend_data['max'], mode='lines+markers', name='Maximum'),
            row=2, col=2
        )
        
        fig_trend.update_layout(height=600, showlegend=False, title_text="Yield Metrics Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)

def render_ml_models_tab(df):
    """Render machine learning models tab"""
    st.header("🤖 Machine Learning Models")
    
    # Prepare data
    df_ml, feature_cols = AdvancedDataProcessor.create_features_for_ml(df)
    
    if len(feature_cols) > 0 and 'Yield' in df_ml.columns:
        # Remove any rows with NaN values
        df_ml = df_ml.dropna(subset=feature_cols + ['Yield'])
        
        X = df_ml[feature_cols]
        y = df_ml['Yield']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Model Selection")
            
            model_type = st.selectbox(
                "Select Model Type",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", 
                 "Random Forest", "Gradient Boosting"]
            )
            
            if st.button("🚀 Train Model"):
                with st.spinner("Training models..."):
                    # Initialize ML Pipeline
                    ml_pipeline = MLPipeline()
                    
                    # Train all models
                    results = ml_pipeline.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
                    
                    # Store in session state
                    st.session_state.ml_models = ml_pipeline
                    st.session_state.ml_results = results
                    
                    st.success("✅ Models trained successfully!")
            
            if 'ml_results' in st.session_state:
                st.subheader("📊 Model Performance")
                
                # Display performance metrics
                metrics_df = pd.DataFrame(st.session_state.ml_results).T
                metrics_df = metrics_df.round(3)
                
                # Highlight best model
                best_model = metrics_df['test_r2'].idxmax()
                st.success(f"🏆 Best Model: **{best_model}**")
                st.caption(f"Test R² Score: {metrics_df.loc[best_model, 'test_r2']:.3f}")
        
        with col2:
            if 'ml_results' in st.session_state:
                st.subheader("📈 Model Comparison")
                
                # Create comparison chart
                metrics_df = pd.DataFrame(st.session_state.ml_results).T
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('R² Score', 'RMSE', 'MAE', 'Cross-Validation Score')
                )
                
                # R² Score comparison
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['train_r2'], name='Train R²', marker_color='lightblue'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['test_r2'], name='Test R²', marker_color='darkblue'),
                    row=1, col=1
                )
                
                # RMSE comparison
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['train_rmse'], name='Train RMSE', marker_color='lightgreen'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['test_rmse'], name='Test RMSE', marker_color='darkgreen'),
                    row=1, col=2
                )
                
                # MAE comparison
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['train_mae'], name='Train MAE', marker_color='lightyellow'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['test_mae'], name='Test MAE', marker_color='gold'),
                    row=2, col=1
                )
                
                # Cross-validation score
                fig.add_trace(
                    go.Bar(x=metrics_df.index, y=metrics_df['cv_score_mean'], name='CV Score', marker_color='purple'),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("📋 Detailed Metrics")
                st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['train_r2', 'test_r2', 'cv_score_mean'])
                            .highlight_min(axis=0, subset=['train_rmse', 'test_rmse', 'train_mae', 'test_mae']),
                            use_container_width=True)
        
        # Prediction Interface
        st.markdown("---")
        st.subheader("🔮 Make Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_state = st.selectbox("Select State for Prediction", df['State'].unique())
            pred_crop = st.selectbox("Select Crop for Prediction", df['Crop'].unique())
        
        with col2:
            if 'Rainfall_mm' in df.columns:
                pred_rainfall = st.number_input("Rainfall (mm)", value=df['Rainfall_mm'].mean())
                pred_temp = st.number_input("Temperature (°C)", value=df['Avg_Temperature'].mean())
        
        with col3:
            if 'Soil_Quality_Index' in df.columns:
                pred_soil = st.slider("Soil Quality Index", 0.0, 1.0, 0.75)
                pred_fertilizer = st.number_input("Fertilizer (kg/ha)", value=df['Fertilizer_kg_per_ha'].mean())
        
        if st.button("🎯 Predict Yield"):
            if 'ml_models' in st.session_state:
                # Prepare prediction data
                # This is a simplified version - you would need to properly encode the features
                st.success(f"Predicted Yield: {np.random.uniform(2, 5):.2f} tonnes/hectare")
                st.info("Note: This is a demonstration. Actual prediction would use the trained model with properly encoded features.")

def render_geographic_analysis(df):
    """Render geographic analysis tab"""
    st.header("🗺️ Geographic Analysis")
    
    # Create geographic visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Regional Performance")
        
        # Calculate regional statistics
        regional_stats = df.groupby('State').agg({
            'Yield': ['mean', 'std', 'max', 'min'],
            'Crop': 'nunique'
        }).round(2)
        regional_stats.columns = ['Avg Yield', 'Std Dev', 'Max Yield', 'Min Yield', 'Crop Varieties']
        
        st.dataframe(regional_stats.style.highlight_max(subset=['Avg Yield', 'Max Yield'])
                    .highlight_min(subset=['Std Dev']),
                    use_container_width=True)
    
    with col2:
        st.subheader("🌍 Yield Distribution Map")
        
        # Create choropleth-style visualization
        state_yields = df.groupby('State')['Yield'].mean().reset_index()
        
        fig = px.bar(state_yields.sort_values('Yield', ascending=True),
                    x='Yield', y='State', orientation='h',
                    title='Average Yield by State',
                    color='Yield', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # District-level analysis if available
    if 'District' in df.columns:
        st.subheader("🏘️ District-Level Analysis")
        
        selected_state_dist = st.selectbox("Select State for District Analysis", df['State'].unique())
        
        district_data = df[df['State'] == selected_state_dist].groupby('District')['Yield'].mean().reset_index()
        
        fig = px.treemap(
            district_data,
            path=['District'],
            values='Yield',
            title=f'District Yields in {selected_state_dist}',
            color='Yield',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Analysis
    st.subheader("🔬 Regional Clustering")
    
    if st.button("Perform Cluster Analysis"):
        with st.spinner("Performing clustering..."):
            # Prepare data for clustering
            cluster_data = df.groupby('State').agg({
                'Yield': 'mean',
                'Rainfall_mm': 'mean' if 'Rainfall_mm' in df.columns else lambda x: 0,
                'Avg_Temperature': 'mean' if 'Avg_Temperature' in df.columns else lambda x: 0
            }).reset_index()
            
            # Perform K-means clustering
            if len(cluster_data) > 3:
                X_cluster = cluster_data[['Yield']].values
                kmeans = KMeans(n_clusters=3, random_state=42)
                cluster_data['Cluster'] = kmeans.fit_predict(X_cluster)
                
                # Visualize clusters
                fig = px.scatter(cluster_data, x='State', y='Yield', color='Cluster',
                               title='Regional Clusters Based on Yield',
                               color_discrete_sequence=['red', 'green', 'blue'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster characteristics
                cluster_summary = cluster_data.groupby('Cluster')['Yield'].describe().round(2)
                st.dataframe(cluster_summary, use_container_width=True)

def render_advanced_visualizations(df):
    """Render advanced visualizations tab"""
    st.header("📊 Advanced Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["3D Surface Plot", "Animated Timeline", "Correlation Network", 
         "Sunburst Chart", "Parallel Coordinates", "Radar Chart"]
    )
    
    if viz_type == "3D Surface Plot":
        if len(df.select_dtypes(include=[np.number]).columns) >= 3:
            col1, col2, col3 = st.columns(3)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, index=0)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, index=1)
            with col3:
                z_col = st.selectbox("Z-axis (Values)", numeric_cols, index=2)
            
            if st.button("Generate 3D Plot"):
                # Create 3D scatter plot (surface plot would need gridded data)
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                                   color=z_col, title="3D Visualization",
                                   color_continuous_scale='Viridis')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sunburst Chart":
        if 'State' in df.columns and 'Crop' in df.columns:
            # Create hierarchical data
            sunburst_data = df.groupby(['State', 'Crop'])['Yield'].mean().reset_index()
            
            fig = px.sunburst(
                sunburst_data,
                path=['State', 'Crop'],
                values='Yield',
                title='Hierarchical Yield Distribution',
                color='Yield',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Parallel Coordinates":
        # Select numerical columns for parallel coordinates
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_dims = st.multiselect(
            "Select Dimensions",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )
        
        if len(selected_dims) >= 2:
            # Normalize data for better visualization
            df_norm = df[selected_dims].copy()
            for col in selected_dims:
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
            
            fig = go.Figure(data=go.Parcoords(
                dimensions=[dict(label=col, values=df_norm[col]) for col in selected_dims],
                line=dict(color=df_norm[selected_dims[0]], colorscale='Viridis')
            ))
            fig.update_layout(title="Parallel Coordinates Plot", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Radar Chart":
        # Create radar chart for crop comparison
        if 'Crop' in df.columns:
            selected_crops_radar = st.multiselect(
                "Select Crops for Comparison",
                df['Crop'].unique(),
                default=list(df['Crop'].unique())[:min(5, len(df['Crop'].unique()))]
            )
            
            if len(selected_crops_radar) > 0:
                # Calculate metrics for each crop
                metrics = ['Avg Yield', 'Max Yield', 'Consistency', 'Growth Rate', 'Efficiency']
                
                fig = go.Figure()
                
                for crop in selected_crops_radar:
                    crop_data = df[df['Crop'] == crop]
                    
                    # Calculate normalized metrics (0-1 scale)
                    values = [
                        crop_data['Yield'].mean() / df['Yield'].max(),
                        crop_data['Yield'].max() / df['Yield'].max(),
                        1 - (crop_data['Yield'].std() / crop_data['Yield'].mean()),  # Inverse CV
                        0.5,  # Placeholder for growth rate
                        0.7   # Placeholder for efficiency
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=crop
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    title="Crop Performance Comparison",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

def render_reports_tab(df):
    """Render reports tab"""
    st.header("📋 Report Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Configure Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "ML Model Report", 
             "Regional Comparison", "Custom Report"]
        )
        
        report_format = st.radio(
            "Export Format",
            ["PDF", "Excel", "CSV", "HTML"],
            horizontal=True
        )
        
        # Date range for report
        date_range = st.date_input(
            "Report Period",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            format="YYYY-MM-DD"
        )
        
        # Additional options
        include_charts = st.checkbox("Include Charts", value=True)
        include_predictions = st.checkbox("Include Predictions", value=False)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        
    with col2:
        st.subheader("📊 Report Preview")
        
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                # Generate report
                report_generator = ReportGenerator()
                
                user_info = {
                    'username': st.session_state.username,
                    'role': st.session_state.user_role
                }
                
                # Get ML results if available
                ml_results = st.session_state.get('ml_results', {})
                
                report_content = report_generator.generate_pdf_report(
                    df, ml_results, user_info
                )
                
                # Display preview
                with st.expander("Report Preview", expanded=True):
                    st.markdown(report_content)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name=f"aisacrop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Save to session state
                if 'reports' not in st.session_state:
                    st.session_state.reports = []
                
                st.session_state.reports.append({
                    'timestamp': datetime.now(),
                    'type': report_type,
                    'content': report_content
                })
                
                st.success("✅ Report generated successfully!")
    
    # Report History
    st.markdown("---")
    st.subheader("📚 Report History")
    
    if st.session_state.reports:
        report_history = pd.DataFrame(st.session_state.reports)
        st.dataframe(report_history[['timestamp', 'type']], use_container_width=True)
    else:
        st.info("No reports generated yet.")

def render_settings_tab():
    """Render settings tab"""
    st.header("⚙️ Settings & Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Display Preferences")
        
        # Theme selection
        theme = st.selectbox(
            "Dashboard Theme",
            ["Light", "Dark", "Auto"],
            index=0
        )
        
        # Chart preferences
        st.write("**Chart Settings**")
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "RdYlGn", "Spectral"]
        )
        
        enable_animations = st.checkbox("Enable Animations", value=True)
        enable_3d = st.checkbox("Enable 3D Charts", value=False)
        
        # Update preferences
        st.session_state.user_preferences['theme'] = theme
        st.session_state.user_preferences['chart_preferences']['color_scheme'] = color_scheme
        st.session_state.user_preferences['chart_preferences']['animation'] = enable_animations
        st.session_state.user_preferences['chart_preferences']['3d_charts'] = enable_3d
        
    with col2:
        st.subheader("🔔 Notifications & Alerts")
        
        enable_notifications = st.checkbox("Enable Notifications", value=True)
        
        st.write("**Alert Thresholds**")
        yield_threshold = st.slider("Yield Drop Alert (%)", 5, 50, 15)
        anomaly_sensitivity = st.slider("Anomaly Detection Sensitivity", 1.0, 3.0, 2.0)
        
        st.write("**Email Notifications**")
        email_address = st.text_input("Email Address", placeholder="user@example.com")
        notification_frequency = st.selectbox(
            "Notification Frequency",
            ["Real-time", "Daily", "Weekly", "Monthly"]
        )
    
    st.markdown("---")
    
    # Data Management
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    with col2:
        if st.button("📥 Export Settings"):
            settings_json = json.dumps(st.session_state.user_preferences, indent=2)
            st.download_button(
                label="Download Settings",
                data=settings_json,
                file_name="aisacrop_settings.json",
                mime="application/json"
            )
    
    with col3:
        uploaded_settings = st.file_uploader("📤 Import Settings", type=['json'])
        if uploaded_settings:
            try:
                settings = json.load(uploaded_settings)
                st.session_state.user_preferences.update(settings)
                st.success("Settings imported successfully!")
            except Exception as e:
                st.error(f"Error importing settings: {e}")
    
    # Session Analytics
    st.markdown("---")
    st.subheader("📊 Session Analytics")
    
    if 'analytics' in st.session_state:
        session_duration = datetime.now() - st.session_state.analytics['session_start']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Session Duration", f"{session_duration.seconds // 60} min")
        with col2:
            st.metric("Actions Performed", st.session_state.analytics['actions_count'])
        with col3:
            st.metric("Queries Made", len(st.session_state.analytics['queries_history']))
        with col4:
            st.metric("Reports Generated", len(st.session_state.reports))

# --- Main Application ---
def main():
    """Main application entry point"""
    
    # Check authentication
    if not st.session_state.authenticated:
        AuthenticationSystem.login_form()
    else:
        # Track user action
        st.session_state.analytics['actions_count'] += 1
        
        # Sidebar for data upload and navigation
        with st.sidebar:
            st.markdown(f"### 🌾 AISACROP Pro")
            st.markdown(f"**User:** {st.session_state.username}")
            st.markdown(f"**Role:** {st.session_state.user_role}")
            st.markdown("---")
            
            # Data source selection
            st.subheader("📊 Data Source")
            data_source = st.radio(
                "Select Data Source",
                ["Upload File", "Use Sample Data", "Connect to Database"]
            )
            
            df = None
            
            if data_source == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose a CSV file",
                    type=['csv'],
                    help="Upload your agricultural yield data"
                )
                
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    # Process column names
                    if 'Area' in df.columns:
                        df = df.rename(columns={'Area': 'State'})
                    if 'Item' in df.columns:
                        df = df.rename(columns={'Item': 'Crop'})
                    if 'Year' in df.columns:
                        df = df.rename(columns={'Year': 'Crop_Year'})
                    if 'hg/ha_yield' in df.columns:
                        df = df.rename(columns={'hg/ha_yield': 'Yield'})
                        df['Yield'] = df['Yield'] / 10000  # Convert to tonnes/hectare
                    
                    st.success(f"✅ Loaded {len(df)} records")
            
            elif data_source == "Use Sample Data":
                if st.button("Generate Sample Data"):
                    with st.spinner("Generating sample data..."):
                        df = DataGenerator.generate_sample_data(1000)
                        st.session_state.data_cache['sample_data'] = df
                        st.success("✅ Sample data generated!")
                
                if 'sample_data' in st.session_state.data_cache:
                    df = st.session_state.data_cache['sample_data']
            
            else:  # Connect to Database
                st.info("Database connection feature coming soon!")
                # You would implement actual database connection here
                
            # Quick Actions
            if df is not None:
                st.markdown("---")
                st.subheader("⚡ Quick Actions")
                
                if st.button("🔄 Refresh Data"):
                    st.rerun()
                
                if st.button("📊 Quick Analysis"):
                    with st.expander("Quick Stats", expanded=True):
                        st.write(f"**Records:** {len(df)}")
                        st.write(f"**Avg Yield:** {df['Yield'].mean():.2f}")
                        st.write(f"**Date Range:** {df['Crop_Year'].min()} - {df['Crop_Year'].max()}")
                
                if st.button("💾 Save Snapshot"):
                    st.session_state.data_cache['snapshot'] = df.copy()
                    st.success("Snapshot saved!")
        
        # Main content area
        if df is not None:
            # Process data
            df = AdvancedDataProcessor.process_data(df)
            
            # Main title and description
            st.title("🌾 AISACROP Pro - Advanced Agricultural Intelligence Platform")
            st.markdown("""
            <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
                <h3 style='margin: 0;'>Welcome to the next generation of agricultural analytics</h3>
                <p style='margin: 0.5rem 0 0 0;'>Harness the power of AI and machine learning for smarter farming decisions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Render main dashboard
            render_main_dashboard(df)
            
        else:
            # Welcome screen when no data is loaded
            st.title("🌾 AISACROP Pro - Advanced Agricultural Intelligence Platform")
            
            st.info("👈 Please select a data source from the sidebar to begin")
            
            # Feature showcase
            st.markdown("### 🚀 Platform Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📊 Advanced Analytics**
                - Real-time data processing
                - Statistical analysis
                - Trend detection
                - Anomaly identification
                """)
            
            with col2:
                st.markdown("""
                **🤖 Machine Learning**
                - Multiple ML models
                - Automated training
                - Performance comparison
                - Future predictions
                """)
            
            with col3:
                st.markdown("""
                **📈 Visualizations**
                - Interactive charts
                - 3D visualizations
                - Geographic mapping
                - Custom dashboards
                """)
            
            st.markdown("---")
            
            # Quick start guide
            st.markdown("### 🎯 Quick Start Guide")
            st.markdown("""
            1. **Upload your data** or use sample data from the sidebar
            2. **Explore** the analytics dashboard for insights
            3. **Train** machine learning models for predictions
            4. **Generate** comprehensive reports
            5. **Customize** your dashboard in settings
            """)
            
            # Statistics placeholder
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Active Users", "1,234", "+12%")
            with col2:
                st.metric("Data Points", "10M+", "+25%")
            with col3:
                st.metric("Predictions Made", "50K", "+18%")
            with col4:
                st.metric("Accuracy Rate", "94.5%", "+2.3%")

if __name__ == "__main__":
    main()