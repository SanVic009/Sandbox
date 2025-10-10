import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

# --- App Configuration ---
st.set_page_config(
    page_title="AISACROP - Smart Crop Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'layout_style': 'Wide',
        'chart_theme': 'streamlit',
        'favorite_views': [],
        'alert_thresholds': {
            'yield_drop_percentage': 15,
            'anomaly_sensitivity': 2.0,
            'forecast_threshold': 10
        }
    }

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# --- Enhanced Functions ---

@st.cache_data
def load_data(file):
    """
    Loads and preprocesses the crop yield data using the correct column names.
    """
    df = pd.read_csv(file)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'Area': 'State',
        'Item': 'Crop', 
        'Year': 'Crop_Year',
        'hg/ha_yield': 'Yield'
    })
    
    # Basic preprocessing
    df['Crop_Year'] = pd.to_numeric(df['Crop_Year'], errors='coerce')
    df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
    df.dropna(subset=['Crop_Year', 'Yield'], inplace=True)
    df['Crop_Year'] = df['Crop_Year'].astype(int)
    
    # Convert yield from hectograms per hectare to tonnes per hectare
    df['Yield'] = df['Yield'] / 10000
    
    return df

def process_natural_language_query(query, df):
    """
    Process natural language queries about crop data
    """
    query = query.lower()
    
    # Extract entities
    crops = df['Crop'].unique()
    states = df['State'].unique()
    years = df['Crop_Year'].unique()
    
    found_crops = [crop for crop in crops if crop.lower() in query]
    found_states = [state for state in states if state.lower() in query]
    found_years = [year for year in years if str(year) in query]
    
    # Query patterns
    patterns = {
        'highest_yield': r'(highest|maximum|max|best|top).*yield',
        'lowest_yield': r'(lowest|minimum|min|worst|bottom).*yield',
        'average_yield': r'(average|mean|avg).*yield',
        'trend': r'(trend|trending|growth|decline|pattern)',
        'comparison': r'(compare|comparison|versus|vs|difference)',
        'prediction': r'(predict|forecast|future|next|upcoming)',
        'year_specific': r'(in|during|for)\s+(\d{4})'
    }
    
    response = {}
    
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, query):
            response['query_type'] = pattern_name
            break
    
    response['crops'] = found_crops
    response['states'] = found_states
    response['years'] = found_years
    
    return response

def generate_smart_alerts(df, selected_state, selected_crop):
    """
    Generate smart alerts for anomalies and thresholds
    """
    alerts = []
    thresholds = st.session_state.user_preferences['alert_thresholds']
    
    # Filter data
    filtered_df = df[(df['State'] == selected_state) & (df['Crop'] == selected_crop)]
    
    if len(filtered_df) < 3:
        return alerts
    
    yields = filtered_df['Yield'].values
    years = filtered_df['Crop_Year'].values
    
    # 1. Yield drop detection
    if len(yields) >= 2:
        recent_change = ((yields[-1] - yields[-2]) / yields[-2]) * 100
        if recent_change < -thresholds['yield_drop_percentage']:
            alerts.append({
                'type': 'warning',
                'title': f'Significant Yield Drop Detected',
                'message': f'{selected_crop} yield in {selected_state} dropped by {abs(recent_change):.1f}% from {years[-2]} to {years[-1]}',
                'action': 'Consider investigating environmental factors, soil health, or pest issues.'
            })
    
    # 2. Anomaly detection using z-score
    if len(yields) >= 3:
        z_scores = np.abs(stats.zscore(yields))
        anomalies = np.where(z_scores > thresholds['anomaly_sensitivity'])[0]
        
        if len(anomalies) > 0:
            latest_anomaly = anomalies[-1]
            alerts.append({
                'type': 'info',
                'title': 'Yield Anomaly Detected',
                'message': f'Unusual yield pattern detected in {years[latest_anomaly]} ({yields[latest_anomaly]:.2f} tonnes/ha)',
                'action': 'Review data quality and investigate potential causes for this outlier.'
            })
    
    # 3. Forecast threshold alerts
    if len(yields) >= 3:
        X = years.reshape(-1, 1)
        model = LinearRegression().fit(X, yields)
        next_year_pred = model.predict([[years[-1] + 1]])[0]
        
        avg_yield = np.mean(yields)
        pred_change = ((next_year_pred - avg_yield) / avg_yield) * 100
        
        if abs(pred_change) > thresholds['forecast_threshold']:
            alert_type = 'success' if pred_change > 0 else 'warning'
            direction = 'increase' if pred_change > 0 else 'decrease'
            
            alerts.append({
                'type': alert_type,
                'title': f'Significant Forecast {direction.title()}',
                'message': f'Predicted {direction} of {abs(pred_change):.1f}% in {selected_crop} yield for {years[-1] + 1}',
                'action': f'Prepare for {"enhanced productivity opportunities" if pred_change > 0 else "potential yield challenges"}.'
            })
    
    return alerts

def create_advanced_visualization(df, selected_state, selected_crop, viz_type):
    """
    Create advanced visualizations based on user preference
    """
    filtered_df = df[(df['State'] == selected_state) & (df['Crop'] == selected_crop)]
    
    if viz_type == "Interactive Line Chart":
        fig = px.line(filtered_df, x='Crop_Year', y='Yield', 
                     title=f'{selected_crop} Yield Trend in {selected_state}',
                     markers=True, hover_data=['Yield'])
        fig.update_layout(xaxis_title="Year", yaxis_title="Yield (Tonnes/Hectare)")
        return fig
    
    elif viz_type == "Box Plot Analysis":
        fig = px.box(df[df['Crop'] == selected_crop], x='State', y='Yield',
                    title=f'{selected_crop} Yield Distribution by Region')
        fig.update_layout(xaxis_title="Region", yaxis_title="Yield (Tonnes/Hectare)")
        return fig
    
    elif viz_type == "Heatmap":
        pivot_df = df[df['State'] == selected_state].pivot_table(
            index='Crop', columns='Crop_Year', values='Yield', aggfunc='mean'
        )
        fig = px.imshow(pivot_df, aspect="auto", color_continuous_scale='Viridis',
                       title=f'Yield Heatmap for {selected_state}')
        return fig
    
    elif viz_type == "Scatter with Trend":
        fig = px.scatter(filtered_df, x='Crop_Year', y='Yield',
                        title=f'{selected_crop} Yield Scatter Plot with Trend',
                        trendline="ols")
        return fig

# --- Main Application ---
st.title('🌾 AISACROP - Smart Crop Analytics Platform')
st.markdown("""
**Advanced crop yield prediction with AI-powered insights, natural language queries, and smart alerts.**
""")

# --- Sidebar Configuration ---
st.sidebar.header('🎛️ Dashboard Controls')

# Layout Personalization
st.sidebar.subheader("🎨 Dashboard Personalization")
layout_style = st.sidebar.selectbox(
    "Layout Style", 
    ["Wide", "Compact", "Detailed"],
    index=["Wide", "Compact", "Detailed"].index(st.session_state.user_preferences['layout_style'])
)

chart_theme = st.sidebar.selectbox(
    "Chart Theme",
    ["streamlit", "plotly", "plotly_white", "plotly_dark"],
    index=["streamlit", "plotly", "plotly_white", "plotly_dark"].index(st.session_state.user_preferences['chart_theme'])
)

# Update preferences
st.session_state.user_preferences['layout_style'] = layout_style
st.session_state.user_preferences['chart_theme'] = chart_theme

# File Upload
uploaded_file = st.sidebar.file_uploader("📁 Upload Crop Yield Data", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Data filtering options
    st.sidebar.subheader("🔍 Data Filters")
    
    # Multi-select for states and crops
    all_states = sorted(df['State'].unique())
    all_crops = sorted(df['Crop'].unique())
    
    selected_states = st.sidebar.multiselect('Select Regions', all_states, default=all_states[:3])
    selected_crops = st.sidebar.multiselect('Select Crops', all_crops, default=all_crops[:3])
    
    # Year range filter
    min_year, max_year = int(df['Crop_Year'].min()), int(df['Crop_Year'].max())
    year_range = st.sidebar.slider('Year Range', min_year, max_year, (min_year, max_year))
    
    # Filter dataframe
    filtered_df = df[
        (df['State'].isin(selected_states)) & 
        (df['Crop'].isin(selected_crops)) &
        (df['Crop_Year'] >= year_range[0]) &
        (df['Crop_Year'] <= year_range[1])
    ]
    
    # Alert Threshold Configuration
    st.sidebar.subheader("⚠️ Smart Alert Settings")
    yield_drop_threshold = st.sidebar.slider("Yield Drop Alert (%)", 5, 50, 
                                            st.session_state.user_preferences['alert_thresholds']['yield_drop_percentage'])
    anomaly_sensitivity = st.sidebar.slider("Anomaly Sensitivity", 1.0, 3.0, 
                                           st.session_state.user_preferences['alert_thresholds']['anomaly_sensitivity'])
    
    st.session_state.user_preferences['alert_thresholds'].update({
        'yield_drop_percentage': yield_drop_threshold,
        'anomaly_sensitivity': anomaly_sensitivity
    })
    
    # Natural Language Query Section
    st.header("🗣️ Ask Questions About Your Data")
    query_input = st.text_input(
        "Ask a question in plain language:",
        placeholder="e.g., 'What is the highest wheat yield in India?' or 'Show me the trend for rice production'"
    )
    
    if query_input:
        query_result = process_natural_language_query(query_input, filtered_df)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("🤖 **Query Analysis:**")
            if query_result.get('query_type'):
                st.write(f"**Type:** {query_result['query_type'].replace('_', ' ').title()}")
            if query_result.get('crops'):
                st.write(f"**Crops Found:** {', '.join(query_result['crops'])}")
            if query_result.get('states'):
                st.write(f"**Regions Found:** {', '.join(query_result['states'])}")
        
        # Generate answer based on query
        if query_result.get('query_type') == 'highest_yield':
            max_yield_row = filtered_df.loc[filtered_df['Yield'].idxmax()]
            st.success(f"**Answer:** The highest yield is {max_yield_row['Yield']:.2f} tonnes/ha for {max_yield_row['Crop']} in {max_yield_row['State']} ({max_yield_row['Crop_Year']})")
        
        elif query_result.get('query_type') == 'average_yield':
            if query_result.get('crops') and query_result.get('states'):
                crop = query_result['crops'][0]
                state = query_result['states'][0]
                avg_yield = filtered_df[(filtered_df['Crop'] == crop) & (filtered_df['State'] == state)]['Yield'].mean()
                st.success(f"**Answer:** The average yield for {crop} in {state} is {avg_yield:.2f} tonnes/ha")
    
    # Main Dashboard Layout
    if layout_style == "Wide":
        col1, col2, col3 = st.columns([2, 2, 1])
    elif layout_style == "Compact":
        col1, col2, col3 = st.columns([3, 2, 1])
    else:  # Detailed
        col1, col2, col3 = st.columns([1, 2, 1])
    
    # Data Overview
    with col1:
        st.subheader("📊 Data Overview")
        st.metric("Total Records", len(filtered_df))
        st.metric("Regions", len(selected_states))
        st.metric("Crops", len(selected_crops))
        st.metric("Year Span", f"{year_range[1] - year_range[0]} years")
        
        # Quick Statistics
        if len(filtered_df) > 0:
            st.write("**Quick Stats:**")
            st.write(f"• Avg Yield: {filtered_df['Yield'].mean():.2f} t/ha")
            st.write(f"• Max Yield: {filtered_df['Yield'].max():.2f} t/ha")
            st.write(f"• Min Yield: {filtered_df['Yield'].min():.2f} t/ha")
    
    # Visualization Selection
    with col2:
        st.subheader("📈 Visualization Options")
        viz_type = st.selectbox(
            "Select Visualization:",
            ["Interactive Line Chart", "Box Plot Analysis", "Heatmap", "Scatter with Trend"]
        )
        
        if len(selected_states) > 0 and len(selected_crops) > 0:
            primary_state = st.selectbox("Primary Region for Analysis:", selected_states)
            primary_crop = st.selectbox("Primary Crop for Analysis:", selected_crops)
            
            # Generate visualization
            fig = create_advanced_visualization(filtered_df, primary_state, primary_crop, viz_type)
            st.plotly_chart(fig, use_container_width=True)
    
    # Smart Alerts Panel
    with col3:
        st.subheader("🚨 Smart Alerts")
        
        if len(selected_states) > 0 and len(selected_crops) > 0:
            alerts = generate_smart_alerts(filtered_df, primary_state, primary_crop)
            
            if alerts:
                for alert in alerts:
                    if alert['type'] == 'warning':
                        st.warning(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 {alert['action']}")
                    elif alert['type'] == 'success':
                        st.success(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 {alert['action']}")
                    else:
                        st.info(f"**{alert['title']}**\n\n{alert['message']}\n\n💡 {alert['action']}")
            else:
                st.success("✅ No alerts detected. All metrics within normal ranges.")
        
        # Save Favorite View
        st.subheader("⭐ Favorite Views")
        view_name = st.text_input("Save Current View As:", placeholder="My Dashboard")
        if st.button("💾 Save View"):
            if view_name:
                favorite_view = {
                    'name': view_name,
                    'states': selected_states,
                    'crops': selected_crops,
                    'year_range': year_range,
                    'viz_type': viz_type,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
                st.session_state.user_preferences['favorite_views'].append(favorite_view)
                st.success(f"✅ View '{view_name}' saved!")
        
        # Load Favorite Views
        if st.session_state.user_preferences['favorite_views']:
            st.write("**Saved Views:**")
            for i, view in enumerate(st.session_state.user_preferences['favorite_views']):
                if st.button(f"📌 {view['name']}", key=f"view_{i}"):
                    st.rerun()
    
    # Detailed Analysis Section
    st.markdown("---")
    st.header("📋 Detailed Analysis")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 Statistical Summary", "🔮 Predictions", "📈 Comparative Analysis"])
    
    with analysis_tab1:
        if len(filtered_df) > 0:
            st.subheader("Statistical Summary")
            summary_stats = filtered_df.groupby(['State', 'Crop'])['Yield'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            st.dataframe(summary_stats, use_container_width=True)
    
    with analysis_tab2:
        if len(selected_states) > 0 and len(selected_crops) > 0:
            st.subheader(f"Yield Predictions for {primary_crop} in {primary_state}")
            
            primary_data = filtered_df[
                (filtered_df['State'] == primary_state) & 
                (filtered_df['Crop'] == primary_crop)
            ].sort_values('Crop_Year')
            
            if len(primary_data) >= 3:
                X = primary_data[['Crop_Year']]
                y = primary_data['Yield']
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict next 3 years
                future_years = np.array([year_range[1] + 1, year_range[1] + 2, year_range[1] + 3]).reshape(-1, 1)
                predictions = model.predict(future_years)
                
                pred_df = pd.DataFrame({
                    'Year': future_years.flatten(),
                    'Predicted_Yield': np.round(predictions, 2)
                })
                
                st.table(pred_df)
                
                # R-squared score
                r2_score = model.score(X, y)
                st.metric("Model Accuracy (R²)", f"{r2_score:.3f}")
            else:
                st.warning("Need at least 3 data points for prediction.")
    
    with analysis_tab3:
        if len(selected_crops) >= 2:
            st.subheader("Crop Yield Comparison")
            
            comparison_data = []
            for crop in selected_crops[:5]:  # Limit to 5 crops for readability
                crop_data = filtered_df[filtered_df['Crop'] == crop].groupby('Crop_Year')['Yield'].mean().reset_index()
                crop_data['Crop'] = crop
                comparison_data.append(crop_data)
            
            if comparison_data:
                comparison_df = pd.concat(comparison_data, ignore_index=True)
                
                fig = px.line(comparison_df, x='Crop_Year', y='Yield', color='Crop',
                             title='Crop Yield Comparison Over Time',
                             markers=True)
                fig.update_layout(xaxis_title="Year", yaxis_title="Average Yield (Tonnes/Hectare)")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin your smart crop analytics journey.")
    st.markdown("""
    ### 🚀 New Features Available:
    
    - **🗣️ Natural Language Queries**: Ask questions about your data in plain English
    - **🎨 Dashboard Personalization**: Customize layouts, themes, and visualizations  
    - **🚨 Smart Alerts**: Get proactive notifications about yield anomalies and trends
    - **⭐ Favorite Views**: Save and quickly access your preferred dashboard configurations
    - **📊 Advanced Analytics**: Interactive charts, statistical summaries, and comparative analysis
    """)
    
    st.image("https://placehold.co/800x400/2c6e49/ffffff?text=Upload+Your+Data+for+Smart+Analytics", 
             caption="Upload your dataset to unlock advanced crop analytics", use_container_width=True)