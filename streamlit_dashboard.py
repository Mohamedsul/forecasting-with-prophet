"""
streamlit_dashboard.py
----------------------
Interactive sales forecasting dashboard using Streamlit.

This application provides an intuitive interface for:
- Uploading sales data
- Visualizing historical trends
- Generating forecasts with Prophet
- Evaluating model performance
- Downloading predictions

Author: Mohamed Suliman
Date: 2025  

Usage:
------
streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load and parse uploaded CSV file."""
    try:
        data = pd.read_csv(uploaded_file)
        
        # Convert Order Date to datetime with multiple format support
        data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce', dayfirst=False)
        
        # Remove rows with invalid dates
        invalid_dates = data['Order Date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ Found {invalid_dates} rows with invalid dates. These will be removed.")
        
        data = data.dropna(subset=['Order Date'])
        
        # Check if Sales column exists and convert to numeric
        if 'Sales' in data.columns:
            data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
            data = data.dropna(subset=['Sales'])
        
        # Sort by date
        data = data.sort_values('Order Date')
        data = data.reset_index(drop=True)
        
        return data, None
    except Exception as e:
        return None, str(e)


def safe_smape(true, pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    true, pred = np.array(true), np.array(pred)
    denominator = np.abs(true) + np.abs(pred)
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    return 100 / mask.sum() * np.sum(
        2 * np.abs(pred[mask] - true[mask]) / denominator[mask]
    )


def evaluate(true, pred):
    """Calculate comprehensive evaluation metrics."""
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    smape = safe_smape(true, pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }


@st.cache_resource
def train_prophet(df, periods=30, seasonality_mode='multiplicative'):
    """Train Prophet model with optimized parameters."""
    # Prepare data
    df_prophet = df[['Order Date', 'Sales']].copy()
    df_prophet.columns = ['ds', 'y']
    
    # Initialize model with tuned parameters
    model = Prophet(
        seasonality_mode=seasonality_mode,
        weekly_seasonality=10,
        yearly_seasonality=20,
        changepoint_prior_scale=0.1,
        interval_width=0.95
    )
    
    # Add custom monthly seasonality
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=10
    )
    
    # Fit model
    with st.spinner('Training Prophet model...'):
        model.fit(df_prophet)
    
    # Generate forecast
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    
    return model, forecast


def create_plotly_forecast(data, forecast, periods):
    """Create interactive Plotly forecast visualization."""
    # Historical data
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Order Date'],
        y=data['Sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    forecast_dates = forecast['ds'].tail(periods)
    forecast_values = forecast['yhat'].tail(periods)
    upper_bound = forecast['yhat_upper'].tail(periods)
    lower_bound = forecast['yhat_lower'].tail(periods)
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=lower_bound,
        mode='lines',
        name='95% Confidence',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title='Sales Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Sales Forecasting Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #e8f4f8; 
                border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin: 0; font-size: 1.1rem;'>
            Upload your sales data and generate accurate forecasts using Facebook Prophet
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/analytics.png", width=100)
        st.title("⚙️ Configuration")
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Sales Data (CSV)",
            type="csv",
            help="CSV file must contain 'Order Date' and 'Sales' columns"
        )
        
        st.markdown("---")
        
        # Model parameters
        st.subheader("Model Settings")
        
        forecast_periods = st.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=365,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
        
        seasonality_mode = st.selectbox(
            "Seasonality Mode",
            options=['multiplicative', 'additive'],
            index=0,
            help="Multiplicative works better for data with seasonal patterns that grow over time"
        )
        
        show_components = st.checkbox(
            "Show Forecast Components",
            value=False,
            help="Display trend, weekly, and yearly seasonality components"
        )
        
        st.markdown("---")
        
        # Information
        st.subheader("ℹ️ About")
        st.info("""
        **Prophet Model Features:**
        - Handles missing data
        - Captures multiple seasonalities
        - Robust to outliers
        - Provides uncertainty intervals
        """)
        
        st.markdown("---")
        st.markdown("**Developed with ❤️ using Streamlit**")
    
    # Main content
    if uploaded_file is None:
        # Landing page with instructions
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info("👈 Please upload a CSV file to get started")
            
            st.markdown("### 📋 Data Requirements")
            st.markdown("""
            Your CSV file should contain:
            - **Order Date**: Date column (format: YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY)
            - **Sales**: Numerical sales values
            
            Example:
            ```
            Order Date,Sales
            2023-01-01,1250.50
            2023-01-02,1340.25
            ```
            """)
            
            # Sample data download
            sample_data = pd.DataFrame({
                'Order Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                'Sales': np.random.randint(1000, 5000, 100)
            })
            
            st.download_button(
                label="📥 Download Sample Dataset",
                data=sample_data.to_csv(index=False),
                file_name="sample_sales_data.csv",
                mime="text/csv"
            )
    
    else:
        # Load data
        data, error = load_data(uploaded_file)
        
        if error:
            st.error(f"❌ Error loading file: {error}")
            st.stop()
        
        # Validate required columns
        if 'Order Date' not in data.columns or 'Sales' not in data.columns:
            st.error("❌ CSV must contain 'Order Date' and 'Sales' columns")
            st.stop()
        
        # Check if we have valid data
        if len(data) == 0:
            st.error("❌ No valid data found after processing. Please check your CSV file.")
            st.stop()
        
        # Display success message
        st.success(f"✅ Successfully loaded {len(data):,} records")
        
        # Tabs for organization
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview",
            "📈 Forecast",
            "📉 Model Evaluation",
            "💾 Export Results"
        ])
        
        # ----------------------------------------------------------------
        # TAB 1: DATA OVERVIEW
        # ----------------------------------------------------------------
        with tab1:
            st.subheader("📊 Dataset Summary")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Records",
                    value=f"{len(data):,}"
                )
            
            with col2:
                date_range = (data['Order Date'].max() - data['Order Date'].min()).days
                st.metric(
                    label="Date Range",
                    value=f"{date_range} days"
                )
            
            with col3:
                st.metric(
                    label="Total Sales",
                    value=f"${data['Sales'].sum():,.2f}"
                )
            
            with col4:
                st.metric(
                    label="Average Daily Sales",
                    value=f"${data['Sales'].mean():,.2f}"
                )
            
            st.markdown("---")
            
            # Data preview
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**First 10 Rows**")
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.markdown("**Statistical Summary**")
                st.dataframe(data['Sales'].describe(), use_container_width=True)
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("📈 Historical Sales Trends")
            
            # Time series plot
            fig = px.line(
                data,
                x='Order Date',
                y='Sales',
                title='Daily Sales Over Time'
            )
            fig.update_traces(line=dict(color='steelblue', width=1.5))
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    data,
                    x='Sales',
                    nbins=50,
                    title='Sales Distribution'
                )
                fig.update_layout(height=350, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly aggregation
                data['Month'] = data['Order Date'].dt.to_period('M').astype(str)
                monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
                
                fig = px.bar(
                    monthly_sales,
                    x='Month',
                    y='Sales',
                    title='Monthly Total Sales'
                )
                fig.update_layout(height=350, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        
        # ----------------------------------------------------------------
        # TAB 2: FORECAST
        # ----------------------------------------------------------------
        with tab2:
            st.subheader("📈 Generate Sales Forecast")
            
            # Train model
            model, forecast = train_prophet(
                data,
                periods=forecast_periods,
                seasonality_mode=seasonality_mode
            )
            
            st.success(f"✅ Model trained successfully! Forecasting {forecast_periods} days ahead.")
            
            # Forecast visualization
            fig = create_plotly_forecast(data, forecast, forecast_periods)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.markdown("### 📋 Forecast Details")
            
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
            forecast_display.columns = ['Date', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
            forecast_display['Date'] = forecast_display['Date'].dt.date
            
            st.dataframe(
                forecast_display.style.format({
                    'Predicted Sales': '${:,.2f}',
                    'Lower Bound': '${:,.2f}',
                    'Upper Bound': '${:,.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Components plot
            if show_components:
                st.markdown("---")
                st.subheader("🔍 Forecast Components")
                
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
        
        # ----------------------------------------------------------------
        # TAB 3: MODEL EVALUATION
        # ----------------------------------------------------------------
        with tab3:
            st.subheader("📉 Model Performance Evaluation")
            
            # Use last N days for evaluation if available
            if len(data) > forecast_periods:
                st.info(f"Evaluating on the last {forecast_periods} days of historical data")
                
                # Split data
                train_eval = data.iloc[:-forecast_periods]
                test_eval = data.iloc[-forecast_periods:]
                
                # Train and predict
                model_eval, forecast_eval = train_prophet(
                    train_eval,
                    periods=forecast_periods,
                    seasonality_mode=seasonality_mode
                )
                
                predictions = forecast_eval['yhat'].tail(forecast_periods).values
                actuals = test_eval['Sales'].values
                
                # Calculate metrics
                metrics = evaluate(actuals, predictions)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"${metrics['MAE']:,.2f}")
                
                with col2:
                    st.metric("RMSE", f"${metrics['RMSE']:,.2f}")
                
                with col3:
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                
                with col4:
                    st.metric("SMAPE", f"{metrics['SMAPE']:.2f}%")
                
                # Accuracy interpretation
                if metrics['MAPE'] < 10:
                    accuracy_level = "🟢 Excellent"
                elif metrics['MAPE'] < 20:
                    accuracy_level = "🟡 Good"
                else:
                    accuracy_level = "🔴 Fair"
                
                st.info(f"**Model Accuracy: {accuracy_level}** (MAPE: {metrics['MAPE']:.2f}%)")
                
                st.markdown("---")
                
                # Comparison plot
                comparison_df = pd.DataFrame({
                    'Date': test_eval['Order Date'].values,
                    'Actual': actuals,
                    'Predicted': predictions
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=comparison_df['Date'],
                    y=comparison_df['Actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=comparison_df['Date'],
                    y=comparison_df['Predicted'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Actual vs Predicted Sales',
                    xaxis_title='Date',
                    yaxis_title='Sales ($)',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals
                residuals = actuals - predictions
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        x=range(len(residuals)),
                        y=residuals,
                        title='Residual Plot',
                        labels={'x': 'Observation', 'y': 'Residual'}
                    )
                    fig.add_hline(y=0, line_dash='dash', line_color='red')
                    fig.update_layout(height=350, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        x=residuals,
                        nbins=30,
                        title='Residual Distribution',
                        labels={'x': 'Residual'}
                    )
                    fig.update_layout(height=350, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("⚠️ Not enough data for evaluation. Upload more historical data.")
        
        # ----------------------------------------------------------------
        # TAB 4: EXPORT RESULTS
        # ----------------------------------------------------------------
        with tab4:
            st.subheader("💾 Export Forecast Results")
            
            # Prepare export data
            export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
            export_df.columns = ['Date', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📄 Download Forecast")
                
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.markdown("### 📊 Preview")
                st.dataframe(export_df, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Summary Statistics")
                
                summary = pd.DataFrame({
                    'Metric': [
                        'Total Forecasted Sales',
                        'Average Daily Sales',
                        'Min Daily Sales',
                        'Max Daily Sales'
                    ],
                    'Value': [
                        f"${export_df['Predicted_Sales'].sum():,.2f}",
                        f"${export_df['Predicted_Sales'].mean():,.2f}",
                        f"${export_df['Predicted_Sales'].min():,.2f}",
                        f"${export_df['Predicted_Sales'].max():,.2f}"
                    ]
                })
                
                st.dataframe(summary, use_container_width=True, hide_index=True)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()