import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Use a non-interactive backend for matplotlib in headless environments
import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("matplotlib import failed. Install matplotlib (pip install matplotlib)") from e
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page settings
st.set_page_config(page_title="Sephora Business Analytics Dashboard", layout="wide")

# --- 1. DATA ENGINE (Facilități: Merge/Join & Missing Values) ---
@st.cache_data
def load_data():
    # Load CSV files
    try:
        df_sales = pd.read_csv('sephora_analysis.csv')
        df_prod = pd.read_csv('products.csv')
        
        # Feature: Merge/Join
        df = pd.merge(df_sales, df_prod, on='product_id', how='left')
        # If `discount` isn't provided in the CSVs, derive an approximate value from `event_type`
        if 'discount' not in df.columns:
            event_discount_map = {
                'VIB_Sale': 0.20,
                'VIB_Sale': 0.20,
                'Black_Friday': 0.40,
                'Christmas': 0.25,
                'Valentines': 0.15,
                'Easter': 0.10,
                'Normal': 0.00
            }
            df['discount'] = df.get('event_type').map(event_discount_map).fillna(0.0)
        
        # Feature: Dealing with missing values & extremes
        # Clean missing data (equivalent to data cleaning step)
        df = df.dropna(subset=['sales', 'category'])
        df = df[df['sales'] > 0] # Remove invalid rows (non-positive sales)
        
        return df
    except FileNotFoundError:
        st.error("Error: make sure 'sephora_analysis.csv' and 'products.csv' are in the same folder!")
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR MENU (6 Pages) ---
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", 
    ["1. Project Overview", 
     "2. Data Exploration", 
     "3. Seasonal Analysis", 
     "4. Store Locator (Maps)", 
     "5. Sales Predictor", 
    "6. Market Segmentation (ML)",
    "7. Sales Flow (Plotly)"])

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "1. Project Overview":
    st.title(" Sephora Sales & Profitability Analysis")
    st.markdown("""
    This project analyzes Sephora product performance using a hybrid SAS & Python pipeline. The primary objective is inventory optimization based on seasonality and sales volume prediction.

    **Data Import:**  
    The process began by collecting and importing two fundamental CSV datasets: `products.csv` (catalog details, categories, and prices) and `sephora_analysis.csv` (transaction records, promotional periods, and locations). Data were cleaned of null values and harmonized through joins to create a robust dataset for analysis.

    **Data Source:**  
    The datasets used in this case study are derived from public datasets available on Kaggle, which contain real product and category information for Sephora.

    **Source link:** [Kaggle - Sephora Products Dataset](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
    """)
    st.info("The app uses advanced features from GeoPandas to Multiple Regression and Clustering.")

# --- PAGE 2: DATA EXPLORATION (Feature: Statistical Processing & Aggregation) ---
elif page == "2. Data Exploration":
    st.title("🔍 Data Exploration & Quality Check")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Integrated Dataset")
        st.write(df.head(15))
    
    with col2:
        st.subheader("Statistical Summary")
        st.write(df.describe())

    st.subheader("Mean / Median / Mode")
    numeric_cols = [col for col in ['sales', 'unit_price', 'discount'] if col in df.columns]
    if numeric_cols:
        stats_rows = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                continue
            mode_values = series.mode()
            stats_rows.append({
                'Metric': col,
                'Mean': round(series.mean(), 2),
                'Median': round(series.median(), 2),
                'Mode': round(mode_values.iloc[0], 2) if not mode_values.empty else np.nan,
            })

        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df, use_container_width=True)

            st.markdown(
                """
                **Interpretation:**
                - The mean shows the average level across the selected metrics.
                - The median is useful when the data has outliers or skewed values.
                - The mode highlights the most common value and helps spot repeated pricing or sales patterns.
                - If mean and median are far apart, the distribution is likely skewed.
                """
            )
        else:
            st.info("No numeric values available for the mean/median/mode summary.")
    else:
        st.info("No numeric columns available for the mean/median/mode summary.")
    
    # Feature: Grouping & Aggregation
    st.subheader("Total Sales per Category")
    category_summary = df.groupby('category').agg({'sales': 'sum', 'product_id': 'count'}).rename(columns={'product_id': 'Transaction_Count'})
    st.bar_chart(category_summary['sales'])

    

# --- PAGE 3: SEASONAL ANALYSIS (Feature: Matplotlib & Categorical Trends) ---
elif page == "3. Seasonal Analysis":
    st.title("❄️☀️ Seasonal Trends Analysis")
    st.write("Demonstrating hypothesis: Fragrance (Winter) vs Skincare (Summer).")
    
    # Prepare data for plot
    seasonal_data = df.groupby(['season', 'category'])['sales'].sum().unstack()
    
    # Feature: Graphical representation with Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    seasonal_data.plot(kind='bar', ax=ax, colormap='magma')
    plt.title("Revenue by Season and Category")
    plt.ylabel("Total Revenue ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.success("Analysis: Fragrance products show a peak in Winter, likely due to holiday demand.")

# --- PAGE 4: STORE LOCATOR (Feature: GeoPandas) ---
elif page == "4. Store Locator (Maps)":
    st.title(" Regional Distribution (Geo-Data)")
    
    # Simulate coordinates for Sephora stores in Romania
    geo_data = {
        'City': ['Bucharest', 'Cluj-Napoca', 'Timisoara', 'Iasi', 'Constanta'],
        'Lat': [44.4268, 46.7712, 45.7489, 47.1585, 44.1733],
        'Lon': [26.1025, 23.5897, 21.2087, 27.6014, 28.6383],
        'Stores': [12, 4, 3, 2, 2]
    }
    df_geo = pd.DataFrame(geo_data)
    # Streamlit's `st.map` expects lowercase 'lat' and 'lon' column names
    df_geo = df_geo.rename(columns={'Lat': 'lat', 'Lon': 'lon'})
    # Ensure numeric types for mapping
    df_geo['lat'] = pd.to_numeric(df_geo['lat'], errors='coerce')
    df_geo['lon'] = pd.to_numeric(df_geo['lon'], errors='coerce')
    
    # Feature: Using GeoPandas (use lowercase lat/lon)
    geometry = [Point(xy) for xy in zip(df_geo['lon'], df_geo['lat'])]
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometry)

    st.write("Sephora store locations processed via spatial coordinates:")
    st.map(df_geo)
    
    st.write("GeoPandas Data Structure:", gdf)

# --- PAGE 5: SALES PREDICTOR (Feature: Statsmodels - Multiple Regression) ---
elif page == "5. Sales Predictor":
    st.title("📉 Multi-Variable Prediction Model")
    st.write("Regression model to estimate units sold.")
    
    # Check required columns and safely convert types
    required_cols = ['sales', 'unit_price', 'discount']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in dataset: {', '.join(missing)}.\nPlease check 'products.csv' and 'sephora_analysis.csv'.")
        st.stop()

    # Convert columns to numeric (in case they come as strings)
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
    df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0)

    # Calculate Units Sold (round and convert to int)
    df['units_sold'] = (df['sales'] / df['unit_price']).round().fillna(0).astype(int)

    # Feature: Statsmodels
    X = df[['unit_price', 'discount']]
    X = sm.add_constant(X)
    Y = df['units_sold']
    
    model = sm.OLS(Y, X).fit()
    
    # Streamlit interactivity: manager sliders
    st.subheader("Simulate New Product Launch")
    sim_price = st.slider("Select Price ($)", 10, 500, 150)
    sim_disc = st.slider("Select Planned Discount (%)", 0, 70, 15) / 100
    
    prediction = model.predict([1, sim_price, sim_disc])[0]
    st.metric(label="Estimated Units Sold", value=f"{int(prediction)} units")
    
    with st.expander("Show Detailed Regression Statistics"):
        metric_cols = st.columns(4)
        metric_cols[0].metric("R-squared", f"{model.rsquared:.3f}")
        metric_cols[1].metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
        metric_cols[2].metric("AIC", f"{model.aic:.1f}")
        metric_cols[3].metric("BIC", f"{model.bic:.1f}")

        st.markdown(
            """
            **Interpretare scurtă:**
            - `R-squared` arată cât din variația vânzărilor este explicată de model.
            - `Adj. R-squared` e util când compari modele cu număr diferit de variabile.
            - `AIC` și `BIC` ajută la compararea modelelor, iar valori mai mici sunt în general mai bune.
            """
        )

        coef_table = model.summary2().tables[1].reset_index().rename(columns={"index": "Variable"})
        fit_table = pd.DataFrame([
            {"Metric": "R-squared", "Value": round(model.rsquared, 4)},
            {"Metric": "Adj. R-squared", "Value": round(model.rsquared_adj, 4)},
            {"Metric": "F-statistic", "Value": round(float(model.fvalue), 4) if model.fvalue is not None else np.nan},
            {"Metric": "Prob (F-statistic)", "Value": round(float(model.f_pvalue), 4) if model.f_pvalue is not None else np.nan},
            {"Metric": "AIC", "Value": round(model.aic, 4)},
            {"Metric": "BIC", "Value": round(model.bic, 4)},
        ])

        st.markdown("**Coefficient Table**")
        st.dataframe(coef_table, use_container_width=True)
        st.markdown("**Model Fit**")
        st.dataframe(fit_table, use_container_width=True)
        st.caption("Interpretation: coefficients show direction and size of effect, while fit metrics show how well the model explains the data.")

# --- PAGE 6: MARKET SEGMENTATION (Feature: Scikit-learn, Encoding, Scaling) ---
elif page == "6. Market Segmentation (ML)":
    st.title(" Product Clustering & AI Segments")
    
    # Feature: Encoding (transform categories to numbers)
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    # Feature: Scaling (standardize features for clustering)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['unit_price', 'sales']])
    
    # Feature: Scikit-learn (K-Means Clustering)
    km = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = km.fit_predict(scaled_features)
    
    # Cluster visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='unit_price', y='sales', hue='cluster', palette='deep', size='sales', sizes=(20, 200))
    plt.title("Product Segmentation: Mass-Market vs Luxury")
    st.pyplot(fig)
    
    st.write("Cluster 0: Entry Level | Cluster 1: High Volume | Cluster 2: Premium Luxury Items")


# --- PAGE 7: SALES FLOW (Plotly Line Chart) ---
elif page == "7. Sales Flow (Plotly)":
    st.title("📈 Sales Over Time by Category (Plotly)")
    st.write("Interactive line chart showing monthly sales trends by category. You can enable the moving average for trend smoothing.")

    if df.empty:
        st.info("No data available to build the sales time series.")
    else:
        # Ensure `date` is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        time_df = df.dropna(subset=['date', 'sales', 'category']).copy()

        # Aggregate monthly sales per category (use 'ME' for month-end frequency compatible with pandas)
        monthly = time_df.groupby([pd.Grouper(key='date', freq='ME'), 'category'])['sales'].sum().reset_index()

        # Plotly line chart
        fig_line = px.line(monthly, x='date', y='sales', color='category', markers=True,
                           labels={'sales': 'Total Sales', 'date': 'Month'},
                           title='Monthly Sales by Category')

        # Option: show 3-month moving average per category
        if st.checkbox('Show 3-month moving average for trend smoothing'):
            ma_df = monthly.copy()
            ma_df['date'] = pd.to_datetime(ma_df['date'])
            pivot = ma_df.pivot(index='date', columns='category', values='sales').fillna(0)
            ma = pivot.rolling(window=3, min_periods=1).mean()
            for col in ma.columns:
                fig_line.add_scatter(x=ma.index, y=ma[col], mode='lines', name=f"{col} (3m MA)", line=dict(dash='dash'))

        st.plotly_chart(fig_line, use_container_width=True)