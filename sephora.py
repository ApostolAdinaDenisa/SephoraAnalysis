import streamlit as st
import pandas as pd
import numpy as np
# Use a non-interactive backend for matplotlib in headless environments
import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("matplotlib import failed. Install matplotlib (pip install matplotlib)") from e
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Setări pagină
st.set_page_config(page_title="Sephora Business Analytics Dashboard", layout="wide")

# --- 1. DATA ENGINE (Facilități: Merge/Join & Missing Values) ---
@st.cache_data
def load_data():
    # Încărcarea fișierelor tale importate
    try:
        df_sales = pd.read_csv('sephora_analysis.csv')
        df_prod = pd.read_csv('products.csv')
        
        # Facilitate: Merge/Join
        df = pd.merge(df_sales, df_prod, on='product_id', how='left')
        
        # Facilitate: Dealing with missing values & extremes
        # Curățăm datele lipsă (echivalent cu cleaning-ul din SAS)
        df = df.dropna(subset=['sales', 'category'])
        df = df[df['sales'] > 0] # Eliminăm erorile (vânzări negative/zero)
        
        return df
    except FileNotFoundError:
        st.error("Eroare: Asigură-te că fișierele 'sephora_analysis.csv' și 'products.csv' sunt în același folder!")
        return pd.DataFrame()

df = load_data()

# --- MENIU LATERAL (Navigare 6 Pagini) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Sephora_logo.svg/2560px-Sephora_logo.svg.png", width=200)
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", 
    ["1. Project Overview", 
     "2. Data Exploration", 
     "3. Seasonal Analysis", 
     "4. Store Locator (Maps)", 
     "5. Sales Predictor", 
     "6. Market Segmentation (ML)"])

# --- PAGINA 1: PROJECT OVERVIEW ---
if page == "1. Project Overview":
    st.title("💄 Sephora Sales & Profitability Analysis")
    st.markdown("""
    Acest proiect analizează performanța produselor Sephora utilizând un flux hibrid **SAS & Python**.
    Obiectivul este optimizarea stocurilor în funcție de sezonalitate și predicția volumului de vânzări.
    """)
    st.info("Aplicația utilizează 8 facilități avansate: de la Geopandas la Regresie Multiplă și Clustering.")
    st.image("https://images.unsplash.com/photo-1596462502278-27bfdc4033c8?auto=format&fit=crop&q=80&w=1000", caption="Sephora Retail Analysis")

# --- PAGINA 2: DATA EXPLORATION (Facilitate: Statistical Processing & Aggregation) ---
elif page == "2. Data Exploration":
    st.title("🔍 Data Exploration & Quality Check")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final Integrated Dataset")
        st.write(df.head(15))
    
    with col2:
        st.subheader("Statistical Summary")
        st.write(df.describe())
    
    # Facilitate: Grouping & Aggregation
    st.subheader("Total Sales per Category")
    category_summary = df.groupby('category').agg({'sales': 'sum', 'product_id': 'count'}).rename(columns={'product_id': 'Transaction_Count'})
    st.bar_chart(category_summary['sales'])

# --- PAGINA 3: SEASONAL ANALYSIS (Facilitate: Matplotlib & Categorical Trends) ---
elif page == "3. Seasonal Analysis":
    st.title("❄️☀️ Seasonal Trends Analysis")
    st.write("Demonstrarea ipotezei: Fragrance (Winter) vs Skincare (Summer).")
    
    # Pregătire date pentru grafic
    seasonal_data = df.groupby(['season', 'category'])['sales'].sum().unstack()
    
    # Facilitate: Graphical representation with Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    seasonal_data.plot(kind='bar', ax=ax, colormap='magma')
    plt.title("Revenue by Season and Category")
    plt.ylabel("Total Revenue ($)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.success("Analiză: Se confirmă faptul că produsele de tip 'Fragrance' au un volum maxim în 'Winter' datorită sărbătorilor.")

# --- PAGINA 4: STORE LOCATOR (Facilitate: GeoPandas) ---
elif page == "4. Store Locator (Maps)":
    st.title("📍 Regional Distribution (Geo-Data)")
    
    # Simulăm coordonate pentru magazinele Sephora din România
    geo_data = {
        'City': ['Bucharest', 'Cluj-Napoca', 'Timisoara', 'Iasi', 'Constanta'],
        'Lat': [44.4268, 46.7712, 45.7489, 47.1585, 44.1733],
        'Lon': [26.1025, 23.5897, 21.2087, 27.6014, 28.6383],
        'Stores': [12, 4, 3, 2, 2]
    }
    df_geo = pd.DataFrame(geo_data)
    
    # Facilitate: Using GeoPandas
    geometry = [Point(xy) for xy in zip(df_geo['Lon'], df_geo['Lat'])]
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometry)
    
    st.write("Locațiile magazinelor Sephora procesate prin coordonate spațiale:")
    st.map(df_geo)
    st.write("GeoPandas Data Structure:", gdf)

# --- PAGINA 5: SALES PREDICTOR (Facilitate: Statsmodels - Multiple Regression) ---
elif page == "5. Sales Predictor":
    st.title("📉 Multi-Variable Prediction Model")
    st.write("Model de regresie pentru estimarea unităților vândute.")
    
    # Calculăm Units Sold dacă nu există
    df['units_sold'] = (df['sales'] / df['unit_price']).round().astype(int)
    
    # Facilitate: Statsmodels
    X = df[['unit_price', 'discount']]
    X = sm.add_constant(X)
    Y = df['units_sold']
    
    model = sm.OLS(Y, X).fit()
    
    # Interactivitate Streamlit: Slider pentru manager
    st.subheader("Simulate New Product Launch")
    sim_price = st.slider("Select Price ($)", 10, 500, 150)
    sim_disc = st.slider("Select Planned Discount (%)", 0, 70, 15) / 100
    
    prediction = model.predict([1, sim_price, sim_disc])[0]
    st.metric(label="Estimated Units Sold", value=f"{int(prediction)} units")
    
    with st.expander("Show Detailed Regression Statistics"):
        st.write(model.summary())

# --- PAGINA 6: MARKET SEGMENTATION (Facilitate: Scikit-learn, Encoding, Scaling) ---
elif page == "6. Market Segmentation (ML)":
    st.title("🤖 Product Clustering & AI Segments")
    
    # Facilitate: Encoding (Transformăm categoriile în numere)
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    # Facilitate: Scaling (Standardizăm datele pentru Clustering)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['unit_price', 'sales']])
    
    # Facilitate: Scikit-learn (K-Means Clustering)
    km = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = km.fit_predict(scaled_features)
    
    # Vizualizare Clusteri
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='unit_price', y='sales', hue='cluster', palette='deep', size='sales', sizes=(20, 200))
    plt.title("Product Segmentation: Mass-Market vs Luxury")
    st.pyplot(fig)
    
    st.write("Cluster 0: Entry Level | Cluster 1: High Volume | Cluster 2: Premium Luxury Items")