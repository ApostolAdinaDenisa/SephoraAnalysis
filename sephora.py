import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configurare stil pagina
st.set_page_config(page_title="Sephora Analytics Dashboard", layout="wide")

# 1. DATA PREPARATION (Facilitati: Merge, Missing Values, Statistical Processing)
@st.cache_data
def load_and_clean_data():
    # Simulam datele pentru demonstratie (tu poti incarca csv-urile tale)
    data_sales = {
        'product_id': np.random.randint(100, 110, 200),
        'sales': np.random.uniform(100, 10000, 200),
        'season': np.random.choice(['Winter', 'Spring', 'Summer', 'Autumn'], 200),
        'discount': np.random.uniform(0, 0.5, 200),
        'city': np.random.choice(['Bucharest', 'Cluj', 'Timisoara', 'Iasi', 'Constanta'], 200)
    }
    data_prod = {
        'product_id': range(100, 110),
        'product_name': ['Luxury Perfume', 'Sunscreen SPF50', 'Matte Lipstick', 'Night Serum', 'Hydrating Mask', 'Eye Liner', 'Foundation', 'Body Lotion', 'Face Wash', 'Shampoo'],
        'category': ['Fragrance', 'Skincare', 'Makeup', 'Skincare', 'Skincare', 'Makeup', 'Makeup', 'Bodycare', 'Skincare', 'Haircare'],
        'unit_price': [550, 120, 180, 420, 95, 85, 220, 110, 65, 75]
    }
    
    df_sales = pd.DataFrame(data_sales)
    df_prod = pd.DataFrame(data_prod)
    
    # Facility: Merge/Join
    df = pd.merge(df_sales, df_prod, on='product_id', how='left')
    
    # Facility: Dealing with missing/extreme values
    df['sales'] = df['sales'].replace(0, np.nan)
    df = df.dropna(subset=['sales']) 
    return df

df = load_and_clean_data()

# NAVIGARE
st.sidebar.title("Sephora Analysis")
page = st.sidebar.selectbox("Alege Sectiunea:", ["Overview", "Seasonal Trends", "Geo-Distribution", "Price Predictor (Regression)", "Product Clusters (ML)"])

if page == "Overview":
    st.title("📊 Data Overview & Statistics")
    st.write("Explorarea setului de date combinat (Products + Sales)")
    
    # Facility: Statistical processing & Aggregation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primele 10 rânduri")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Statistici Descriptive")
        st.write(df.describe())

elif page == "Seasonal Trends":
    st.title("☀️❄️ Seasonal Category Analysis")
    st.write("Demonstrarea sezonalității: Fragrance iarna vs Skincare vara.")
    
    # Facility: Grouping and Aggregation
    seasonal_agg = df.groupby(['season', 'category'])['sales'].sum().unstack()
    
    # Facility: Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    seasonal_agg.plot(kind='bar', ax=ax)
    plt.title("Vânzări Totale pe Sezon și Categorie")
    plt.ylabel("Vânzări ($)")
    st.pyplot(fig)
    st.info("Observație: Se observă vârful vânzărilor de Fragrance în sezonul 'Winter'.")

elif page == "Geo-Distribution":
    st.title("📍 Store Locations & Regional Sales")
    
    # Facility: GeoPandas
    cities_coords = {
        'city': ['Bucharest', 'Cluj', 'Timisoara', 'Iasi', 'Constanta'],
        'lat': [44.43, 46.77, 45.75, 47.16, 44.17],
        'lon': [26.10, 23.59, 21.21, 27.60, 28.63]
    }
    df_geo = pd.DataFrame(cities_coords)
    geometry = [Point(xy) for xy in zip(df_geo['lon'], df_geo['lat'])]
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometry)
    
    st.write("Distribuția magazinelor Sephora (Coordonate procesate cu GeoPandas):")
    st.map(df_geo)

elif page == "Price Predictor (Regression)":
    st.title("📉 Multi-Variable Regression Analysis")
    st.write("Predicția unităților vândute în funcție de Preț și Discount.")
    
    # Facility: Statsmodels (Multiple Regression)
    df['units_sold'] = (df['sales'] / df['unit_price']).astype(int)
    X = df[['unit_price', 'discount']]
    X = sm.add_constant(X)
    Y = df['units_sold']
    
    model = sm.OLS(Y, X).fit()
    
    st.write(model.summary()) # Tabelul de regresie ca in proiectul model
    
    # Predictor Interactiv
    st.subheader("Predictor Interactiv")
    p = st.slider("Alege Prețul Unitat ($)", 50, 600, 200)
    d = st.slider("Alege Discountul (%)", 0, 50, 10) / 100
    pred = model.predict([1, p, d])[0]
    st.success(f"Estimare Unități Vândute: {int(pred)}")

elif page == "Product Clusters (ML)":
    st.title("🤖 AI Market Segmentation")
    
    # Facility: Encoding & Scaling
    le = LabelEncoder()
    df['cat_num'] = le.fit_transform(df['category'])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['unit_price', 'sales']])
    
    # Facility: Scikit-learn (KMeans)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Grafic Demonstrativ
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='unit_price', y='sales', hue='cluster', palette='viridis', ax=ax)
    plt.title("Segmentarea Produselor: Low, Mid, Luxury")
    st.pyplot(fig)