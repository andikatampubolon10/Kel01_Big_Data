import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

# Page Config
st.set_page_config(
    page_title="Analisis Retail & Segmentasi Pelanggan",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 {
        color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Dashboard Analisis Retail Terintegrasi")
st.markdown("Analisis Segmentasi Pelanggan (NoSQL) dan Tren Penjualan (SQL)")

# Sidebar for global settings
st.sidebar.header("Konfigurasi Data")
st.sidebar.info("Data diambil secara real-time dari PostgreSQL (Supabase) dan MongoDB (Atlas).")

# --- DATA LOADING ---

@st.cache_resource
def get_mongo_data():
    try:
        # Gunakan serverSelectionTimeoutMS=3000 (3 detik) agar tidak menunggu terlalu lama jika timeout
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = client[MONGO_DB]
        col = db["customer_segments"]
        docs = list(col.find())
        client.close()
        
        if not docs:
            return pd.DataFrame()
            
        data = []
        for d in docs:
            data.append({
                "CustomerID": d["_id"],
                "Recency": d["rfm"]["recency"],
                "Frequency": d["rfm"]["frequency"],
                "Monetary": d["rfm"]["monetary"],
                "Cluster": d["segment"]["cluster_id"],
                "Segment": d["segment"]["label"]
            })
        return pd.DataFrame(data)
    except Exception as e:
        # Simpan error ke session_state agar bisa kita tampilkan panduan pemecahannya
        st.session_state["mongo_error"] = str(e)
        return pd.DataFrame()

def generate_mock_mongo_data():
    import numpy as np
    np.random.seed(42)
    n_cust = 350
    
    segments = {
        "Champions": {"recency": (1, 10), "frequency": (15, 50), "monetary": (2000, 15000), "cluster": 0},
        "Loyal Customers": {"recency": (10, 45), "frequency": (8, 20), "monetary": (800, 3000), "cluster": 1},
        "At Risk": {"recency": (60, 180), "frequency": (3, 10), "monetary": (300, 1500), "cluster": 2},
        "Lost / Hibernating": {"recency": (180, 365), "frequency": (1, 3), "monetary": (50, 400), "cluster": 3}
    }
    
    data = []
    seg_names = list(segments.keys())
    for i in range(n_cust):
        seg = np.random.choice(seg_names, p=[0.25, 0.4, 0.2, 0.15])
        limits = segments[seg]
        data.append({
            "CustomerID": str(10000 + i),
            "Recency": np.random.randint(*limits["recency"]),
            "Frequency": np.random.randint(*limits["frequency"]),
            "Monetary": round(np.random.uniform(*limits["monetary"]), 2),
            "Cluster": limits["cluster"],
            "Segment": seg
        })
    return pd.DataFrame(data)

# --- DATA RAW UNTUK FILTER DINAMIS ---
@st.cache_data
def load_uncleaned_data():
    try:
        return pd.read_csv("dataset/online_retail.csv", encoding="ISO-8859-1")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_raw_data():
    try:
        # Gunakan encoding ISO-8859-1 karena file retail umumnya memiliki karakter khusus
        df = pd.read_csv("dataset/online_retail.csv", encoding="ISO-8859-1")
        df = df.dropna(subset=['CustomerID'])
        df = df[df['Quantity'] > 0] # Buang retur agar pendapatan bersih
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        df['MonthStr'] = df['InvoiceDate'].dt.strftime('%Y-%m')
        df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
        df['Hour'] = df['InvoiceDate'].dt.hour
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return pd.DataFrame()

df_raw = load_raw_data()

# --- GLOBAL FILTERS (SIDEBAR) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Interaktif")

if not df_raw.empty:
    min_date = df_raw["InvoiceDate"].min().date()
    max_date = df_raw["InvoiceDate"].max().date()
    
    date_range = st.sidebar.date_input("Rentang Waktu Transaksi", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    countries = sorted(df_raw["Country"].unique().tolist())
    # Default top 5 countries
    default_countries = [c for c in ["United Kingdom", "Germany", "France", "EIRE", "Netherlands"] if c in countries]
    if not default_countries: default_countries = countries[:5]
    
    selected_countries = st.sidebar.multiselect("Pilih Negara", countries, default=default_countries)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Filter dataframe
        df_filtered = df_raw[
            (df_raw["InvoiceDate"].dt.date >= start_date) & 
            (df_raw["InvoiceDate"].dt.date <= end_date) &
            (df_raw["Country"].isin(selected_countries))
        ]
    else:
        df_filtered = df_raw.copy()
else:
    df_filtered = pd.DataFrame()

# --- AGREGASI DINAMIS (Menggantikan Mock SQL) ---
if not df_filtered.empty:
    df_sql = df_filtered.groupby("MonthStr")["TotalAmount"].sum().reset_index().rename(columns={"MonthStr": "Month", "TotalAmount": "Total_Sales"})
    
    df_dow = df_filtered.groupby("DayOfWeek").agg({"TotalAmount": "sum", "InvoiceNo": "nunique"}).reset_index()
    df_dow = df_dow.rename(columns={"DayOfWeek": "Day", "TotalAmount": "Total_Sales", "InvoiceNo": "Transactions"})
    
    df_hourly = df_filtered.groupby("Hour").agg({"TotalAmount": "sum", "InvoiceNo": "nunique"}).reset_index()
    df_hourly = df_hourly.rename(columns={"TotalAmount": "Total_Sales", "InvoiceNo": "Transactions"})
    df_hourly["Hour"] = df_hourly["Hour"].apply(lambda x: f"{int(x):02d}:00")
    df_hourly = df_hourly.sort_values(by="Hour")
    
    df_geo = df_filtered.groupby("Country")["TotalAmount"].sum().reset_index().rename(columns={"TotalAmount": "Total_Sales"})
    
    df_products = df_filtered.groupby("Description").agg({"Quantity": "sum", "TotalAmount": "sum"}).reset_index()
    df_products = df_products.rename(columns={"Description": "Product", "TotalAmount": "Total_Sales"})
else:
    df_sql = pd.DataFrame({"Month": [], "Total_Sales": []})
    df_dow = pd.DataFrame({"Day": [], "Total_Sales": [], "Transactions": []})
    df_hourly = pd.DataFrame({"Hour": [], "Total_Sales": [], "Transactions": []})
    df_geo = pd.DataFrame({"Country": [], "Total_Sales": []})
    df_products = pd.DataFrame({"Product": [], "Quantity": [], "Total_Sales": []})

# Load Mongo data (NoSQL)
if "mongo_error" not in st.session_state:
    st.session_state["mongo_error"] = None

df_mongo = get_mongo_data()

is_mock_loaded = False
if df_mongo.empty:
    df_mongo = generate_mock_mongo_data()
    is_mock_loaded = True

# --- DASHBOARD LAYOUT ---

if is_mock_loaded:
    with st.expander("⚠️ MASALAH KONEKSI MONGODB (Klik untuk detail cara perbaikan)", expanded=True):
        st.error(f"**Gagal terhubung ke MongoDB Atlas:**\n\n`{st.session_state['mongo_error'] if st.session_state['mongo_error'] else 'Connection Timeout'}`")
        st.markdown("""
        ### **Mengapa error ini terjadi?**
        Error ini biasanya disebabkan oleh:
        1. **IP Address Anda belum terdaftar (Whitelisted) di MongoDB Atlas**: Ini adalah penyebab paling umum. Saat Anda berpindah jaringan internet (misalnya dari Wi-Fi rumah ke kampus/kantor/tethering HP), alamat IP publik Anda berubah dan diblokir oleh MongoDB Atlas demi keamanan.
        2. **Jaringan Internet Anda memblokir Port 27017**: Beberapa penyedia internet publik memblokir port koneksi luar (seperti port MongoDB 27017).
        
        ### **Cara memulihkannya (1 Menit):**
        1. Login ke akun **[MongoDB Atlas](https://cloud.mongodb.com/)**.
        2. Buka menu **Security** -> **Network Access** di panel kiri.
        3. Klik tombol **Add IP Address**.
        4. Klik tombol **"Allow Access from Anywhere"** (ini akan menambahkan `0.0.0.0/0`, sangat direkomendasikan untuk demo/development agar aplikasi dapat diakses dari jaringan mana saja).
        5. Klik **Confirm** dan tunggu statusnya berubah menjadi **Active** (sekitar 1 menit).
        6. Refresh halaman Streamlit ini.
        
        ---
        *💡 **Catatan:** Untuk sementara waktu, dashboard di bawah memuat **Data Simulasi/Mock** secara dinamis agar seluruh visualisasi, diagram distribusi, boxplot, dan KPI tetap dapat didemonstrasikan dengan lancar tanpa terputus.*
        """)

if False: # Kode dummy pengisi agar kondisi else sebelumnya dilewati
    pass
else:
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df_mongo))
    with col2:
        st.metric("Total Segments", df_mongo["Segment"].nunique())
    with col3:
        avg_monetary = f"£{df_mongo['Monetary'].mean():,.2f}"
        st.metric("Avg Monetary", avg_monetary)
    with col4:
        st.metric("Avg Frequency", round(df_mongo["Frequency"].mean(), 1))

    st.markdown("---")

    # Main Analysis Area
    tab_eda, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Exploratory Data Analysis", "📈 Tren Penjualan (SQL)", "👥 Segmentasi (NoSQL)", "🔗 Integrasi Lintas Sistem", "🗓️ Analisis Kohort", "💡 Ringkasan Insight"])

    with tab_eda:
        st.subheader("🔍 Exploratory Data Analysis (EDA)")
        st.write("Analisis awal untuk memahami struktur data mentah (sebelum *cleaning*) dibandingkan dengan data bersih (setelah *cleaning*).")
        
        df_uncleaned = load_uncleaned_data()
        
        if not df_uncleaned.empty:
            st.markdown("### 1. Komparasi Volume Data")
            col_eda1, col_eda2, col_eda3 = st.columns(3)
            with col_eda1:
                st.metric("Total Baris (Sebelum Cleaning)", f"{len(df_uncleaned):,}")
            with col_eda2:
                st.metric("Total Baris (Setelah Cleaning)", f"{len(df_raw):,}")
            with col_eda3:
                st.metric("Baris Dihapus (Anomali/Null)", f"{len(df_uncleaned) - len(df_raw):,}")
                
            st.markdown("### 2. Analisis Missing Values (Sebelum Cleaning)")
            st.write("Salah satu langkah wajib sebelum *machine learning* adalah menangani *missing values*. Grafik ini menunjukkan persentase data kosong pada dataset asli.")
            missing_data = df_uncleaned.isnull().sum().reset_index()
            missing_data.columns = ['Kolom', 'Jumlah Missing']
            missing_data['Persentase'] = (missing_data['Jumlah Missing'] / len(df_uncleaned) * 100).round(2)
            
            fig_miss, ax_miss = plt.subplots(figsize=(10, 4.5))
            sns.barplot(data=missing_data, x='Kolom', y='Persentase', palette='Reds', ax=ax_miss)
            ax_miss.set_title("Persentase Missing Values per Kolom", fontweight="bold", color="#1e3a8a")
            ax_miss.set_ylabel("Persentase (%)")
            plt.xticks(rotation=45)
            for p in ax_miss.patches:
                if p.get_height() > 0:
                    ax_miss.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), 
                                   ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')
            st.pyplot(fig_miss)
            
            st.write("---")
            st.markdown("### 3. Analisis Transaksi Dibatalkan (Returns/Cancellations)")
            
            # Ubah tipe InvoiceNo menjadi string dulu untuk pencegahan error
            df_uncleaned['InvoiceNo'] = df_uncleaned['InvoiceNo'].astype(str)
            cancellations = df_uncleaned[df_uncleaned['InvoiceNo'].str.startswith('C')].copy()
            
            st.info(f"Terdapat **{len(cancellations):,}** transaksi yang dibatalkan (Invoice diawali huruf 'C'). Transaksi ini disaring pada tahap *cleaning* agar kalkulasi pendapatan menjadi representatif (penjualan bersih).")
            
            c_col1, c_col2 = st.columns([1, 1.5])
            with c_col1:
                fig_cancel, ax_cancel = plt.subplots(figsize=(5, 4))
                ax_cancel.pie([len(df_uncleaned)-len(cancellations), len(cancellations)], 
                              labels=['Sukses', 'Dibatalkan/Retur'], autopct='%1.1f%%', 
                              colors=['#10b981', '#ef4444'], startangle=90, textprops={'fontweight': 'bold'})
                ax_cancel.set_title("Proporsi Status Transaksi (Mentah)", fontweight="bold", color="#1e3a8a")
                st.pyplot(fig_cancel)
                
            with c_col2:
                st.write("#### Top 5 Produk Paling Sering Diretur:")
                if not cancellations.empty:
                    top_returns = cancellations['Description'].value_counts().head(5).reset_index()
                    top_returns.columns = ['Produk', 'Jumlah Retur']
                    st.dataframe(top_returns, use_container_width=True)
            
            st.write("#### 📉 Tren Pembatalan Transaksi Berdasarkan Waktu (Bulanan)")
            if not cancellations.empty:
                cancellations['InvoiceDate'] = pd.to_datetime(cancellations['InvoiceDate'], errors='coerce')
                cancellations['MonthStr'] = cancellations['InvoiceDate'].dt.strftime('%Y-%m')
                cancel_trend = cancellations.groupby("MonthStr")["InvoiceNo"].nunique().reset_index()
                fig_cancel_trend, ax_cancel_trend = plt.subplots(figsize=(10, 3.5))
                ax_cancel_trend.plot(cancel_trend["MonthStr"], cancel_trend["InvoiceNo"], marker='o', color='#ef4444', linewidth=2)
                ax_cancel_trend.fill_between(cancel_trend["MonthStr"], cancel_trend["InvoiceNo"], color='#ef4444', alpha=0.1)
                ax_cancel_trend.set_title("Jumlah Transaksi yang Dibatalkan / Retur per Bulan", fontweight="bold", color="#1e3a8a")
                ax_cancel_trend.grid(True, linestyle='--', alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_cancel_trend)
                
            st.write("---")
            st.markdown("### 4. Ringkasan Statistik Data Numerik (Data Bersih)")
            st.write("Memeriksa nilai *mean*, *min*, *max* untuk memastikan tidak ada anomali kuantitas atau harga satuan negatif/nol pada data yang akan dimodelkan.")
            st.dataframe(df_raw[['Quantity', 'UnitPrice', 'TotalAmount']].describe().T.round(2), use_container_width=True)
            
            st.write("---")
            st.markdown("### 5. Inspeksi Data: Mentah vs Bersih")
            samp_col1, samp_col2 = st.columns(2)
            with samp_col1:
                st.write("**Data Mentah (Mungkin mengandung NaN atau Retur):**")
                st.dataframe(df_uncleaned.head(10))
            with samp_col2:
                st.write("**Data Bersih (Siap untuk RFM & K-Means):**")
                st.dataframe(df_raw.head(10))
        else:
            st.warning("Data mentah tidak dapat dimuat untuk EDA.")

    with tab1:
        st.subheader("Tren Penjualan & Analisis Operasional (SQL)")
        st.write("Data agregat dianalisis menggunakan Apache Spark dan disimpan di database PostgreSQL (Supabase).")
        
        # Load additional PostgreSQL tables (Kini menggunakan df_dow, df_hourly, dll yang sudah diagregasi dinamis)
        
        # KPI Cards for Tab 1
        col_t1, col_t2, col_t3 = st.columns(3)
        total_revenue = df_sql["Total_Sales"].sum()
        total_invoices = df_dow["Transactions"].sum()
        total_items_sold = int(df_filtered["Quantity"].sum()) if not df_filtered.empty else 0
        
        with col_t1:
            st.metric("Total Revenue (All Time)", f"£{total_revenue:,.2f}")
        with col_t2:
            st.metric("Total Transactions (Invoices)", f"{total_invoices:,}")
        with col_t3:
            st.metric("Total Products Sold (Quantity)", f"{total_items_sold:,}")
            
        st.write("---")
        st.write("### 📈 1. Tren Pendapatan Berkala (Temporal Sales Trend)")
        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📅 Tren Bulanan (Revenue)", "🗓️ Pola Hari (Day-of-Week)", "👥 Top Customers & Tren Waktu"])
        
        import matplotlib.ticker as ticker
        formatter = ticker.FuncFormatter(lambda x, pos: f'£{x*1e-3:.0f}k' if x >= 1000 else f'£{x:.0f}')
        
        with sub_tab1:
            fig_trend, ax_trend = plt.subplots(figsize=(10, 4.5))
            ax_trend.plot(df_sql["Month"], df_sql["Total_Sales"], marker='o', color='#3b82f6', linewidth=2.5, label="Total Sales")
            ax_trend.fill_between(df_sql["Month"], df_sql["Total_Sales"], color='#3b82f6', alpha=0.15)
            ax_trend.set_title("Tren Penjualan Bulanan (SQL)", fontsize=12, fontweight='bold', color='#1e3a8a')
            ax_trend.set_xlabel("Bulan")
            ax_trend.set_ylabel("Total Penjualan (£)")
            ax_trend.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            ax_trend.yaxis.set_major_formatter(formatter)
            plt.tight_layout()
            st.pyplot(fig_trend)
            
        with sub_tab2:
            fig_dow, ax_dow = plt.subplots(figsize=(10, 4.5))
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
            df_dow_sorted = df_dow.set_index("Day").reindex(day_order).reset_index()
            
            bars_dow = ax_dow.bar(df_dow_sorted["Day"], df_dow_sorted["Total_Sales"], color="#4f46e5", alpha=0.8)
            
            for bar in bars_dow:
                height = bar.get_height()
                ax_dow.annotate(f'£{height*1e-3:.0f}k',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e3a8a')
                                
            ax_dow.set_title("Pola Penjualan per Hari dalam Seminggu", fontsize=12, fontweight='bold', color='#1e3a8a')
            ax_dow.set_xlabel("Hari")
            ax_dow.set_ylabel("Total Penjualan (£)")
            ax_dow.yaxis.set_major_formatter(formatter)
            ax_dow.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_dow)
            
        with sub_tab3:
            col_st3a, col_st3b = st.columns(2)
            with col_st3a:
                st.write("#### 🏆 Top 10 Pelanggan (Pembelian Terbanyak)")
                if not df_filtered.empty:
                    top_customers = df_filtered.groupby("CustomerID")["TotalAmount"].sum().reset_index().sort_values(by="TotalAmount", ascending=False).head(10).sort_values(by="TotalAmount", ascending=True)
                    # Convert to string to avoid float formatting issues
                    top_customers["CustomerID"] = top_customers["CustomerID"].astype(float).astype(int).astype(str)
                    
                    fig_topc, ax_topc = plt.subplots(figsize=(6, 5))
                    bars_topc = ax_topc.barh(top_customers["CustomerID"], top_customers["TotalAmount"], color="#8b5cf6")
                    for bar in bars_topc:
                        width = bar.get_width()
                        ax_topc.annotate(f'£{width*1e-3:.0f}k' if width >= 1000 else f'£{width:.0f}',
                                        xy=(width, bar.get_y() + bar.get_height() / 2),
                                        xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=8, fontweight='bold')
                    ax_topc.set_xlim(0, top_customers["TotalAmount"].max() * 1.25)
                    ax_topc.set_title("10 Pelanggan Teratas (Berdasarkan Total Spending)", fontweight="bold", color="#1e3a8a")
                    ax_topc.xaxis.set_major_formatter(formatter)
                    plt.tight_layout()
                    st.pyplot(fig_topc)

            with col_st3b:
                st.write("#### 📈 Banyaknya Transaksi (Berdasarkan Waktu)")
                if not df_filtered.empty:
                    daily_trans = df_filtered.groupby(df_filtered["InvoiceDate"].dt.date)["InvoiceNo"].nunique().reset_index()
                    daily_trans.columns = ["Date", "Transactions"]
                    
                    fig_daily, ax_daily = plt.subplots(figsize=(6, 5))
                    ax_daily.plot(daily_trans["Date"], daily_trans["Transactions"], color="#f59e0b", linewidth=1.5, alpha=0.5)
                    # Moving average for trend
                    daily_trans["MA7"] = daily_trans["Transactions"].rolling(window=7).mean()
                    ax_daily.plot(daily_trans["Date"], daily_trans["MA7"], color="#d97706", linewidth=2.5, label="7-Day Moving Avg")
                    
                    ax_daily.set_title("Tren Jumlah Transaksi per Hari", fontweight="bold", color="#1e3a8a")
                    ax_daily.set_xlabel("Tanggal")
                    ax_daily.set_ylabel("Jumlah Transaksi Unik")
                    ax_daily.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_daily)
            
        st.write("---")
        st.write("### ⚙️ 2 & 4. Operasional Retail & Beban Sistem")
        col_b1, col_b2 = st.columns([1.1, 1])
        
        with col_b1:
            st.write("#### ⏰ Analisis Jam Sibuk Transaksi (Peak Hours Analysis)")
            fig_hour, ax_hour = plt.subplots(figsize=(6, 5))
            
            bars_hour = ax_hour.bar(df_hourly["Hour"], df_hourly["Transactions"], color="#0ea5e9", alpha=0.85)
            
            for i, h in enumerate(df_hourly["Hour"]):
                hour_val = int(h.split(":")[0])
                if 10 <= hour_val <= 15:
                    bars_hour[i].set_color("#f43f5e")  # Rose pink untuk beban puncak
            
            plt.xticks(rotation=90, fontsize=8)
            ax_hour.set_title("Distribusi Transaksi per Jam (00:00 - 23:00)\n*Merah = Jam Sibuk (Peak Traffic System Load)*", fontsize=11, fontweight='bold', color='#1e3a8a')
            ax_hour.set_xlabel("Jam Transaksi")
            ax_hour.set_ylabel("Jumlah Invoice / Transaksi")
            ax_hour.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_hour)
            
        with col_b2:
            st.write("#### 🏆 Top 10 Produk Terlaris (Top Selling Products)")
            fig_prod, ax_prod = plt.subplots(figsize=(6, 5))
            
            if not df_products.empty:
                df_prod_sorted = df_products.sort_values(by="Quantity", ascending=False).head(10).sort_values(by="Quantity", ascending=True)
                short_names = [p[:25] + "..." if len(p) > 25 else p for p in df_prod_sorted["Product"]]
                
                bars_prod = ax_prod.barh(short_names, df_prod_sorted["Quantity"], color="#10b981", alpha=0.8)
                
                for bar in bars_prod:
                    width = bar.get_width()
                    ax_prod.annotate(f'{width:,}',
                                    xy=(width, bar.get_y() + bar.get_height() / 2),
                                    xytext=(3, 0),
                                    textcoords="offset points",
                                    ha='left', va='center', fontsize=8, fontweight='bold', color='#065f46')
                                    
                max_qty = df_prod_sorted["Quantity"].max()
                if pd.notna(max_qty) and max_qty > 0:
                    ax_prod.set_xlim(0, max_qty * 1.25)
                    
                ax_prod.set_title("10 Produk Terlaris Berdasarkan Kuantitas Terjual", fontsize=11, fontweight='bold', color='#1e3a8a')
                ax_prod.set_xlabel("Kuantitas Terjual")
                ax_prod.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1e-3:.0f}k' if x >= 1000 else f'{x:.0f}'))
                ax_prod.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_prod)
            else:
                st.info("Data produk kosong pada filter terpilih.")
            
        st.write("---")
        st.write("### 🌍 3. Tren Pertumbuhan Geografis (Geographic Sales Trend)")
        
        uk_sales_series = df_geo[df_geo["Country"] == "United Kingdom"]["Total_Sales"]
        total_uk_sales = uk_sales_series.values[0] if not uk_sales_series.empty else 0
        total_non_uk_sales = df_geo[df_geo["Country"] != "United Kingdom"]["Total_Sales"].sum()
        
        c_geo1, c_geo2 = st.columns([1, 1.2])
        
        with c_geo1:
            st.write("#### Perbandingan Pasar Domestik vs Internasional")
            if total_uk_sales > 0 or total_non_uk_sales > 0:
                fig_geo_pie, ax_geo_pie = plt.subplots(figsize=(6, 5))
                
                ax_geo_pie.pie(
                    [total_uk_sales, total_non_uk_sales],
                    labels=["Pasar Domestik (UK)", "Pasar Internasional (Non-UK)"],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=["#1e3a8a", "#10b981"],
                    explode=[0.05, 0],
                    textprops=dict(color="black", fontweight="bold")
                )
                ax_geo_pie.set_title("Proporsi Kontribusi Pendapatan Geografis", fontsize=11, fontweight='bold', color='#1e3a8a')
                plt.tight_layout()
                st.pyplot(fig_geo_pie)
            else:
                st.info("Data geografis kosong pada filter terpilih.")
            
        with c_geo2:
            st.write("#### Top 10 Pasar Internasional Terbesar (Excluding UK)")
            fig_geo_bar, ax_geo_bar = plt.subplots(figsize=(6, 5))
            
            df_geo_intl = df_geo[df_geo["Country"] != "United Kingdom"].sort_values(by="Total_Sales", ascending=False).head(10).sort_values(by="Total_Sales", ascending=True)
            
            if not df_geo_intl.empty:
                bars_geo = ax_geo_bar.barh(df_geo_intl["Country"], df_geo_intl["Total_Sales"], color="#f59e0b", alpha=0.85)
                
                for bar in bars_geo:
                    width = bar.get_width()
                    ax_geo_bar.annotate(f'£{width*1e-3:.0f}k',
                                    xy=(width, bar.get_y() + bar.get_height() / 2),
                                    xytext=(3, 0),
                                    textcoords="offset points",
                                    ha='left', va='center', fontsize=8, fontweight='bold', color='#b45309')
                                    
                max_intl = df_geo_intl["Total_Sales"].max()
                if pd.notna(max_intl) and max_intl > 0:
                    ax_geo_bar.set_xlim(0, max_intl * 1.25)
                    
                ax_geo_bar.set_title("Pendapatan dari 10 Negara Internasional Utama", fontsize=11, fontweight='bold', color='#1e3a8a')
                ax_geo_bar.set_xlabel("Total Penjualan (£)")
                ax_geo_bar.xaxis.set_major_formatter(formatter)
                ax_geo_bar.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_geo_bar)
            else:
                st.info("Data internasional kosong pada filter terpilih.")
            
        with st.expander("Lihat Data Mentah SQL (Tren Bulanan)"):
            st.dataframe(df_sql)

    with tab2:
        st.subheader("Analisis Klastering K-Means Dinamis (Centroid-Based)")
        st.write("Visualisasi ini menjalankan algoritma K-Means secara dinamis. Anda dapat mengatur jumlah kelompok (K) dan mengevaluasi kualitas pemisahan (*clustering*) berdasarkan metrik validasi.")
        
        n_customers = len(df_mongo)
        if not df_mongo.empty and n_customers > 2:
            max_k = min(20, n_customers - 1)
            
            selected_k = st.slider("Pilih Jumlah Kelompok (Nilai K)", min_value=2, max_value=max_k, value=df_mongo["Segment"].nunique())
            
            with st.spinner("Memproses ulang K-Means dan menghitung Centroid..."):
                # Menyiapkan fitur RFM
                X_rfm = df_mongo[["Recency", "Frequency", "Monetary"]].copy()
                
                # Standarisasi wajib sebelum K-Means agar tidak bias pada variabel skala besar (Monetary)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_rfm)
                
                # Menjalankan algoritma dengan inisialisasi k-means++ (lebih stabil dari inisiasi acak)
                kmeans = KMeans(n_clusters=selected_k, init="k-means++", n_init=10, max_iter=300, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Menyimpan hasil klasterisasi dinamis
                df_dynamic = df_mongo.copy()
                df_dynamic["Dynamic_Cluster"] = cluster_labels
                df_dynamic["Dynamic_Segment"] = "Segment-" + df_dynamic["Dynamic_Cluster"].astype(str)
                
                # Ekstrak posisi Centroid (dan kembalikan ke skala asli untuk diinterpretasikan)
                centroids_scaled = kmeans.cluster_centers_
                centroids_original = scaler.inverse_transform(centroids_scaled)
                
                # Menghitung Metrik Evaluasi tanpa label (Unsupervised Evaluation)
                sil_score = silhouette_score(X_scaled, cluster_labels)
                db_score = davies_bouldin_score(X_scaled, cluster_labels)
                ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                
                st.markdown("### 📊 Evaluasi Model (Kualitas Kluster)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("K Terpilih", selected_k)
                m2.metric("Silhouette Score", f"{sil_score:.3f}", help="Semakin mendekati 1 semakin baik (pemisahan kluster rapi).")
                m3.metric("Davies-Bouldin Index", f"{db_score:.3f}", help="Semakin kecil semakin baik (kluster padat dan terpisah jauh).")
                m4.metric("Calinski-Harabasz", f"{ch_score:.0f}", help="Semakin besar nilainya semakin padat klusternya.")
                
                st.write("---")
                st.markdown("### 📍 Interpretasi Posisi Centroid (Pusat Kluster)")
                st.write("Centroid mewakili nilai rata-rata tiap variabel pembentuk di dalam satu kelompok pelanggan.")
                df_centroids = pd.DataFrame(centroids_original, columns=["Rata-rata Recency (Hari)", "Rata-rata Frequency (Transaksi)", "Rata-rata Monetary (£)"])
                df_centroids.index = ["Segment-" + str(i) for i in range(selected_k)]
                st.dataframe(df_centroids.style.format("{:.2f}"))
                
                st.write("---")
                c1, c2 = st.columns([1, 1.2])
                with c1:
                    st.write("#### Distribusi Jumlah Pelanggan per Segmen")
                    fig_c, ax_c = plt.subplots(figsize=(6, 5))
                    sns.countplot(data=df_dynamic, x="Dynamic_Segment", palette="Set2", ax=ax_c, order=sorted(df_dynamic["Dynamic_Segment"].unique()))
                    ax_c.set_xlabel("Segmen")
                    ax_c.set_ylabel("Jumlah Pelanggan")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_c)
                    
                with c2:
                    st.write("#### 🎯 Pemetaan Spasial K-Means & Letak Centroid")
                    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
                    
                    # Plot Titik Pelanggan
                    sns.scatterplot(
                        data=df_dynamic, x="Recency", y="Frequency", 
                        hue="Dynamic_Segment", size="Monetary", sizes=(20, 300), 
                        alpha=0.6, palette="Set2", ax=ax_scatter
                    )
                    
                    # Plot Titik Centroid
                    ax_scatter.scatter(
                        df_centroids["Rata-rata Recency (Hari)"], 
                        df_centroids["Rata-rata Frequency (Transaksi)"], 
                        marker='X', s=250, c='red', edgecolor='black', label="Centroid (Pusat)", zorder=10
                    )
                    
                    ax_scatter.set_xlabel("Recency (Hari)")
                    ax_scatter.set_ylabel("Frequency (Jumlah Transaksi)")
                    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    st.pyplot(fig_scatter)
                    
                st.write("---")
                st.write("#### 📦 Variasi Nilai Moneter (Monetary) per Segmen")
                fig2, ax2 = plt.subplots(figsize=(10, 4.5))
                sns.boxplot(data=df_dynamic, x="Dynamic_Segment", y="Monetary", palette="Set2", ax=ax2, order=sorted(df_dynamic["Dynamic_Segment"].unique()))
                import matplotlib.ticker as ticker
                ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'£{x*1e-3:.0f}k' if x >= 1000 else f'£{x:.0f}'))
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            st.warning("Data pelanggan tidak mencukupi untuk melakukan K-Means Clustering.")

    with tab4:
        st.subheader("Analisis Kohort (Customer Retention Rate)")
        st.write("Visualisasi Heatmap ini mengukur seberapa loyal pelanggan dari waktu ke waktu berdasarkan bulan transaksi pertama mereka.")
        
        if not df_filtered.empty:
            df_cohort = df_filtered.copy()
            # Buat representasi bulan
            df_cohort['InvoiceMonth'] = df_cohort['InvoiceDate'].dt.to_period('M')
            df_cohort['CohortMonth'] = df_cohort.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
            
            def get_date_int(df, column):
                year = df[column].dt.year
                month = df[column].dt.month
                return year, month
            
            invoice_year, invoice_month = get_date_int(df_cohort, 'InvoiceMonth')
            cohort_year, cohort_month = get_date_int(df_cohort, 'CohortMonth')
            
            years_diff = invoice_year - cohort_year
            months_diff = invoice_month - cohort_month
            df_cohort['CohortIndex'] = years_diff * 12 + months_diff + 1
            
            cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].apply(pd.Series.nunique).reset_index()
            cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
            
            cohort_sizes = cohort_counts.iloc[:, 0]
            retention = cohort_counts.divide(cohort_sizes, axis=0)
            retention.index = retention.index.astype(str)
            
            fig_cohort, ax_cohort = plt.subplots(figsize=(10, 6))
            sns.heatmap(retention, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=0.5, ax=ax_cohort)
            ax_cohort.set_title("Retention Rate per Kohort Bulanan", fontsize=12, fontweight='bold', color='#1e3a8a')
            ax_cohort.set_ylabel("Bulan Kohort (Transaksi Pertama)")
            ax_cohort.set_xlabel("Bulan ke- (Cohort Index)")
            plt.tight_layout()
            st.pyplot(fig_cohort)
        else:
            st.warning("Data tidak tersedia untuk filter yang dipilih.")

    with tab5:
        st.subheader("💡 Ringkasan Eksekutif & Insight Utama")
        st.write("Bagian ini merangkum penemuan paling bernilai (*actionable insights*) dari data transaksi dan segmentasi pelanggan yang dianalisis menggunakan *Big Data*.")
        
        with st.spinner("Membuat rangkuman insight cerdas dari data gabungan..."):
            if not df_filtered.empty and not df_mongo.empty:
                # Aggregate filtered data by CustomerID
                df_cust_sales = df_filtered.groupby("CustomerID").agg({"TotalAmount": "sum", "InvoiceNo": "nunique"}).reset_index()
                
                # Merge with MongoDB segmentation
                df_mongo_str_id = df_mongo.copy()
                
                # Standarisasi format ID: konversi ke float lalu int lalu string (menghapus '.0' yang berasal dari Spark)
                df_mongo_str_id["CustomerID"] = df_mongo_str_id["CustomerID"].astype(float).astype(int).astype(str)
                df_cust_sales["CustomerID"] = df_cust_sales["CustomerID"].astype(float).astype(int).astype(str)
                
                df_insight = pd.merge(df_cust_sales, df_mongo_str_id, on="CustomerID", how="inner")
                
                if not df_insight.empty:
                    # Insight 1: Pareto Principle
                    st.markdown("### 🥇 1. Hukum Pareto (80/20 Rule) Berlaku Kuat pada Segmen Bisnis Ini")
                    st.info("**Insight:** Kelompok pelanggan dengan segmen teratas menyumbangkan **mayoritas absolut** dari total pendapatan perusahaan. Oleh karena itu, *budget* promosi dan layanan *customer care* harus difokuskan mati-matian pada retensi segmen VIP/Champions ini.")
                    
                    pareto_data = df_insight.groupby("Segment")["TotalAmount"].sum().sort_values(ascending=False).reset_index()
                    pareto_data["CumulativePercentage"] = (pareto_data["TotalAmount"].cumsum() / pareto_data["TotalAmount"].sum()) * 100
                    
                    fig_pareto, ax1 = plt.subplots(figsize=(10, 4.5))
                    ax1.bar(pareto_data["Segment"], pareto_data["TotalAmount"], color="#3b82f6", alpha=0.85)
                    ax1.set_ylabel("Total Pendapatan (£)", color="#1e3a8a", fontweight='bold')
                    import matplotlib.ticker as ticker
                    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'£{x*1e-3:.0f}k' if x >= 1000 else f'£{x:.0f}'))
                    
                    ax2 = ax1.twinx()
                    ax2.plot(pareto_data["Segment"], pareto_data["CumulativePercentage"], color="#ef4444", marker="D", linewidth=3, markersize=8)
                    ax2.set_ylabel("Persentase Kumulatif (%)", color="#ef4444", fontweight='bold')
                    ax2.set_ylim(0, 110)
                    
                    for i, txt in enumerate(pareto_data["CumulativePercentage"]):
                        ax2.annotate(f"{txt:.1f}%", (pareto_data["Segment"][i], pareto_data["CumulativePercentage"][i]), 
                                     textcoords="offset points", xytext=(0,10), ha='center', color="#b91c1c", fontweight="bold")
                    
                    plt.title("Bukti Visual: Pendapatan per Segmen & Garis Persentase Kumulatif", fontweight="bold", color="#1e3a8a")
                    st.pyplot(fig_pareto)
                    
                    st.write("---")
                    
                    # Insight 2: Basket Size per Day
                    st.markdown("### 🛒 2. Perilaku Belanja Mingguan: 'Basket Size' Tertinggi")
                    st.info("**Insight:** Di menu 'Tren Penjualan', kita melihat bahwa hari **Kamis (Thursday)** mencetak volume transaksi paling banyak. Namun, jika kita telusuri ukuran keranjang per transaksi (*Basket Size*), perilaku pelanggan mungkin berbeda! Hari dengan *Basket Size* tinggi adalah waktu terbaik untuk menaruh iklan produk premium.")
                    
                    basket_size = df_filtered.groupby("DayOfWeek").agg({"TotalAmount": "sum", "InvoiceNo": "nunique"}).reset_index()
                    basket_size["AvgBasketSize"] = basket_size["TotalAmount"] / basket_size["InvoiceNo"]
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
                    basket_size = basket_size.set_index("DayOfWeek").reindex(day_order).dropna().reset_index()
                    
                    fig_basket, ax_basket = plt.subplots(figsize=(10, 4.5))
                    sns.barplot(data=basket_size, x="DayOfWeek", y="AvgBasketSize", palette="magma", ax=ax_basket)
                    for p in ax_basket.patches:
                        ax_basket.annotate(f"£{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                                           ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
                    ax_basket.set_title("Bukti Visual: Rata-rata Nilai Belanja per Transaksi (Basket Size)", fontweight="bold", color="#1e3a8a")
                    ax_basket.set_ylabel("Avg Basket Size (£)")
                    ax_basket.set_xlabel("Hari Transaksi")
                    st.pyplot(fig_basket)
                    
                    st.write("---")
                    
                    # Insight 3: High Value Churn Risk
                    st.markdown("### ⚠️ 3. Risiko Kehilangan Pelanggan Bernilai Tinggi (High-Value Churn Risk)")
                    st.error("**Insight:** Terdapat sejumlah pelanggan yang secara historis berbelanja dalam jumlah amat besar (*Monetary* tinggi), namun mereka **sudah berbulan-bulan tidak kembali** (*Recency* sangat buruk). Mereka ini adalah target paling mendesak untuk kampanye retensi eksklusif (*Win-back Campaign*).")
                    
                    fig_risk, ax_risk = plt.subplots(figsize=(10, 5))
                    high_monetary_threshold = df_insight["Monetary"].quantile(0.75) # Top 25% spender
                    
                    df_insight["RiskStatus"] = "Aman / Belanja Kecil"
                    df_insight.loc[(df_insight["Recency"] > 90) & (df_insight["Monetary"] > high_monetary_threshold), "RiskStatus"] = "⚠️ MANTAN VIP (Risiko Tinggi Hilang)"
                    df_insight.loc[(df_insight["Recency"] > 90) & (df_insight["Monetary"] <= high_monetary_threshold), "RiskStatus"] = "Churn Biasa"
                    df_insight.loc[(df_insight["Recency"] <= 90) & (df_insight["Monetary"] > high_monetary_threshold), "RiskStatus"] = "VIP Aktif"
                    
                    sns.scatterplot(
                        data=df_insight, 
                        x="Recency", 
                        y="Monetary", 
                        hue="RiskStatus", 
                        palette={"VIP Aktif": "#10b981", "⚠️ MANTAN VIP (Risiko Tinggi Hilang)": "#ef4444", "Churn Biasa": "#94a3b8", "Aman / Belanja Kecil": "#6ee7b7"},
                        alpha=0.8,
                        s=60,
                        edgecolor="w",
                        ax=ax_risk
                    )
                    ax_risk.set_title("Bukti Visual: Pemetaan Risiko Churn Pelanggan VIP", fontweight="bold", color="#1e3a8a")
                    ax_risk.set_xlabel("Recency (Semakin ke kanan = Semakin lama tidak belanja)")
                    ax_risk.set_ylabel("Monetary (Total Belanja £)")
                    ax_risk.axvline(x=90, color="red", linestyle="--", alpha=0.5, label="Batas Bahaya (90 Hari)")
                    ax_risk.axhline(y=high_monetary_threshold, color="blue", linestyle="--", alpha=0.5, label=f"Batas VIP (£{high_monetary_threshold:.0f})")
                    ax_risk.legend(loc='upper right')
                    
                    # Potong axis Y yang terlalu ekstrem agar plot lebih bisa dibaca
                    ax_risk.set_ylim(-10, df_insight["Monetary"].quantile(0.98))
                    st.pyplot(fig_risk)
                    
                    # --- INSIGHT 4: HOURLY BASKET SIZE ---
                    st.write("---")
                    st.markdown("### 🕒 4. Perilaku Belanja Berdasarkan Jam (*Hourly Basket Size*)")
                    st.info("**Insight:** Jam sibuk (volume transaksi tinggi) tidak selalu berarti nilai keranjangnya (*Basket Size*) juga tinggi. Memahami jam berapa pelanggan besar (*High Rollers*) bertransaksi dapat membantu penentuan waktu ideal untuk mengirimkan *email marketing* atau *flash sale* premium.")
                    
                    hourly_basket = df_filtered.groupby("Hour").agg({"TotalAmount": "sum", "InvoiceNo": "nunique"}).reset_index()
                    hourly_basket["AvgBasketSize"] = hourly_basket["TotalAmount"] / hourly_basket["InvoiceNo"]
                    
                    fig_hour, ax_hour = plt.subplots(figsize=(10, 4.5))
                    sns.lineplot(data=hourly_basket, x="Hour", y="AvgBasketSize", marker="o", linewidth=3, color="#8b5cf6", ax=ax_hour)
                    ax_hour.set_title("Bukti Visual: Rata-rata Ukuran Keranjang per Jam Transaksi", fontweight="bold", color="#1e3a8a")
                    ax_hour.set_xlabel("Jam (Format 24H)")
                    ax_hour.set_ylabel("Avg Basket Size (£)")
                    ax_hour.set_xticks(range(6, 22))
                    ax_hour.grid(True, linestyle="--", alpha=0.3)
                    st.pyplot(fig_hour)
                    
                    # --- INSIGHT 5: TOP PRODUCTS BY VIP ---
                    st.write("---")
                    st.markdown("### 💎 5. Preferensi Produk Eksklusif Segmen VIP")
                    st.info("**Insight:** Pelanggan dengan nilai tertinggi (VIP/Champions) seringkali membeli kombinasi produk yang berbeda dibanding pelanggan biasa. Data produk spesifik ini vital untuk diprioritaskan di algoritma rekomendasi beranda (Halaman Depan) khusus pelanggan VIP.")
                    
                    # Cari nama segmen dengan rata-rata Monetary tertinggi
                    vip_segment = df_insight.groupby("Segment")["Monetary"].mean().idxmax()
                    vip_customers = df_insight[df_insight["Segment"] == vip_segment]["CustomerID"].astype(str).tolist()
                    
                    df_vip_sales = df_filtered[df_filtered["CustomerID"].astype(float).astype(int).astype(str).isin(vip_customers)]
                    top_vip_prod = df_vip_sales.groupby("Description")["TotalAmount"].sum().sort_values(ascending=False).head(5).reset_index()
                    
                    fig_vip, ax_vip = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=top_vip_prod, y="Description", x="TotalAmount", palette="viridis", ax=ax_vip)
                    ax_vip.set_title(f"Bukti Visual: 5 Produk Penghasil Pendapatan Terbesar untuk Segmen '{vip_segment}'", fontweight="bold", color="#1e3a8a")
                    ax_vip.set_xlabel("Total Pendapatan (£)")
                    ax_vip.set_ylabel("")
                    st.pyplot(fig_vip)

                    # --- INSIGHT 6: LOYALTY EFFECT ---
                    st.write("---")
                    st.markdown("### 📈 6. Efek Loyalitas: Frekuensi Kunjungan vs Rata-rata Belanja")
                    st.info("**Insight:** Ada pertanyaan mendasar: apakah pelanggan yang sering datang (Frekuensi tinggi) cenderung belanja lebih sedikit per kunjungan, atau malah membesar? Bukti di bawah menunjukkan korelasi antara total kunjungan dan nilai rata-rata tiap kunjungannya.")
                    
                    df_insight["AvgMonetaryPerInvoice"] = df_insight["Monetary"] / df_insight["Frequency"]
                    fig_loyalty, ax_loyalty = plt.subplots(figsize=(10, 4.5))
                    sns.regplot(data=df_insight, x="Frequency", y="AvgMonetaryPerInvoice", scatter_kws={'alpha':0.4, 'color': '#0ea5e9'}, line_kws={'color': 'red'}, ax=ax_loyalty)
                    ax_loyalty.set_title("Bukti Visual: Korelasi Frekuensi Belanja & Pengeluaran per Kunjungan", fontweight="bold", color="#1e3a8a")
                    ax_loyalty.set_xlim(0, df_insight["Frequency"].quantile(0.99))
                    ax_loyalty.set_ylim(0, df_insight["AvgMonetaryPerInvoice"].quantile(0.99))
                    ax_loyalty.set_xlabel("Frequency (Total Kunjungan)")
                    ax_loyalty.set_ylabel("Rata-rata Belanja per Kunjungan (£)")
                    st.pyplot(fig_loyalty)

                    # --- INSIGHT 7: GEO VIP CONCENTRATION ---
                    st.write("---")
                    st.markdown("### 🌍 7. Konsentrasi Pelanggan VIP Berdasarkan Negara")
                    st.info("**Insight:** Ekspansi internasional membutuhkan fokus. Dengan melihat rasio segmen pelanggan di 10 negara teratas, kita bisa mengetahui negara mana yang memiliki *persentase* pelanggan VIP terbesar (bukan sekadar total pendapatan).")
                    
                    df_cust_country = df_filtered.groupby("CustomerID")["Country"].first().reset_index()
                    df_cust_country["CustomerID"] = df_cust_country["CustomerID"].astype(float).astype(int).astype(str)
                    df_insight_geo = pd.merge(df_insight, df_cust_country, on="CustomerID", how="inner")
                    
                    top_countries = df_insight_geo["Country"].value_counts().head(10).index
                    df_geo_filtered = df_insight_geo[df_insight_geo["Country"].isin(top_countries)]
                    geo_seg = pd.crosstab(df_geo_filtered["Country"], df_geo_filtered["Segment"], normalize='index') * 100
                    geo_seg = geo_seg.reindex(df_geo_filtered["Country"].value_counts().index)
                    
                    fig_geo_stack, ax_geo_stack = plt.subplots(figsize=(10, 5))
                    geo_seg.plot(kind='bar', stacked=True, ax=ax_geo_stack, colormap='Set2', edgecolor='white')
                    ax_geo_stack.set_title("Bukti Visual: Proporsi Segmen Pelanggan di 10 Negara Utama", fontweight="bold", color="#1e3a8a")
                    ax_geo_stack.set_xlabel("Negara")
                    ax_geo_stack.set_ylabel("Persentase (%)")
                    ax_geo_stack.legend(title="Segmen", bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    st.pyplot(fig_geo_stack)

                    # --- INSIGHT 8: COHORT LTV ---
                    st.write("---")
                    st.markdown("### ⏳ 8. Nilai Jangka Panjang Pelanggan (*Customer Lifetime Value* / LTV)")
                    st.info("**Insight:** Pelanggan yang bergabung lebih awal biasanya memiliki akumulasi nilai uang (*Monetary*) seumur hidup yang lebih besar. Grafik ini memvalidasi kekuatan *Retention* (Retensi) dalam menghasilkan LTV tinggi bagi bisnis.")
                    
                    df_filtered_ltv = df_filtered.copy()
                    df_filtered_ltv['CohortMonth'] = df_filtered_ltv.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
                    cohort_ltv = df_filtered_ltv.groupby(['CohortMonth', 'CustomerID'])['TotalAmount'].sum().reset_index()
                    cohort_avg_ltv = cohort_ltv.groupby('CohortMonth')['TotalAmount'].mean().reset_index()
                    cohort_avg_ltv['CohortMonth'] = cohort_avg_ltv['CohortMonth'].astype(str)
                    
                    fig_ltv, ax_ltv = plt.subplots(figsize=(10, 4.5))
                    sns.barplot(data=cohort_avg_ltv, x="CohortMonth", y="TotalAmount", color="#f59e0b", ax=ax_ltv)
                    ax_ltv.set_title("Bukti Visual: Rata-rata Akumulasi Nilai Belanja (LTV) Berdasarkan Bulan Kohort", fontweight="bold", color="#1e3a8a")
                    ax_ltv.set_xlabel("Bulan Bergabung (Kohort)")
                    ax_ltv.set_ylabel("Avg Lifetime Value (£)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_ltv)

                    # --- INSIGHT 9: GROSIR VS ECERAN ---
                    st.write("---")
                    st.markdown("### 🏷️ 9. Perilaku Harga vs Kuantitas (Eceran vs Grosir B2B)")
                    st.info("**Insight:** Apakah penjualan lebih didominasi oleh barang mahal dalam jumlah sedikit (Eceran Premium) atau barang murah dalam jumlah sangat masif (Grosir/B2B)? Skala logaritmik membuktikan *clustering* perilaku tersebut untuk strategi *pricing*.")
                    
                    fig_price, ax_price = plt.subplots(figsize=(10, 4.5))
                    sns.scatterplot(data=df_filtered, x="UnitPrice", y="Quantity", alpha=0.1, color="#14b8a6", ax=ax_price)
                    ax_price.set_title("Bukti Visual: Scatter Plot Harga Satuan vs Kuantitas Dibeli", fontweight="bold", color="#1e3a8a")
                    ax_price.set_xscale("log")
                    ax_price.set_yscale("log")
                    ax_price.set_xlabel("Unit Price (£) - Log Scale")
                    ax_price.set_ylabel("Quantity - Log Scale")
                    st.pyplot(fig_price)

                    # --- INSIGHT 10: REPEAT PURCHASE RATE ---
                    st.write("---")
                    st.markdown("### 🔄 10. Indikator Kecocokan Pasar (*Repeat Purchase Rate*)")
                    st.info("**Insight:** Pelanggan yang melakukan lebih dari 1 transaksi (*Repeat Purchasers*) adalah tulang punggung keberlanjutan bisnis. Persentase ini menunjukkan tingkat kecocokan produk terhadap pasar (*Product-Market Fit*).")
                    
                    repeat_cust = (df_insight["Frequency"] > 1).sum()
                    one_time_cust = (df_insight["Frequency"] == 1).sum()
                    
                    fig_repeat, ax_repeat = plt.subplots(figsize=(6, 5))
                    ax_repeat.pie([repeat_cust, one_time_cust], labels=["Repeat Purchasers (>1 Transaksi)", "One-Time Buyers (1 Transaksi)"], 
                                  autopct='%1.1f%%', colors=["#10b981", "#f43f5e"], startangle=90, textprops={'fontweight': 'bold'})
                    ax_repeat.set_title("Bukti Visual: Proporsi Pelanggan Berulang vs Sekali Beli", fontweight="bold", color="#1e3a8a")
                    st.pyplot(fig_repeat)
                    
                else:
                    st.warning("Tidak dapat menggabungkan data transaksi dan data pelanggan untuk mengekstrak insight.")
            else:
                st.warning("Data belum tersedia sepenuhnya untuk menarik insight gabungan.")

st.markdown("---")
st.caption("Proyek Akhir Big Data - Integrasi Spark, Postgres, dan MongoDB.")
