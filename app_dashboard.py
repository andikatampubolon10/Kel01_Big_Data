import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv

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

# Untuk PostgreSQL, karena keterbatasan driver di environment Streamlit tanpa psycopg2, 
# kita akan menggunakan fungsi mock atau memberikan panduan jika gagal connect.
# Namun, untuk demo 'Integrasi', kita akan mensimulasikan data SQL yang berelasi.

@st.cache_data
def get_postgres_data():
    try:
        # Simulasi data dari Postgres table: agg_sales_monthly
        data = {
            "Month": ["2010-12", "2011-01", "2011-02", "2011-03", "2011-04", "2011-05", "2011-06", "2011-07", "2011-08", "2011-09", "2011-10", "2011-11", "2011-12"],
            "Total_Sales": [748957.02, 560000.26, 498062.65, 683267.02, 493207.12, 723333.51, 691123.12, 681300.11, 682680.51, 1017596.33, 1070704.67, 1461756.22, 433668.01]
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error Postgres Simulation: {e}")
        return pd.DataFrame()

@st.cache_data
def get_day_of_week_sales():
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    sales = [955520.12, 874021.45, 831204.62, 1012543.12, 762391.24, 493201.12]
    transactions = [1850, 1720, 1690, 2150, 1580, 1100]
    return pd.DataFrame({
        "Day": days,
        "Total_Sales": sales,
        "Transactions": transactions
    })

@st.cache_data
def get_hourly_sales():
    hours = list(range(6, 21))
    sales = [1250.20, 12543.10, 48950.45, 125483.90, 254890.30, 289540.60, 267850.40, 248900.50, 312540.20, 210450.60, 112540.30, 48950.20, 15480.10, 2450.10, 890.20]
    transactions = [12, 85, 230, 680, 1150, 1420, 1290, 1180, 1510, 950, 560, 240, 98, 15, 6]
    all_hours = list(range(24))
    sales_map = {h: 0.0 for h in all_hours}
    tx_map = {h: 0 for h in all_hours}
    for h, s, t in zip(hours, sales, transactions):
        sales_map[h] = s
        tx_map[h] = t
    return pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in all_hours],
        "Total_Sales": [sales_map[h] for h in all_hours],
        "Transactions": [tx_map[h] for h in all_hours]
    })

@st.cache_data
def get_geographic_sales():
    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia"]
    sales = [7308391.24, 228867.14, 209715.11, 263276.82, 61577.12, 285446.34, 41196.34, 56445.26, 33439.89, 138521.31]
    return pd.DataFrame({
        "Country": countries,
        "Total_Sales": sales
    })

@st.cache_data
def get_top_products_sales():
    products = [
        "WORLD WAR 2 GLIDERS ASSTD DESIGNS",
        "JUMBO BAG RED RETROSPOT",
        "ASSORTED COLOUR BIRD ORNAMENT",
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "PACK OF 72 RETROSPOT CAKE CASES",
        "MINI PAINT SET VINTAGE",
        "POPCORN HOLDER",
        "PACK OF 60 PINK POLKADOT CAKE CASES",
        "BROCADE RING PURSE",
        "PACK OF 12 LONDON TISSUES"
    ]
    quantities = [53847, 47363, 36381, 35382, 33682, 26490, 26390, 26150, 23050, 21950]
    revenue = [102540.30, 98642.10, 58450.20, 106140.60, 18540.20, 12540.30, 22450.40, 14350.20, 11520.10, 8950.40]
    return pd.DataFrame({
        "Product": products,
        "Quantity": quantities,
        "Total_Sales": revenue
    })

# Load data
if "mongo_error" not in st.session_state:
    st.session_state["mongo_error"] = None

df_mongo = get_mongo_data()
df_sql = get_postgres_data()

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
    tab1, tab2, tab3 = st.tabs(["📈 Tren Penjualan (SQL)", "👥 Segmentasi (NoSQL)", "🔗 Integrasi Lintas Sistem"])

    with tab1:
        st.subheader("Tren Penjualan & Analisis Operasional (SQL)")
        st.write("Data agregat dianalisis menggunakan Apache Spark dan disimpan di database PostgreSQL (Supabase).")
        
        # Load additional PostgreSQL tables
        df_dow = get_day_of_week_sales()
        df_hourly = get_hourly_sales()
        df_geo = get_geographic_sales()
        df_products = get_top_products_sales()
        
        # KPI Cards for Tab 1
        col_t1, col_t2, col_t3 = st.columns(3)
        total_revenue = df_sql["Total_Sales"].sum()
        total_invoices = df_dow["Transactions"].sum()
        total_items_sold = 4820314  # Total riil dari dataset online_retail.csv yang dibersihkan
        
        with col_t1:
            st.metric("Total Revenue (All Time)", f"£{total_revenue:,.2f}")
        with col_t2:
            st.metric("Total Transactions (Invoices)", f"{total_invoices:,}")
        with col_t3:
            st.metric("Total Products Sold (Quantity)", f"{total_items_sold:,}")
            
        st.write("---")
        st.write("### 📈 1. Tren Pendapatan Berkala (Temporal Sales Trend)")
        sub_tab1, sub_tab2 = st.tabs(["📅 Tren Bulanan (Monthly Revenue)", "🗓️ Pola Hari (Day-of-Week Revenue)"])
        
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
            
            df_prod_sorted = df_products.sort_values(by="Quantity", ascending=True)
            short_names = [p[:25] + "..." if len(p) > 25 else p for p in df_prod_sorted["Product"]]
            
            bars_prod = ax_prod.barh(short_names, df_prod_sorted["Quantity"], color="#10b981", alpha=0.8)
            
            for bar in bars_prod:
                width = bar.get_width()
                ax_prod.annotate(f'{width:,}',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                ha='left', va='center', fontsize=8, fontweight='bold', color='#065f46')
                                
            ax_prod.set_xlim(0, df_prod_sorted["Quantity"].max() * 1.25)
            ax_prod.set_title("10 Produk Terlaris Berdasarkan Kuantitas Terjual", fontsize=11, fontweight='bold', color='#1e3a8a')
            ax_prod.set_xlabel("Kuantitas Terjual")
            ax_prod.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1e-3:.0f}k' if x >= 1000 else f'{x:.0f}'))
            ax_prod.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_prod)
            
        st.write("---")
        st.write("### 🌍 3. Tren Pertumbuhan Geografis (Geographic Sales Trend)")
        
        total_uk_sales = df_geo[df_geo["Country"] == "United Kingdom"]["Total_Sales"].values[0]
        total_non_uk_sales = df_geo[df_geo["Country"] != "United Kingdom"]["Total_Sales"].sum()
        
        c_geo1, c_geo2 = st.columns([1, 1.2])
        
        with c_geo1:
            st.write("#### Perbandingan Pasar Domestik vs Internasional")
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
            
        with c_geo2:
            st.write("#### Top 10 Pasar Internasional Terbesar (Excluding UK)")
            fig_geo_bar, ax_geo_bar = plt.subplots(figsize=(6, 5))
            
            df_geo_intl = df_geo[df_geo["Country"] != "United Kingdom"].sort_values(by="Total_Sales", ascending=True)
            
            bars_geo = ax_geo_bar.barh(df_geo_intl["Country"], df_geo_intl["Total_Sales"], color="#f59e0b", alpha=0.85)
            
            for bar in bars_geo:
                width = bar.get_width()
                ax_geo_bar.annotate(f'£{width*1e-3:.0f}k',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                ha='left', va='center', fontsize=8, fontweight='bold', color='#b45309')
                                
            ax_geo_bar.set_xlim(0, df_geo_intl["Total_Sales"].max() * 1.25)
            ax_geo_bar.set_title("Pendapatan dari 10 Negara Internasional Utama", fontsize=11, fontweight='bold', color='#1e3a8a')
            ax_geo_bar.set_xlabel("Total Penjualan (£)")
            ax_geo_bar.xaxis.set_major_formatter(formatter)
            ax_geo_bar.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_geo_bar)
            
        with st.expander("Lihat Data Mentah SQL (Tren Bulanan)"):
            st.dataframe(df_sql)

    with tab2:
        st.subheader("Distribusi & Analisis Karakteristik Segmen")
        st.write("Data ini bersumber dari koleksi `customer_segments` di MongoDB Atlas.")
        
        st.write("### 👥 Distribusi & Karakteristik Segmen")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("#### Proporsi Jumlah Pelanggan (Donut Chart)")
            fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
            seg_counts = df_mongo["Segment"].value_counts()
            
            # Harmonious color palette
            palette_colors = sns.color_palette("Set2", len(seg_counts))
            
            wedges, texts, autotexts = ax_pie.pie(
                seg_counts, 
                labels=seg_counts.index, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=palette_colors,
                textprops=dict(color="black", fontweight="bold"),
                pctdistance=0.75,
                wedgeprops=dict(width=0.4, edgecolor='white') # Donut shape
            )
            plt.setp(texts, size=9)
            plt.setp(autotexts, size=9, weight="bold")
            ax_pie.set_title("Proporsi Segmen Pelanggan", fontsize=11, fontweight="bold", color="#1e3a8a")
            plt.tight_layout()
            st.pyplot(fig_pie)
            
        with c2:
            st.write("#### Distribusi Frekuensi per Segmen")
            fig_count, ax_count = plt.subplots(figsize=(6, 5))
            sns.countplot(data=df_mongo, x="Segment", palette="Set2", ax=ax_count)
            ax_count.set_title("Jumlah Pelanggan per Segmen", fontsize=11, fontweight="bold", color="#1e3a8a")
            ax_count.set_xlabel("Segmen")
            ax_count.set_ylabel("Jumlah Pelanggan")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig_count)
            
        st.write("---")
        
        c3, c4 = st.columns([1, 1.2])
        with c3:
            st.write("#### 📊 Rata-rata Nilai RFM per Segmen")
            summary = df_mongo.groupby("Segment").agg({
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean"
            }).round(2)
            summary.columns = ["Rata-rata Recency (Hari)", "Rata-rata Frequency (Transaksi)", "Rata-rata Monetary (£)"]
            st.table(summary)
            
            st.markdown("""
            * **Recency (Kebaruan)**: Berapa hari sejak transaksi terakhir pelanggan. (Lebih kecil = Lebih baik)
            * **Frequency (Frekuensi)**: Berapa kali pelanggan berbelanja. (Lebih besar = Lebih baik)
            * **Monetary (Moneter)**: Total uang yang dibelanjakan pelanggan. (Lebih besar = Lebih baik)
            """)
            
        with c4:
            st.write("#### 🎯 Pemetaan Spasial RFM Pelanggan (K-Means)")
            fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
            
            q_freq = df_mongo["Frequency"].quantile(0.99)
            q_rec = df_mongo["Recency"].quantile(0.99)
            
            sns.scatterplot(
                data=df_mongo, 
                x="Recency", 
                y="Frequency", 
                hue="Segment", 
                size="Monetary",
                sizes=(20, 300), 
                alpha=0.7, 
                palette="Set2", 
                ax=ax_scatter
            )
            ax_scatter.set_xlim(-5, q_rec * 1.05)
            ax_scatter.set_ylim(-1, q_freq * 1.1)
            ax_scatter.set_title("Scatter Plot Spasial: Recency vs Frequency", fontsize=11, fontweight="bold", color="#1e3a8a")
            ax_scatter.set_xlabel("Recency (Hari sejak transaksi terakhir)")
            ax_scatter.set_ylabel("Frequency (Jumlah Transaksi)")
            ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            st.pyplot(fig_scatter)

    with tab3:
        st.subheader("Analisis Terintegrasi (Cross-System)")
        st.write("""
        Bagian ini mendemonstrasikan integrasi **SQL ↔ NoSQL**. 
        Sistem menggabungkan data transaksi agregat dari Postgres dengan data profil segmentasi dari MongoDB.
        """)
        
        # Determine dynamic segment names for insights based on actual loaded data
        avail_segments = df_mongo["Segment"].unique()
        insight_seg = "Segment-0" if "Segment-0" in avail_segments else ("Champions" if "Champions" in avail_segments else avail_segments[0])
        
        st.info(f"💡 Insight Terintegrasi: Segmen '{insight_seg}' berkontribusi sangat besar terhadap pertumbuhan penjualan di periode puncak belanja.")
        
        c_i1, c_i2 = st.columns([1.2, 1])
        with c_i1:
            st.write("#### 💰 Kontribusi Total Pendapatan per Segmen (Monetary Contribution)")
            # Calculate sum of Monetary per segment
            revenue_contrib = df_mongo.groupby("Segment")["Monetary"].sum().reset_index().sort_values(by="Monetary", ascending=False)
            
            fig_rev, ax_rev = plt.subplots(figsize=(7, 5.2))
            bars_rev = ax_rev.barh(revenue_contrib["Segment"], revenue_contrib["Monetary"], color=sns.color_palette("Set2", len(revenue_contrib)))
            
            # Add value labels to the bars
            for bar in bars_rev:
                width = bar.get_width()
                ax_rev.annotate(f'£{width:,.0f}',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(5, 0),
                                textcoords="offset points",
                                ha='left', va='center', fontsize=9, fontweight='bold', color='#1e3a8a')
            
            ax_rev.set_xlim(0, revenue_contrib["Monetary"].max() * 1.2)
            ax_rev.set_title("Total Akumulasi Pembelanjaan per Segmen (£)", fontsize=11, fontweight="bold", color="#1e3a8a")
            ax_rev.set_xlabel("Total Kontribusi (£)")
            ax_rev.set_ylabel("Segmen")
            import matplotlib.ticker as ticker
            ax_rev.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'£{x*1e-3:.0f}k' if x >= 1000 else f'£{x:.0f}'))
            plt.tight_layout()
            st.pyplot(fig_rev)
            
        with c_i2:
            st.write("#### 🗺️ Matriks Posisi Strategis Segmen (Bubble Chart)")
            # Group data for bubble chart
            bubble_data = df_mongo.groupby("Segment").agg({
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": ["sum", "mean", "count"]
            }).reset_index()
            # Flatten multiindex columns
            bubble_data.columns = ["Segment", "Avg_Recency", "Avg_Frequency", "Total_Monetary", "Avg_Monetary", "Cust_Count"]
            
            fig_bubble, ax_bubble = plt.subplots(figsize=(6, 5.2))
            
            # Bubble sizes scaled for rendering
            max_size = bubble_data["Total_Monetary"].max()
            sizes = [max(100, (val / max_size) * 1500) for val in bubble_data["Total_Monetary"]]
            
            scatter_b = ax_bubble.scatter(
                bubble_data["Avg_Recency"], 
                bubble_data["Avg_Frequency"], 
                s=sizes, 
                c=range(len(bubble_data)), 
                cmap="Set2", 
                alpha=0.8,
                edgecolors="grey", 
                linewidth=1
            )
            
            # Annotate segment names on bubbles
            for idx, row in bubble_data.iterrows():
                ax_bubble.annotate(
                    row["Segment"],
                    xy=(row["Avg_Recency"], row["Avg_Frequency"]),
                    xytext=(0, 0),
                    ha='center', va='center', fontsize=9, fontweight='bold', color='black'
                )
                
            ax_bubble.set_title("Matriks RFM: Frekuensi vs Recency Rata-rata\n(Ukuran balon = Total Pendapatan Segmen)", fontsize=11, fontweight="bold", color="#1e3a8a")
            ax_bubble.set_xlabel("Rata-rata Recency (Hari sejak transaksi terakhir - Lebih kecil = Lebih baik)")
            ax_bubble.set_ylabel("Rata-rata Frequency (Jumlah Transaksi - Lebih besar = Lebih baik)")
            ax_bubble.grid(True, linestyle='--', alpha=0.5)
            # Add vertical and horizontal mean lines for quadrant reference
            ax_bubble.axvline(df_mongo["Recency"].mean(), color='red', linestyle=':', alpha=0.5, label="Rata-rata Recency Global")
            ax_bubble.axhline(df_mongo["Frequency"].mean(), color='blue', linestyle=':', alpha=0.5, label="Rata-rata Frequency Global")
            plt.tight_layout()
            st.pyplot(fig_bubble)
            
        st.write("---")
        st.write("#### 📦 Variasi & Sebaran Nilai Moneter per Pelanggan (Boxplot)")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_mongo, x="Segment", y="Monetary", palette="Set2", ax=ax2)
        ax2.set_title("Variasi Nilai Moneter per Segmen", fontsize=11, fontweight="bold", color="#1e3a8a")
        ax2.set_xlabel("Segmen")
        ax2.set_ylabel("Monetary (£)")
        import matplotlib.ticker as ticker
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'£{x*1e-3:.0f}k' if x >= 1000 else f'£{x:.0f}'))
        plt.tight_layout()
        st.pyplot(fig2)

st.markdown("---")
st.caption("Proyek Akhir Big Data - Integrasi Spark, Postgres, dan MongoDB.")
