import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Clustering Cacat Produk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. JUDUL APLIKASI
st.title("🏭 Analisis Clustering Cacat Produk Industri Manufaktur")
st.markdown("Aplikasi interaktif untuk segmentasi cacat produksi berbasis *Machine Learning* (K-Means Clustering).")

# 2. LOAD DATASET
@st.cache_data
def load_data():
    # Membaca file defects_data.csv yang di-upload di folder yang sama
    df = pd.read_csv("defects_data.csv")
    
    # Preprocessing: Mapping Ordinal Severity
    severity_mapping = {'Minor': 1, 'Moderate': 2, 'Critical': 3}
    df['severity_score'] = df['severity'].map(severity_mapping)
    return df

try:
    df = load_data()
    
    # 3. SIDEBAR CONTROLS
    st.sidebar.header("Pengaturan Model")
    optimal_k = st.sidebar.slider("Pilih Jumlah Klaster (K):", min_value=2, max_value=5, value=3)

    # 4. PROSES CLUSTERING (REAL-TIME)
    X = df[['repair_cost', 'severity_score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans_model.fit_predict(X_scaled)

    # 5. TAMPILAN ANTARMUKA UTAMA (TABS)
    tab1, tab2, tab3 = st.tabs(["📊 Visualisasi & Data", "💡 Interpretasi & Insight Bisnis", "🛠️ Prediksi Data Baru"])

    with tab1:
        st.header("Sebaran Klaster Cacat Manufaktur")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualisasi Scatter Plot
            fig, ax = plt.subplots(figsize=(10, 6.5))
            sns.scatterplot(
                x=df['repair_cost'], 
                y=df['severity_score'], 
                hue=df['cluster_kmeans'], 
                palette='Set1', 
                s=100, 
                alpha=0.85, 
                edgecolor='black', 
                linewidth=0.5,
                ax=ax
            )
            ax.set_title(f'Hasil Segmentasi Kasus Cacat Menggunakan K-Means (K={optimal_k})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Biaya Perbaikan Produk ($ / Repair Cost)', fontsize=12)
            ax.set_ylabel('Tingkat Keparahan Cacat (Severity Score)', fontsize=12)
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['1 (Minor)', '2 (Moderate)', '3 (Critical)'])
            ax.legend(title='Klaster K-Means', loc='upper left')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Statistik Rata-rata Tiap Klaster")
            profil_kmeans = df.groupby('cluster_kmeans')[['repair_cost', 'severity_score']].mean()
            st.dataframe(profil_kmeans.style.format({"repair_cost": "${:.2f}", "severity_score": "{:.2f}"}))
            
            st.subheader("Jumlah Sampel per Klaster")
            st.bar_chart(df['cluster_kmeans'].value_counts())

        st.subheader("Sampel Data Hasil Clustering")
        st.dataframe(df[['defect_id', 'product_id', 'defect_type', 'severity', 'repair_cost', 'cluster_kmeans']].head(20))

    with tab2:
        st.header("💡 Interpretasi Model & Insights Bisnis")
        st.markdown("""
        Berdasarkan hasil pembagian kelompok di atas, berikut adalah penjelasan karakteristik segmen cacat produksi:
        
        1. **🔴 Segmen Risiko Finansial Tinggi (High-Cost & Critical Defects):**
           * **Karakteristik:** Klaster yang memiliki tingkat keparahan tinggi (*Critical*) dan rata-rata pengeluaran biaya perbaikan paling besar.
           * **Insight Bisnis:** Lini produk atau area kerja ini memerlukan intervensi segera. Manajemen *Quality Assurance* (QA) wajib menempatkan sistem kendali mutu otomatis (*Automated Testing*) utama pada jalur ini untuk mencegah lolosnya cacat ke tangan konsumen yang memicu pembengkakan biaya garansi.
           
        2. **🟢 Segmen Moderat (Medium Cost / Moderate Defects):**
           * **Karakteristik:** Klaster dengan tingkat keparahan menengah dan biaya perbaikan yang cenderung stabil/sedang.
           * **Insight Bisnis:** Diperlukan pelatihan berkala (*retraining*) bagi operator di area terkait untuk menekan angka *human error* yang menyebabkan cacat fungsi operasional.
           
        3. **🔵 Segmen Minoritas Efisien (Low-Cost / Minor Defects):**
           * **Karakteristik:** Klaster dengan kerugian finansial kecil dan tingkat keparahan rendah (*Minor*).
           * **Insight Bisnis:** Penanganan dapat dilakukan melalui inspeksi berkala reguler (*Visual Inspection*) tanpa perlu intervensi atau alokasi biaya operasional yang besar karena tidak mengancam fungsionalitas utama produk.
        """)

    with tab3:
        st.header("🛠️ Input Data Cacat Baru untuk Prediksi Klaster")
        st.markdown("Masukkan data komponen cacat baru untuk mengetahui masuk ke dalam segmen/klaster mana.")
        
        input_cost = st.number_input("Masukkan Biaya Perbaikan ($):", min_value=0.0, value=50.0, step=5.0)
        input_severity = st.selectbox("Pilih Tingkat Keparahan:", ['Minor', 'Moderate', 'Critical'])
        
        severity_mapped = {'Minor': 1, 'Moderate': 2, 'Critical': 3}[input_severity]
        
        if st.button("Prediksi Klaster"):
            new_data = np.array([[input_cost, severity_mapped]])
            new_data_scaled = scaler.transform(new_data)
            pred_cluster = kmeans_model.predict(new_data_scaled)[0]
            
            st.success(f"Data cacat baru masuk ke dalam **Klaster {pred_cluster}**")
            
            if pred_cluster == 0: 
                st.warning("⚠️ **Rekomendasi:** Alokasikan tindakan perbaikan segera (Prioritas Utama) karena potensi pembengkakan biaya tinggi.")
            else:
                st.info("ℹ️ **Rekomendasi:** Lakukan penanganan standar sesuai standar operasional berkala.")

except FileNotFoundError:
    st.error("❌ File 'defects_data.csv' belum ditemukan di repositori GitHub Anda. Silakan upload file dataset terlebih dahulu agar aplikasi dapat bekerja.")
