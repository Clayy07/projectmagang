import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json

# Import fungsi dari seo_utils.py
import seo_utils

# --- Konfigurasi Nama File ---
PROCESSED_DATA_FILE = 'hasil_artikel_seo.xlsx'
MODEL_FILE = 'model_seo.joblib'
METRICS_FILE = 'evaluation_metrics.json'
TEST_DATA_X_FILE = 'X_test_seo.csv'
TEST_DATA_y_FILE = 'y_test_seo.csv'

@st.cache_data
def load_processed_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading processed data ({file_path}): {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model ({model_path}): {e}")
        return None

@st.cache_data
def load_evaluation_metrics(metrics_path):
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        st.error(f"Gagal memuat metrik evaluasi ({metrics_path}): {e}")
        return None

@st.cache_data
def load_test_data(x_path, y_path):
    try:
        x_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path)
        return x_test, y_test
    except Exception as e:
        st.error(f"Gagal memuat data uji ({x_path}, {y_path}): {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    st.set_page_config(page_title="SEO ML Dashboard", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Dashboard Prediksi Artikel SEO</h1>", unsafe_allow_html=True)

    # --- Load assets ---
    model = load_model(MODEL_FILE)
    metrics = load_evaluation_metrics(METRICS_FILE)
    df_display_original = load_processed_data(PROCESSED_DATA_FILE)
    X_test_saved, y_test_saved = load_test_data(TEST_DATA_X_FILE, TEST_DATA_y_FILE)

    # --- Sidebar filter ---
    st.sidebar.header("📂 Filter Kategori Artikel")
    kategori_options = ["Semua"]
    kategori = "Semua"
    df_display_renamed = pd.DataFrame()

    if not df_display_original.empty:
        df_display_renamed = df_display_original.rename(
            columns={'judul_artikel': 'Judul Artikel', 'density': 'Density',
                     'readibility': 'Readability', 'jumlah_heading': 'Jumlah Heading', 'label': 'Label SEO'}
        )
        if 'Label SEO' in df_display_renamed.columns:
            if 1 in df_display_renamed['Label SEO'].unique():
                kategori_options.append("SEO Friendly")
            if 0 in df_display_renamed['Label SEO'].unique():
                kategori_options.append("Not SEO Friendly")

        kategori = st.sidebar.radio(
            "Pilih kategori artikel yang ingin ditampilkan:",
            kategori_options,
            key="kategori_filter_artikel_utama"
        )
    else:
        st.sidebar.warning("Data artikel tidak tersedia untuk difilter.")

    # --- Tampilkan dataset ---
    st.header("📑 Dataset Artikel")
    if not df_display_original.empty and not df_display_renamed.empty:
        df_to_show_in_table = df_display_renamed.copy()
        subheader_dynamic = "Semua Artikel"

        if kategori == "SEO Friendly":
            df_to_show_in_table = df_display_renamed[df_display_renamed['Label SEO'] == 1]
            subheader_dynamic = "Daftar Artikel SEO Friendly"
        elif kategori == "Not SEO Friendly":
            df_to_show_in_table = df_display_renamed[df_display_renamed['Label SEO'] == 0]
            subheader_dynamic = "Daftar Artikel Not SEO Friendly"

        st.subheader(subheader_dynamic)
        if not df_to_show_in_table.empty:
            st.dataframe(
                df_to_show_in_table[['Judul Artikel', 'Density', 'Readability', 'Jumlah Heading', 'Label SEO']]
                .reset_index(drop=True).rename_axis('No.').rename(index=lambda x: x+1),
                use_container_width=True
            )
            st.write(f"Total data asli: {len(df_display_original)} baris. Ditampilkan: {len(df_to_show_in_table)} baris.")
        else:
            st.warning("Tidak ada data artikel yang sesuai dengan kategori filter yang dipilih.")
            st.write(f"Total data asli: {len(df_display_original)} baris.")
    else:
        st.warning(f"Tidak dapat memuat data dari {PROCESSED_DATA_FILE} untuk ditampilkan.")

    # --- Visualisasi interaktif ---
    if not df_display_original.empty:
        st.subheader("📈 Visualisasi Distribusi Fitur")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_density = px.histogram(df_display_original, x='density', nbins=30, title="Keyword Density", color_discrete_sequence=['skyblue'])
            st.plotly_chart(fig_density, use_container_width=True)
        with col2:
            fig_read = px.histogram(df_display_original, x='readibility', nbins=30, title="Readability Score", color_discrete_sequence=['lightgreen'])
            st.plotly_chart(fig_read, use_container_width=True)
        with col3:
            fig_heading = px.histogram(df_display_original, x='jumlah_heading', nbins=30, title="Jumlah Heading", color_discrete_sequence=['salmon'])
            st.plotly_chart(fig_heading, use_container_width=True)

    # --- Evaluasi Model ---
    st.subheader("📉 Evaluasi Model")
    if metrics:
        st.metric("Akurasi Model", f"{metrics['accuracy']:.2f}")
        st.write("#### 📋 Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose().round(3)
        if 'support' in report_df.columns:
            report_df['support'] = report_df['support'].astype(int)
        report_df = report_df.reset_index().rename(columns={"index": "Metrik/Label"})
        st.dataframe(report_df, use_container_width=True)

        st.write("#### 📊 Confusion Matrix")
        cm_df = pd.DataFrame(metrics['confusion_matrix'],
                             index=['Aktual: Not SEO', 'Aktual: SEO'],
                             columns=['Prediksi: Not SEO', 'Prediksi: SEO'])
        st.table(cm_df)
    else:
        st.warning("Metrik evaluasi tidak dapat dimuat.")

    # --- Hasil Prediksi pada Data Uji yang Disimpan ---
    if model and not X_test_saved.empty and not y_test_saved.empty:
        st.subheader("📋 Hasil Prediksi pada Data Uji Asli yang Disimpan")
        y_pred_saved = model.predict(X_test_saved)
        result_df_saved = X_test_saved.copy()
        result_df_saved['Actual Label'] = y_test_saved['label'].values
        result_df_saved['Predicted Label'] = y_pred_saved

        st.dataframe(
            result_df_saved.reset_index(drop=True)
            .rename_axis('No.').rename(index=lambda x: x+1)
            .rename(columns={'density':'Density', 'readibility':'Readability', 'jumlah_heading':'Jumlah Heading'}),
            use_container_width=True
        )
    else:
        st.warning("Data uji atau model tidak dapat dimuat untuk menampilkan hasil prediksi pada data uji asli.")

    # --- Upload Artikel untuk Prediksi Baru ---
    st.subheader("📤 Upload Artikel untuk Prediksi SEO")
    uploaded_file = st.file_uploader(
        "Unggah file Excel yang berisi judul artikel, isi artikel, dan keyword utama",
        type=["xlsx"]
    )

    if uploaded_file is not None and model is not None:
        try:
            uploaded_df = pd.read_excel(uploaded_file)
            required_cols_upload = {'judul_artikel', 'isi_artikel'}
            if not required_cols_upload.issubset(uploaded_df.columns):
                st.error(f"Kolom wajib yang harus ada di file unggahan: {required_cols_upload}")
            else:
                with st.spinner('Menghitung fitur SEO untuk artikel yang diunggah...'):
                    df_uploaded_features = seo_utils.extract_features_for_df(
                        uploaded_df,
                        keyword_column='keyword_utama',
                        title_column='judul_artikel',
                        content_column='isi_artikel'
                    )

                X_new_uploaded = df_uploaded_features[['density', 'readibility', 'jumlah_heading']]
                if X_new_uploaded.empty:
                    st.warning("Tidak ada fitur yang bisa diekstrak dari file yang diunggah.")
                else:
                    with st.spinner('Memprediksi artikel yang diunggah...'):
                        predictions_uploaded = model.predict(X_new_uploaded)
                        df_uploaded_features['Prediksi Label'] = ['SEO Friendly' if p == 1 else 'Not SEO Friendly' for p in predictions_uploaded]

                    st.success("✅ Prediksi untuk file yang diunggah selesai!")

                    df_display_uploaded_predictions = df_uploaded_features[
                        ['judul_artikel', 'density', 'readibility', 'jumlah_heading', 'Prediksi Label']
                    ]
                    st.dataframe(
                        df_display_uploaded_predictions
                        .reset_index(drop=True).rename_axis('No.').rename(index=lambda x: x+1)
                        .rename(columns={
                            'judul_artikel': 'Judul Artikel',
                            'density': 'Density',
                            'readibility': 'Readability',
                            'jumlah_heading': 'Jumlah Heading'
                        }),
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Gagal memproses file yang diunggah: {str(e)}")
    elif uploaded_file is not None and model is None:
        st.error("Model tidak dapat dimuat, prediksi tidak bisa dilakukan.")

if __name__ == "__main__":
    main()
