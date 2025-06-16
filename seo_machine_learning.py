import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Impor utilitas SEO
import seo_utils

# --- Nama File ---
RAW_DATASET_FILE = 'dataset_seo_articles.xlsx'
PROCESSED_DATA_FILE = 'hasil_artikel_seo.xlsx'
MODEL_FILE = 'model_seo.joblib'
METRICS_FILE = 'evaluation_metrics.json'
TEST_DATA_X_FILE = 'X_test_seo.csv'
TEST_DATA_y_FILE = 'y_test_seo.csv'
NEW_ARTICLES_FOR_TESTING_FILE = 'testing_artikel.xlsx'

def create_features(input_file=RAW_DATASET_FILE, output_file=PROCESSED_DATA_FILE):
    try:
        df = pd.read_excel(input_file)
        print(f"Data awal dari {input_file} berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File {input_file} tidak ditemukan.")
        return None
    except Exception as e:
        print(f"Error saat memuat {input_file}: {e}")
        return None

    required_cols = ['judul_artikel', 'isi_artikel', 'label']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Kolom wajib ({required_cols}) tidak ditemukan.")
        return None

    print("Menghitung fitur SEO...")
    df_processed = seo_utils.extract_features_for_df(
        df,
        keyword_column='keyword_utama',
        title_column='judul_artikel',
        content_column='isi_artikel'
    )

    try:
        df_processed.to_excel(output_file, index=False)
        print(f"Data dengan fitur disimpan ke {output_file}")
        return df_processed
    except Exception as e:
        print(f"Error saat menyimpan {output_file}: {e}")
        return None

def train_and_evaluate_model(df_processed):
    if df_processed is None or df_processed.empty:
        print("DataFrame kosong, pelatihan dibatalkan.")
        return

    X = df_processed[['density', 'readibility', 'jumlah_heading']]
    y = df_processed['label']

    print("\nDistribusi Label Sebelum Balancing:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nMemulai GridSearchCV dengan pipeline SMOTE...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nParameter terbaik: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=['Actual Not SEO', 'Actual SEO'], columns=['Predicted Not SEO', 'Predicted SEO']))

    # Simpan model
    joblib.dump(best_model, MODEL_FILE)
    print(f"Model disimpan di {MODEL_FILE}")

    # Simpan metrik evaluasi
    metrics = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrik disimpan di {METRICS_FILE}")

    # Simpan data uji
    X_test.to_csv(TEST_DATA_X_FILE, index=False)
    y_test.to_csv(TEST_DATA_y_FILE, index=False, header=['label'])
    print(f"Data uji disimpan di {TEST_DATA_X_FILE} dan {TEST_DATA_y_FILE}")

    # Feature importance
    print("\nFeature Importance:")
    importances = best_model.named_steps['classifier'].feature_importances_
    for name, importance in zip(X.columns, importances):
        print(f"{name}: {importance:.4f}")

def predict_new_articles(model_file=MODEL_FILE, articles_file=NEW_ARTICLES_FOR_TESTING_FILE):
    try:
        model = joblib.load(model_file)
        print(f"Model {model_file} berhasil dimuat.")
    except FileNotFoundError:
        print(f"Model tidak ditemukan.")
        return
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return

    try:
        df_test = pd.read_excel(articles_file)
        print(f"Artikel baru dimuat dari {articles_file}.")
    except Exception as e:
        print(f"Error saat membaca artikel: {e}")
        return

    required_cols = ['judul_artikel', 'isi_artikel']
    if not all(col in df_test.columns for col in required_cols):
        print(f"Kolom wajib tidak ditemukan di {articles_file}.")
        return

    print("Menghitung fitur artikel baru...")
    df_test_processed = seo_utils.extract_features_for_df(
        df_test,
        keyword_column='keyword_utama',
        title_column='judul_artikel',
        content_column='isi_artikel'
    )

    X_new = df_test_processed[['density', 'readibility', 'jumlah_heading']]
    if X_new.empty:
        print("Tidak ada fitur yang valid untuk diprediksi.")
        return

    print("Melakukan prediksi...")
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    print("\n--- Hasil Prediksi Artikel Baru ---")
    for i, row in df_test_processed.iterrows():
        label = "SEO Friendly" if predictions[i] == 1 else "Not SEO Friendly"
        prob_seo = probabilities[i][1] * 100
        print(f"- Judul: {row['judul_artikel']}")
        print(f"  Prediksi: {label} (Probabilitas SEO Friendly: {prob_seo:.1f}%)")
        print(f"  Fitur: Density: {row['density']} | Readability: {row['readibility']} | Heading: {row['jumlah_heading']}\n")

def main():
    df_processed = create_features()
    if df_processed is not None and not df_processed.empty:
        train_and_evaluate_model(df_processed)
        predict_new_articles()
    else:
        print("Proses dihentikan karena gagal memproses data.")

if __name__ == "__main__":
    main()
