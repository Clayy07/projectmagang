import re
import textstat
import pandas as pd


def clean_html(raw_html):
    """
    Menghapus tag HTML dari teks.
    """
    if pd.isna(raw_html) or raw_html is None:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', str(raw_html))
    return cleantext

def calculate_keyword_density(text_content, keyword):
    if pd.isna(text_content) or not text_content or pd.isna(keyword) or not keyword:
        return 0.0

    cleaned_text = clean_html(text_content).lower()
    keyword_lower = str(keyword).lower().strip()

    if not keyword_lower:
        return 0.0

    words = re.findall(r'\b\w+\b', cleaned_text)
    total_kata = len(words)

    if total_kata == 0:
        return 0.0

    # Regex lebih fleksibel untuk frasa atau satu kata
    pattern = r'(?<!\w)' + re.escape(keyword_lower) + r'(?!\w)'
    jumlah_keyword = len(re.findall(pattern, cleaned_text))

    density = (jumlah_keyword / total_kata) * 100
    return round(density, 2)



def calculate_readability_score(text_content):
    """
    Menghitung skor Flesch Reading Ease pada teks yang sudah dibersihkan dari HTML.
    """
    if pd.isna(text_content) or not text_content:
        return 0.0 # Atau nilai standar lain untuk teks kosong/pendek
    
    cleaned_text = clean_html(text_content)
    if not cleaned_text.strip(): # Jika setelah cleaning hanya whitespace
        return 0.0
        
    # textstat mungkin error jika teks terlalu pendek
    try:
        score = textstat.flesch_reading_ease(cleaned_text)
    except: # Menangkap error umum dari textstat jika teks tidak valid
        score = 0.0 # Default score jika ada error
    return round(score, 2)

def calculate_heading_count(html_content):
    """
    Menghitung jumlah tag heading (H1-H6) dalam konten HTML mentah.
    Mencari tag seperti <h1>, <h2 class="foo">, dll.
    """
    if pd.isna(html_content) or not html_content:
        return 0
    # Regex ini mencari tag pembuka <h1...>, <h2...>, dst.
    headings = re.findall(r'<h[1-6][\s>]', str(html_content), re.IGNORECASE)
    return len(headings)

def extract_features_from_row(row, keyword_column='keyword_utama', title_column='judul_artikel', content_column='isi_artikel'):
    """
    Mengekstrak semua fitur SEO dari satu baris DataFrame.
    Menggunakan keyword_column jika ada, jika tidak, gunakan title_column sebagai keyword.
    """
    judul = str(row[title_column]).strip() if title_column in row and pd.notna(row[title_column]) else ""
    isi_artikel_html = str(row[content_column]).strip() if content_column in row and pd.notna(row[content_column]) else ""
    
    # Tentukan keyword
    if keyword_column in row and pd.notna(row[keyword_column]) and str(row[keyword_column]).strip():
        keyword = str(row[keyword_column]).strip()
    else:
        keyword = judul # Fallback ke judul jika keyword_utama tidak ada atau kosong

    # 1. Hitung Keyword Density (pada teks yang sudah dibersihkan HTML, menggunakan keyword yang ditentukan)
    density = calculate_keyword_density(isi_artikel_html, keyword)
    
    # 2. Hitung Readability Score (pada teks yang sudah dibersihkan HTML)
    readability = calculate_readability_score(isi_artikel_html)
    
    # 3. Hitung Jumlah Heading (pada konten HTML mentah)
    heading_count = calculate_heading_count(isi_artikel_html)
    
    return pd.Series([density, readability, heading_count])

def extract_features_for_df(df, keyword_column='keyword_utama', title_column='judul_artikel', content_column='isi_artikel'):
    """
    Menerapkan ekstraksi fitur ke seluruh DataFrame.
    Akan menimpa kolom 'density', 'readibility', 'jumlah_heading' jika sudah ada,
    untuk memastikan tidak ada duplikasi.
    """
    feature_names = ['density', 'readibility', 'jumlah_heading']
    
    # Membuat salinan DataFrame untuk dimodifikasi
    df_processed = df.copy()

    # Menghitung fitur baru menggunakan .apply()
    # Hasilnya akan berupa DataFrame dengan kolom bernama 0, 1, 2
    new_features_df = df_processed.apply(
        lambda row: extract_features_from_row(row, keyword_column, title_column, content_column), 
        axis=1
    )
    # Mengganti nama kolom hasil .apply() menjadi nama fitur yang benar
    new_features_df.columns = feature_names

    # Menimpa atau menambahkan kolom fitur yang baru dihitung ke df_processed
    for col_name in feature_names:
        df_processed[col_name] = new_features_df[col_name]
        
    return df_processed