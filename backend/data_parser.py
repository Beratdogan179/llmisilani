import json
import re
import os
import glob
import sys
# Try-except blokları ile kütüphanelerin yüklü olmama durumunu yönetiyoruz.
# Bu, Mac'teki uyumsuzluklara rağmen kodun çökmesini engeller.
try:
    from docx import Document
    import pdfplumber
    from PIL import Image
    import pytesseract
except ImportError:
    # Gerekli kütüphaneler yoksa bile temel parsing devam eder
    print("UYARI: PDF/DOCX/OCR kütüphaneleri bulunamadı. Sadece TXT dosyaları işlenecektir.")
    pass

# ====================================================================
# A. CV PARSING FONKSİYONU (Veriyi Ayıklama)
# ====================================================================

# Yardımcı fonksiyon: Başlıklar arasındaki metni temizce yakalar
def extract_field_fixed(start_patterns, text):
    all_headers = [
        r'Ad Soyad', r'Name', r'Pozisyon', r'Konum', r'Location', r'Address', 
        r'Özgeçmiş Özeti', r'Professional Summary', r'Profile', r'Summary',
        r'İş Deneyimi', r'Experience', r'Work Experience', r'Eğitim', r'Education', 
        r'Eğitim Bilgileri', r'Beceriler', r'Skills', r'Yetenekler', r'Teknik Beceriler', 
        r'Diller', r'Languages', r'Referanslar', r'Kişisel Bilgiler'
    ]
    all_headers_pattern = r'(?=' + '|'.join(all_headers) + r')'

    start_regex = r'(' + '|'.join(start_patterns) + r')[:\s\n\r]*'
    start_match = re.search(start_regex, text, re.DOTALL | re.IGNORECASE)
    
    if start_match:
        start_index = start_match.end()
        end_match = re.search(all_headers_pattern, text[start_index:], re.DOTALL | re.IGNORECASE)
        
        if end_match:
            end_index = start_index + end_match.start()
        else:
            end_index = len(text)
            
        return text[start_index:end_index].strip()
    return "Bilinmiyor"

def parse_cv(cv_text):
    cv_data = {}
    cv_text_clean = cv_text.strip()
    
    # --- 1. Temel Bilgiler ---
    name_match_labeled = re.search(r'(Ad Soyad|Name):\s*(.*?)\s*(\n|\r\n)', cv_text_clean, re.IGNORECASE)
    if name_match_labeled:
        cv_data["name"] = name_match_labeled.group(2).strip()
    else:
        name_only_match = re.search(r'^\s*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?: [A-ZÇĞİÖŞÜ][a-zçğıöşü]+){1,3})\s*(\n|\r\n)', cv_text_clean[:50], re.DOTALL)
        cv_data["name"] = name_only_match.group(1).strip() if name_only_match else "Bilinmiyor"
            
    pos_match = re.search(r'(Pozisyon|Job Title|Title):\s*(.*?)\s*(\n|\r\n)', cv_text_clean, re.IGNORECASE)
    cv_data["position"] = pos_match.group(2).strip() if pos_match else "Bilinmiyor"
    
    loc_match = re.search(r'(Konum|Location|Address):\s*(.*?)\s*(\n|\r\n)', cv_text_clean, re.IGNORECASE)
    cv_data["location"] = loc_match.group(2).strip() if loc_match else "Bilinmiyor"
    
    # --- 2. Multiline Alanları Yakala ---
    cv_data["summary"] = extract_field_fixed([r'Özgeçmiş Özeti', r'Professional Summary', r'Özgeçmiş', r'Profile', r'Summary'], cv_text_clean)
    cv_data["experience_raw"] = extract_field_fixed([r'İş Deneyimleri', r'Experience', r'Work Experience', r'İş deneyimi'], cv_text_clean)
    cv_data["education_raw"] = extract_field_fixed([r'Eğitim', r'Education', r'Eğitim Bilgileri'], cv_text_clean)
    cv_data["skills_raw"] = extract_field_fixed([r'Beceriler', r'Skills', r'Yetenekler', r'Teknik Beceriler'], cv_text_clean)
    cv_data["languages_raw"] = extract_field_fixed([r'Diller', r'Languages'], cv_text_clean)
    
    # --- Post-Processing (Gürültü Temizleme) ---
    if cv_data["skills_raw"] and cv_data["skills_raw"] != "Bilinmiyor":
        skills_lines = cv_data["skills_raw"].split('\n')
        cleaned_lines = []
        found_actual_skill_list = False
        
        for line in skills_lines:
            stripped_line = line.strip()
            if re.search(r'\d/\d', stripped_line) or stripped_line.startswith('-'):
                 found_actual_skill_list = True
            
            if found_actual_skill_list:
                 cleaned_lines.append(line)
        
        if cleaned_lines:
            cv_data["skills_raw"] = '\n'.join(cleaned_lines).strip()
            
    return cv_data

# ====================================================================
# B. İLAN PARSING FONKSİYONU
# ====================================================================

def parse_job(job_text):
    job_pattern = re.compile(
        r'Company:\s*(.*?)\s*'
        r'Sector:\s*(.*?)\s*'
        r'Job Title:\s*(.*?)\s*'
        r'Location:\s*(.*?)\s*'
        r'Work Type:\s*(.*?)\s*'
        r'Department:\s*(.*?)\s*'
        r'Experience:\s*(.*?)\s*'
        r'Education:\s*(.*?)\s*'
        r'Description:(.*?)\s*'
        r'Qualifications:(.*?)\s*'
        , re.DOTALL | re.IGNORECASE
    )
    match = job_pattern.search(job_text)
    
    if match:
        job_data = {
            "company": match.group(1).strip(),
            "sector": match.group(2).strip(),
            "job_title": match.group(3).strip(),
            "location": match.group(4).strip(),
            "work_type": match.group(5).strip(),
            "department": match.group(6).strip(),
            "experience": match.group(7).strip(),
            "education_level": match.group(8).strip(),
            "description": match.group(9).strip(),
            "qualifications_raw": match.group(10).strip()
        }
        return job_data
    return None

# ====================================================================
# C. HAM METİN PARSING YÖNETİCİSİ (Tek Toplu Metin için)
# ====================================================================

def process_raw_data(raw_data_string, is_cv=True):
    if is_cv:
        # Regex ile tüm CV'leri ayır
        entries = re.split(r'={20,}\s*CV\s*\d+\s*—\s*.*?\s*={20,}', raw_data_string)
        parser = parse_cv
    else:
        # Regex ile tüm ilanları ayır
        entries = re.split(r'={3}JOB\d+={3}', raw_data_string)
        parser = parse_job
    
    parsed_list = []
    
    start_index = 1 
    if is_cv and len(entries) < 5: 
         start_index = 0

    for i, entry in enumerate(entries[start_index:]):
        entry_content = entry.strip()
        if entry_content:
            parsed_data = parser(entry_content)
            if parsed_data:
                parsed_data['id'] = f"{'CV' if is_cv else 'JOB'}_{i + 1}"
                parsed_list.append(parsed_data)
                
    return parsed_list

# ====================================================================
# D. TEMİZLEME VE POST-PROCESSING FONKSİYONU
# ====================================================================

def clean_parsed_cvs(cv_list):
    # Bu fonksiyon, daha önce tanımladığımız temizleme mantığını içerir.
    
    cleaned_list = []
    for cv_data in cv_list:
        
        # --- Genel Temizlik ---
        for key in ['summary', 'experience_raw', 'education_raw', 'skills_raw', 'languages_raw']:
            text = cv_data.get(key, "Bilinmiyor")
            if text != "Bilinmiyor":
                text = re.sub(r'(Ad Soyad|Name|Pozisyon|Title|Konum|Location|Address|Özgeçmiş Özeti|Professional Summary|İş Deneyimleri|Experience|Eğitim|Education|Beceriler|Skills|Diller)[\s\S]*?:', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s+', ' ', text).strip()
                cv_data[key] = text
        
        # --- Kritik Çakışma Düzeltmeleri ---
        if cv_data['summary'] != "Bilinmiyor" and cv_data['experience_raw'] != "Bilinmiyor":
            exp_text_start = cv_data['experience_raw'].split('—')[0].split('-')[0].strip()
            if cv_data['summary'].endswith(exp_text_start) and len(exp_text_start) > 10:
                 cv_data['summary'] = cv_data['summary'].split('.')[-2].strip() + '.' if '.' in cv_data['summary'] else cv_data['summary'][:150]
        
        if cv_data['skills_raw'] != "Bilinmiyor":
            skills_lines = cv_data['skills_raw'].split('\n')
            cleaned_lines = []
            found_actual_skill_list = False
            
            for line in skills_lines:
                stripped_line = line.strip()
                if re.search(r'\d/\d', stripped_line) or stripped_line.startswith('-'):
                     found_actual_skill_list = True
                
                if found_actual_skill_list:
                     cleaned_lines.append(line)
            
            if cleaned_lines:
                cv_data['skills_raw'] = '\n'.join(cleaned_lines).strip()
            
        cleaned_list.append(cv_data)

    return cleaned_list

# ====================================================================
# E. KOD ÇALIŞTIRMA BLOĞU (Execution - SON)
# ====================================================================

if __name__ == '__main__':
    
    # 💡 Çözüm: Ham veriyi ana klasöre yerleştirdik. Şimdi onu okuyacağız.
    RAW_CV_FILE = 'cv_hepsi.text' 
    RAW_JOB_FILE = 'ilanlar_hepsi.text' 

    try:
        with open(RAW_CV_FILE, 'r', encoding='utf-8') as f:
            raw_cv_data = f.read()
        with open(RAW_JOB_FILE, 'r', encoding='utf-8') as f:
            raw_job_data = f.read()
    except FileNotFoundError:
        print(f"KRİTİK HATA: '{RAW_CV_FILE}' veya '{RAW_JOB_FILE}' bulunamadı. Lütfen toplu ham veriyi ana klasöre (data_parser.py'nin yanına) yerleştirin.")
        sys.exit(1) # Hata koduyla çık

    # Parsing ve Temizleme
    cv_json = process_raw_data(raw_cv_data, is_cv=True)
    job_json = process_raw_data(raw_job_data, is_cv=False)
    final_cv_json = clean_parsed_cvs(cv_json)
    
    
    print(f"Toplam {len(final_cv_json)} CV ve {len(job_json)} İlan başarıyla ayrıştırıldı.")
    
    # Örnek Çıktı
    if final_cv_json:
        print("\n" + "="*20 + " FİNAL TEMİZLENMİŞ CV ÇIKTISI (İlk Örnek) " + "="*20)
        print(json.dumps(final_cv_json[0], indent=4, ensure_ascii=False))

    # JSON dosyalarına KAYIT
    with open('parsed_cvs_FINAL.json', 'w', encoding='utf-8') as f:
        json.dump(final_cv_json, f, indent=4, ensure_ascii=False)
        
    with open('parsed_jobs_FINAL.json', 'w', encoding='utf-8') as f:
        json.dump(job_json, f, indent=4, ensure_ascii=False)
        
    print("\n✅ Hafta 4 - Parsing/Temizleme tamamlandı. LLM Reranking'e geçebiliriz.")