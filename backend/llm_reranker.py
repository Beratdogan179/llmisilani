import os
import json
import re
import sys
# from sentence_transformers import SentenceTransformer # Kütüphane kilitlenmesini önlemek için kapalı tutuyoruz
# import faiss
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====================================================================
# A. YARDIMCI FONKSİYONLAR (VERİ YÜKLEME VE KURAL TABANLI SKORLAMA)
# ====================================================================

def load_data(file_name):
    """Final JSON dosyasını okur."""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"KRİTİK HATA: {file_name} dosyası bulunamadı. Program sonlandırılıyor.")
        sys.exit(1)

# --- HAFTA 8: KURAL TABANLI SKORLAMA FONKSİYONLARI ---

def map_education_level(level_str):
    """Eğitim seviyesini sayısal değere çevirir ve Lisans/Üniversite eşleştirmesini güçlendirir."""
    level_str = level_str.lower()
    
    # LLM/Scoring için akademik seviye ataması
    if "doktora" in level_str: return 5
    if "yüksek lisans" in level_str: return 4
    
    # KRİTİK DÜZELTME: Hem "Lisans" hem de "Üniversite" kelimelerini kontrol et.
    # Bu, "Üniversite(Mezun)" gibi ifadeleri 3. seviyeye eşleştirmeyi garanti eder.
    if "üniversite" in level_str or "lisans" in level_str: return 3
    
    if "ön lisans" in level_str: return 2
    if "lise" in level_str: return 1
    return 0

def calculate_education_score(cv_edu, job_edu):
    """Eğitim gereksinimi uyumunu hesaplar (0-1)."""
    cv_level = map_education_level(cv_edu)
    job_level = map_education_level(job_edu)
    
    if cv_level >= job_level:
        return 1.0 
    if cv_level == job_level - 1 and job_level > 0:
        return 0.5 
    return 0.0 

def calculate_location_score(cv_loc, job_loc):
    """Lokasyon uyumunu hesaplar (0-1)."""
    cv_city = cv_loc.lower().split(',')[0].strip().split('(')[0].strip()
    job_city = job_loc.lower().split(',')[0].strip().split('(')[0].strip()
    
    if cv_city == job_city and cv_city in ["istanbul", "ankara", "izmir"]:
        return 1.0 
    if "istanbul" in cv_city and ("avrupa" in job_loc.lower() or "asya" in job_loc.lower()):
        return 1.0 
    return 0.0 

def calculate_salary_score(job_title):
    """Maaş/Kıdem Uyumunu basitleştirilmiş şekilde hesaplar (0-1)."""
    if "danışman" in job_title.lower() or "uzman" in job_title.lower() or "şef" in job_title.lower() or "sorumlu" in job_title.lower():
        return 1.0
    return 0.0

def is_hard_academic_barrier_satisfied(cv_edu, job_title):
    """Zorunlu Tıp/Mühendislik/Diş Hekimi eğitimi gereksinimini kontrol eder."""
    
    hard_barrier = ['hekim', 'radyoloji', 'cildiye', 'uzmanı', 'mühendis', 'diş hekimi']
    if not any(field in job_title.lower() for field in hard_barrier):
        return True # Özel bariyer yoksa, sorun yok.

    if 'hemşirelik' in cv_edu.lower() and job_title.lower() in ['hemşire', 'radyoloji teknisyeni - teknikeri']:
        return True # Hemşirelik, Teknisyen/Hemşire rollerine kısmen uyar.
    
    if 'diş hekimliği' in cv_edu.lower() and 'diş hekimi' in job_title.lower():
        return True
    
    # Kural: Hemşire veya Diş Hekimi isen, Hekim/Uzman (Doktor) olamazsın.
    if ('hemşire' in cv_edu.lower() or 'diş hekimi' in cv_edu.lower()) and job_title.lower() in ['ambulans hekimi', 'cildiye uzmanı']:
        return False 
        
    return False

# --- FINAL AĞIRLIKLANDIRMA FONKSİYONU ---
def calculate_weighted_score(result, query_cv, job_details):
    
    # Kural tabanlı skorları hesapla (S_edu, S_loc, S_sal, S_LLM_Normalized)
    
    S_LLM_Normalized = result['llm_score'] / 10.0
    
    # LLM skorunu Deneyim ve Yetenek Skorlarına EŞİT DAĞITIYORUZ (0.35 / 0.35)
    S_Exp = S_LLM_Normalized
    S_Skills = S_LLM_Normalized
    
    S_edu = calculate_education_score(query_cv.get('education_raw', ''), job_details.get('education_level', ''))
    S_loc = calculate_location_score(query_cv.get('location', ''), job_details.get('location', ''))
    S_sal = calculate_salary_score(job_details['job_title'])

    # KRİTİK CEZA: LLM'in atladığı Tıp/Uzmanlık Engelini Python'da Uygula
    if not is_hard_academic_barrier_satisfied(query_cv.get('education_raw', ''), result['job_title']):
        # Eğer akademik engel (Tıp/Uzman Hekimlik) aşılamazsa, Semantic Skorları sıfırla
        S_Exp = 0.0
        S_Skills = 0.0
        
    # Nihai Ağırlıklı Skor (Hocanın Formülü: 0.35*Exp + 0.35*Skills + ...)
    final_score = (0.35 * S_Exp) + (0.35 * S_Skills) + (0.10 * S_edu) + (0.10 * S_loc) + (0.10 * S_sal)

    # Yeni dönüş değerleri: Final Skor, Eğitim, Konum, Maaş, Deneyim Skoru, Yetenek Skoru
    return final_score, S_edu, S_loc, S_sal, S_Exp, S_Skills


# ====================================================================
# B. HYBRID RERANKING MANTIĞI (LLM İLE CANLI SKORLAMA)
# ====================================================================

# --- Simülasyon Verileri (LLM'e sadece ilgili 5 ilanı göndermek için) ---
CV_JOB_FILTERS = {
    'CV_1': ['Seyahat Danışmanı', 'Müşteri Danışmanı / İşe Alım Uzman Yardımcısı', 'Radyoloji Teknisyeni - Teknikeri', 'Satış Destek Sorumlusu', 'Bölüm Sekreteri'],
    'CV_5': ['Hemşire', 'Radyoloji Teknisyeni - Teknikeri', 'Ambulans Hekimi', 'Cildiye Uzmanı', 'Temizlik Görevlisi'],
    'CV_33': ['Diş Hekimi Asistanı', 'Radyoloji Teknisyeni - Teknikeri', 'Ambulans Hekimi', 'Cildiye Uzmanı', 'Bölüm Sekreteri'],
}

def rank_with_llm(query_cv, all_job_data):
    # LLM Modeli Tanımlama
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) 
    reranked_results = []
    
    # FAISS Simülasyonu: CV'ye özel ilan listesini çek
    relevant_job_titles = CV_JOB_FILTERS.get(query_cv['id'], [])
    jobs_to_rerank = [
        job for job in all_job_data if job['job_title'] in relevant_job_titles
    ]

    # Prompt (Cezalandırma ve JSON Çıktısı isteyen prompt)
    prompt_template = PromptTemplate.from_template("""
    Sen yüksek vasıflı bir İK Değerlendirme Uzmanısın. Amacın, adayın CV'sini bir iş ilanıyla karşılaştırarak **nihai mantıksal uyum skorunu (0-10)** ve nedenini (özellikle ceza varsa) döndürmektir.
    ADAY CV: {cv_name} ({cv_position})
    ADAYIN TEMEL YETKİNLİKLERİ: {cv_text}
    İLANA İLİŞKİN GEREKSİNİMLER: {job_qualifications}
    İLANA İLİŞKİN AÇIKLAMA: {job_description}
    --- YÖNERGE ---
    1. **ZORUNLU KURAL (CEZA):** Eğer ilan Tıp, Mühendislik veya Programlama gibi akademik bir unvan zorunluluğu getiriyorsa ve CV'de bu eğitim yoksa, puandan **6 puan DÜŞ** (Minimum puan 0 olabilir).
    2. **BAŞLANGIÇ PUANI:** Adayın iletişim, ekip çalışması ve genel hizmet becerileri için 7 puan ile başla.
    3. **NİHAİ ÇIKTI:** Nihai değerlendirmeni yap ve sadece aşağıdaki **JSON formatını** döndür. Başka hiçbir açıklama, giriş veya ek metin ekleme.
    TALEP EDİLEN JSON FORMATI: {{ "uyum_skoru": [PUAN (0-10)], "gerekce": "[SKORUN NEDENİ]" }}
    """)

    for job in jobs_to_rerank:
        llm_score = 0
        
        # CV Metnini Prompt için hazırla
        combined_cv_text = (
            query_cv.get('summary', '') + " " + 
            query_cv.get('experience_raw', '') + " " + 
            query_cv.get('skills_raw', '')
        )[:500].replace('\n', ' ').strip() 

        formatted_prompt = prompt_template.format(
            cv_name=query_cv.get('name', 'Aday'),
            cv_position=query_cv.get('position', 'Pozisyon'),
            cv_text=combined_cv_text,
            job_qualifications=job['qualifications_raw'],
            job_description=job['description']
        )
        
        # LLM'i çağır
        try:
            score_response_content = llm.invoke(formatted_prompt).content.strip()
            json_match = re.search(r'\{[\s\S]*\}', score_response_content)
            
            if json_match:
                json_string = json_match.group(0).replace('\n', ' ').replace('\r', '')
                response_json = json.loads(json_string) 
                llm_score = float(response_json.get("uyum_skoru", 0))
            
        except Exception as e:
             llm_score = 0
             
        job_result = job.copy()
        job_result['llm_score'] = llm_score
        reranked_results.append(job_result)

    # SONUÇLARI TEKRAR SIRALA (RERANK)
    reranked_results.sort(key=lambda x: x['llm_score'], reverse=True)
    
    return reranked_results


# ====================================================================
# C. FINAL EXECUTION (HAFTA 8 UYUM SKORU HESAPLAMASI)
# ====================================================================

if __name__ == '__main__':
    
    # --- Güvenlik Kontrolü ---
    if "OPENAI_API_KEY" not in os.environ:
        print("\nKRİTİK HATA: OPENAI_API_KEY ortam değişkeni ayarlanmamış.")
        sys.exit(1)

    # 1. Veriyi Yükle
    cv_data = load_data('parsed_cvs_FINAL.json')
    job_data = load_data('parsed_jobs_FINAL.json')
        
    if not cv_data or not job_data:
        print("Veri setleri eksik. Program sonlandırılıyor.")
        sys.exit(1)
        
    # --- ÇOKLU CV DÖNGÜSÜ ---
    
    DEMO_CV_IDS = ['CV_1', 'CV_5', 'CV_33'] # Bahar Tanay, Elif Kara, Ayla Yıldırım
    
    print("\n" + "="*70)
    print("HAFTA 8: FINAL AĞIRLIKLI UYUM SKORLARI (ÇOKLU CV DEMOSU)")
    print(f"Ağırlıklar: LLM(0.70) + Eğitim(0.10) + Konum(0.10) + Maaş/Kıdem(0.10)")
    print("="*70)

    for cv_id in DEMO_CV_IDS:
        
        query_cv = next((item for item in cv_data if item['id'] == cv_id), None)
        if not query_cv:
            continue
            
        print(f"\n[İŞLEM BAŞLADI] {query_cv.get('name', 'Bilinmiyor')} için LLM Analizi Çalışıyor...")
        
        # 3. Hybrid Reranking'i Çalıştır (LLM'den skorları alır)
        retrieved_jobs_llm_scored = rank_with_llm(query_cv, job_data) 

        final_score_table = []
        
        # 4. HAFTA 8: FINAL AĞIRLIKLI SKOR HESAPLAMA
        for llm_result in retrieved_jobs_llm_scored:
            
            # job_details'ı job_data'dan, job_title üzerinden bul
            job_details = next((item for item in job_data if item['job_title'] == llm_result['job_title']), None)
            
            if not job_details:
                continue

            # Kural tabanlı skorları hesapla
            # NOTE: Bu kısım, is_hard_academic_barrier_satisfied ve diğer fonksiyonları çağırır.
            final_score, S_edu, S_loc, S_sal, S_Exp, S_Skills = calculate_weighted_score(llm_result, query_cv, job_details)

            final_score_table.append({
    'Job Title': llm_result['job_title'],
    
    # LLM'in tek çıktısı olan 0-10 skoru
    'S_LLM_0_10': llm_result['llm_score'], 
    
    # HOCANIN İSTEDİĞİ AYRIŞTIRILMIŞ BİLEŞENLER (YENİ EKLENENLER)
    'S_Exp': S_Exp,
    'S_Skills': S_Skills,
    
    # Kural Tabanlı Bileşenler
    'S_edu': S_edu,
    'S_loc': S_loc,
    'S_sal': S_sal,
    
    'Final Weighted Score': final_score
})

        # Final tablosunu Nihai Skora göre sırala
        final_score_table.sort(key=lambda x: x['Final Weighted Score'], reverse=True)


        # 5. FİNAL SONUÇ RAPORU
        print("~"*70)
        print(f"CV: {query_cv['name']} ({query_cv['position']}) SONUÇLARI")
        print("~"*70)
        
        for i, result in enumerate(final_score_table[:5]): 
            print(f"Sıra {i+1} | NİHAİ SKOR: {result['Final Weighted Score']:.4f} / 1.00")
            print(f"   -> İlan: {result['Job Title']}")
            print(f"   -> S_EXP: {result['S_Exp']:.1f} (A: 0.35) | S_SKILLS: {result['S_Skills']:.1f} (A: 0.35)")
            print(f"   -> S_EDU: {result['S_edu']:.1f} (A: 0.10) | S_LOC: {result['S_loc']:.1f} (A: 0.10) | S_SAL: {result['S_sal']:.1f} (A: 0.10)")
            print("-" * 70)
        
    print("\n✅ Proje başarıyla tamamlandı. Dinamik Uyum Skoru hesaplandı.")