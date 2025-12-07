import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# --- A. VERİ YÜKLEME VE TEMİZLEME FONKSİYONLARI ---

def load_data(file_name):
    """Final JSON dosyasını okur."""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"HATA: {file_name} dosyası bulunamadı.")
        return []

def create_combined_text(data, is_cv=True):
    """CV veya İlan parçalarını modelin anlamlandıracağı tek bir metin bloğunda birleştirir."""
    
    combined_list = []
    
    if is_cv:
        for item in data:
            # Özet, Deneyim ve Becerileri birleştiriyoruz.
            combined_text = (
                item.get('summary', '') + " " +
                item.get('experience_raw', '') + " " +
                item.get('skills_raw', '')
            ).strip()
            combined_list.append({
                'id': item['id'],
                'name': item['name'],
                'position': item.get('position', 'Bilinmiyor'), # Position'ı ekledik
                'text': combined_text 
            })
    else: # İş İlanları için
        for item in data:
            # Açıklama, Nitelikler ve başlıkları birleştiriyoruz.
            combined_text = (
                item.get('job_title', '') + " " +
                item.get('sector', '') + " " +
                item.get('description', '') + " " +
                item.get('qualifications_raw', '')
            ).strip()
            combined_list.append({
                'id': item['id'],
                'job_title': item['job_title'],
                'sector': item['sector'],
                'text': combined_text
            })

    return combined_list

# --- B. EMBEDDING VE VEKTÖR DEPO YÖNETİMİ ---

def create_embeddings_and_index(texts, job_ids, model_name='all-MiniLM-L6-v2'):
    """Metin listesini vektörlere dönüştürür ve FAISS indeksi oluşturur."""
    
    print(f"\n[MODEL YÜKLENİYOR] Sentence Transformer: {model_name}")
    # Multilingual model yükleniyor (Türkçe/İngilizce desteği için)
    model = SentenceTransformer(model_name)
    
    print("[VEKTÖRLEŞTİRİLİYOR] Metinler vektörlere dönüştürülüyor...")
    # Metinleri vektörlere dönüştür
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Vektörlerin boyutunu al (Embedding Boyutu)
    embedding_dim = embeddings.shape[1]
    
    # FAISS indeksi oluşturma (Vektör Deposu)
    # IndexFlatL2, Öklid mesafesine göre benzerlik arar.
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Vektörleri FAISS indeksine ekle
    index.add(embeddings)
    
    print(f"[FAISS BAŞARILI] {index.ntotal} adet vektör FAISS indeksine eklendi. Boyut: {embedding_dim}")
    
    return index, embeddings, job_ids

# --- C. ANA ÇALIŞTIRMA BLOĞU ---

if __name__ == '__main__':
    # 1. Veriyi Yükle (Hafta 4'ün çıktıları)
    parsed_cvs = load_data('parsed_cvs_FINAL.json')
    parsed_jobs = load_data('parsed_jobs_FINAL.json')

    if not parsed_cvs or not parsed_jobs:
        print("Veri setleri eksik. Lütfen 'parsed_cvs_FINAL.json' ve 'parsed_jobs_FINAL.json' dosyalarını kontrol edin.")
    else:
        # 2. Metinleri Birleştir
        cv_data = create_combined_text(parsed_cvs, is_cv=True)
        job_data = create_combined_text(parsed_jobs, is_cv=False)
        
        # Sadece metin listelerini al
        job_texts = [item['text'] for item in job_data]
        job_ids = [item['id'] for item in job_data]
        
        # 3. Embedding ve FAISS İndeksi Oluştur
        job_index, job_embeddings, job_ids = create_embeddings_and_index(job_texts, job_ids)

        # 4. Basit Eşleştirme Örneği (Hafta 5 Teslimatı)
        
        # Eşleştirmek istediğimiz bir CV seçelim (Örn: CV 1 - Bahar Tanay, Kabin Memuru)
        query_cv = cv_data[0] 
        query_text = query_cv['text']
        
        print(f"\n--- SORGULANAN CV: {query_cv['name']} ({query_cv['position']}) ---")
        
        # Sorgu metnini vektöre dönüştür
        query_embedding = job_index.reconstruct(0).reshape(1, -1) # Basitçe ilk ilan vektörünü sorgu yapalım
        query_embedding = job_index.reconstruct(0).reshape(1, -1) # Düzeltme: Model ile sorgu vektörünü oluşturmalıyız
        
        # Yeni model yükleyip sorguyu vektörleştirelim:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query_text], convert_to_numpy=True)


        # En yakın 5 iş ilanını FAISS ile bul
        k = 5 # En yakın 5 ilanı bul
        distances, indices = job_index.search(query_embedding, k)
        
        print(f"\n[SONUÇ] En yakın {k} İlan (CV: {query_cv['name']}):")
        
        for i, idx in enumerate(indices[0]):
            match_job = job_data[idx]
            # FAISS mesafesini benzerlik puanına (0-1 arası) dönüştürelim
            # Öklid mesafesi için basit ters çevirme (score = 1 / (1 + distance))
            similarity_score = 1 / (1 + distances[0][i])
            
            print(f"{i+1}. {match_job['id']} ({match_job['job_title']}) - Benzerlik Skoru: {similarity_score:.4f}")