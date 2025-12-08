# LLM Tabanlı Akıllı İş İlanı – Aday Eşleştirme Sistemi (Hybrid Reranking)

Bu proje, geleneksel anahtar kelime eşleşmesinin ötesine geçerek, **Büyük Dil Modelleri (LLM)** ve **Vektör Arama (FAISS)** teknolojilerini birleştiren hibrit bir işe alım asistanıdır. Adayların özgeçmişlerini (CV) iş ilanları ile anlamsal, mantıksal ve kural tabanlı olarak eşleştirir.

## Temel Özellikler

* **Çoklu Format Desteği:** PDF, DOCX, PNG ve JPG formatındaki CV'leri işleyebilir (OCR Entegreli).
* **İki Aşamalı Hibrit Sıralama (Hybrid Reranking):**
    1.  **Aşama 1 (Geniş Filtreleme):** FAISS Vektör Arama ile aday havuzunu tarar.
    2.  **Aşama 2 (Akıllı Sıralama):** OpenAI GPT-3.5 Turbo ile adayları mantıksal olarak analiz eder.
* **Zorunlu Kısıtlamalar (Hard Constraints):** Tıp, Hukuk, Mühendislik gibi alanlarda akademik uyumsuzlukları tespit eder ve puan cezası uygular (Örn: Mühendis CV'sine Garson ilanı önerilmez).
* **Açıklanabilir Yapay Zeka:** Her eşleşme için "Neden Uygun?" veya "Neden Uygun Değil?" şeklinde İK raporu üretir.

## 🛠️ Teknoloji Yığını

* **Backend:** Python, FastAPI
* **Yapay Zeka:** LangChain, OpenAI, Sentence-Transformers
* **Veri tabanı:** FAISS (Vektör), JSON (Veri)
* **Frontend:** Next.js, React, Tailwind CSS
* **Veri İşleme:** PyTesseract (OCR), PDFPlumber

## ⚙️ Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

### Ön Hazırlık
* Python 3.10+
* Node.js & npm
* **Tesseract OCR** (Sisteminizde kurulu olmalıdır)

### 1. Backend (API) Kurulumu

```bash
cd backend

# Sanal ortamı oluştur ve aktif et
python -m venv llm_env
source llm_env/bin/activate  # Windows: llm_env\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# OpenAI API Anahtarını Tanımla
export OPENAI_API_KEY='sk-...'

# Sunucuyu Başlat
python llm_api.py