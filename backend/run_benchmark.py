import json
import asyncio
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

# Senin server.py doslandan gerekli parçaları alıyoruz
# NOT: server.py ile bu dosya aynı klasörde olmalı
from server import process_single_job, matching_template, MatchResult, MODEL_NAME

# --- AYARLAR ---
GOLD_FILE = "gold_standard.json"
OUTPUT_EXCEL = "tez_performans_raporu.xlsx"

async def run_test():
    print(f"🔄 Gold Standart verisi yükleniyor: {GOLD_FILE}")
    
    try:
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("❌ HATA: gold_standard.json dosyası bulunamadı! Önce bu dosyayı oluşturmalısın.")
        return

    # Modeli hazırla
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=MatchResult)
    prompt = ChatPromptTemplate.from_template(matching_template)

    results = []
    
    print(f"🚀 {len(test_cases)} adet senaryo test ediliyor...\n")

    for case in test_cases:
        print(f"   🔹 Test Ediliyor: {case['job_title']} (ID: {case['id']})")
        
        # Sahte bir Document objesi oluştur (server.py yapısına uyması için)
        dummy_doc = Document(
            page_content="...", 
            metadata={
                "job_title": case['job_title'],
                "description": case['job_desc']
            }
        )

        # Senin sistemindeki fonksiyonu çağır
        try:
            ai_result = await process_single_job(
                doc=dummy_doc,
                cv_text=case['cv_text'],
                llm=llm,
                parser=parser,
                prompt_template=prompt
            )
            
            ai_score = ai_result.general_score if ai_result else 0.0
            
        except Exception as e:
            print(f"      ⚠️ Hata: {e}")
            ai_score = 0.0

        # İstatistikleri Hesapla
        human_score = case['human_score']
        diff = ai_score - human_score
        abs_diff = abs(diff) # Mutlak Hata (MAE için)
        
        # Başarı Kriteri: Yapay zeka insandan en fazla %25 (0.25) sapabilir
        is_successful = abs_diff <= 0.25
        
        results.append({
            "ID": case['id'],
            "Senaryo Tipi": case.get('case_type', 'Genel'),
            "İş İlanı": case['job_title'],
            "İnsan Puanı (Gold)": human_score,
            "AI Puanı (Sistem)": ai_score,
            "Fark": round(diff, 2),
            "Mutlak Hata": round(abs_diff, 2),
            "Başarılı mı?": "EVET" if is_successful else "HAYIR"
        })

    # --- RAPORLAMA ---
    df = pd.DataFrame(results)
    
    # Metrikler
    mae = df["Mutlak Hata"].mean()
    accuracy = (df[df["Başarılı mı?"] == "EVET"].count()["ID"] / len(df)) * 100
    correlation = df["İnsan Puanı (Gold)"].corr(df["AI Puanı (Sistem)"])

    print("\n" + "="*40)
    print("🎓 TEZ PERFORMANS SONUÇLARI")
    print("="*40)
    print(f"Toplam Test Sayısı: {len(df)}")
    print(f"Ortalama Hata (MAE): {mae:.3f} (Düşük olması iyidir)")
    print(f"Doğruluk Oranı (Accuracy): %{accuracy:.1f}")
    print(f"Korelasyon (Correlation): {correlation:.3f} (1'e yakın olması iyidir)")
    print("="*40)

    # Excel'e kaydet
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✅ Detaylı rapor kaydedildi: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    asyncio.run(run_test())