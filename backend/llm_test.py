import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# API Anahtarını kontrol et
if "OPENAI_API_KEY" not in os.environ:
    print("KRİTİK HATA: OPENAI_API_KEY ayarlanmamış.")
else:
    print("✅ API Anahtarı bulundu. Bağlantı Testi Başlıyor...")
    
    try:
        # LLM Modeli Tanımlama (Çok hızlı, düşük sıcaklıkta)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) 
        
        # Basit Sorgu Şablonu
        prompt = PromptTemplate.from_template("Sadece 'TAMAM' kelimesini döndür. Başka hiçbir şey yazma.")
        
        # LLM'i çağır
        response = llm.invoke(prompt.format())
        
        print("\n--- LLM BAĞLANTI TESTİ SONUCU ---")
        print(f"LLM Yanıtı: {response.content.strip()}")
        
        if response.content.strip() == "TAMAM":
            print("\n🎉 BAŞARILI: LLM API Bağlantısı Kuruldu ve Çalışıyor!")
        else:
            print("\n❌ HATA: LLM Bağlandı ancak Yanlış Yanıt Verdi (API'de sorun olabilir).")
            
    except Exception as e:
        print("\n❌ KRİTİK HATA: LLM Bağlantısı Kurulamadı!")
        print(f"Hata detayı: {e}")