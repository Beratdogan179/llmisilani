'use client'; // Next.js App Router'da olay yöneticileri için gereklidir

import React, { useState, useCallback } from 'react';
import { FileText, Upload, RefreshCcw, CheckCircle, XCircle } from 'lucide-react';

// Varsayılan API URL'si. Backend'i (FastAPI) bu adreste çalıştırmanız gerekir.
const API_BASE_URL = 'http://localhost:8000'; 

// TypeScript tipleri
interface MatchResult {
  job_title: string;
  general_score: number;
  skill_match: number;
  experience_match: number;
  report_summary: string;
}

const formatScore = (score: number) => (score * 100).toFixed(1) + '%';

// Ana Uygulama Bileşeni
export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<MatchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Dosya seçme işlemi
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  // CV yükleme ve API'ye gönderme işlemi
  const handleFileUpload = useCallback(async () => {
    if (!file) {
      setError("Lütfen bir CV dosyası seçin.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    // FastAPI'nin 'file' adında bir alan beklediğini varsayıyoruz.
    formData.append('file', file); 

    try {
      // Örnek: FastAPI endpoint'ine POST isteği gönderiyoruz.
      const response = await fetch(`${API_BASE_URL}/api/match_cv`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // API tarafında 4xx veya 5xx hatası varsa
        const errorData = await response.json();
        throw new Error(errorData.detail || `API hatası: ${response.status}`);
      }

      // Başarılı yanıtı JSON olarak alıyoruz (Burada mock verisi kullanılmaktadır)
      const data: MatchResult[] = await response.json(); 
      setResults(data);

    } catch (err) {
      console.error("Yükleme veya eşleştirme hatası:", err);
      setError(`Eşleştirme başarısız oldu. Hata: ${err instanceof Error ? err.message : String(err)}`);
      
      // 👇 ESKİ KOD: setResults(mockResults); (BUNU SİLİYORUZ)
      
      // 👇 YENİ KOD: Sonuçları sıfırlıyoruz, böylece ekranda eski veya sahte veri kalmaz.
      setResults(null); 

    } finally {
      setLoading(false);
    }
  }, [file]);

  // Skor çubuğu bileşeni
  const ScoreBar = ({ score, label }: { score: number, label: string }) => {
    const color = score > 0.75 ? 'bg-green-500' : score > 0.5 ? 'bg-yellow-500' : 'bg-red-500';
    const width = `${score * 100}%`;

    return (
      <div className="mb-2">
        <div className="flex justify-between mb-1 text-sm">
          <span>{label}</span>
          <span className="font-semibold">{formatScore(score)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div className={`h-2.5 rounded-full transition-all duration-500 ${color}`} style={{ width }}></div>
        </div>
      </div>
    );
  };

  

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-4 sm:p-8 font-inter">
      <header className="text-center mb-10 mt-5">
        <h1 className="text-4xl font-bold text-gray-800">LLM Akıllı Eşleştirme Sistemi</h1>
        <p className="text-gray-500 mt-2">CV nizi yükleyin, en uygun iş ilanlarını saniyeler içinde bulun.</p>
      </header>

      <div className="w-full max-w-4xl bg-white shadow-xl rounded-xl p-6 sm:p-10 border border-gray-100">
        
        {/* Yükleme Alanı */}
        <div className="flex flex-col sm:flex-row items-center justify-between border-b pb-6 mb-6">
          <div className="flex items-center space-x-3 mb-4 sm:mb-0">
            <FileText className="w-8 h-8 text-blue-500" />
            <span className="text-lg font-medium text-gray-700">
              {file ? file.name : "CV Dosyası Seçilmedi"}
            </span>
          </div>

          <label htmlFor="file-upload" className="cursor-pointer bg-blue-600 text-white px-5 py-2.5 rounded-lg shadow-md hover:bg-blue-700 transition duration-150 flex items-center">
            <input 
              id="file-upload" 
              type="file" 
              accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
              onChange={handleFileChange} 
              className="hidden" 
              disabled={loading}
            />
            <Upload className="w-5 h-5 mr-2" />
            Dosya Seç ({file ? 'Değiştir' : 'Seç'})
          </label>
        </div>

        {/* Eşleştirme Butonu ve Durumlar */}
        <div className="flex flex-col items-center">
          <button
            onClick={handleFileUpload}
            disabled={!file || loading}
            className={`w-full sm:w-auto px-8 py-3 rounded-xl text-white font-semibold transition duration-200 shadow-lg ${
              !file || loading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-green-600 hover:bg-green-700 active:bg-green-800'
            }`}
          >
            {loading ? (
              <div className="flex items-center">
                <RefreshCcw className="w-5 h-5 mr-2 animate-spin" />
                Eşleştiriliyor...
              </div>
            ) : (
              "Eşleştirmeyi Başlat"
            )}
          </button>
          
          {error && (
            <div className="mt-4 flex items-center text-red-600 bg-red-50 p-3 rounded-lg w-full max-w-sm">
              <XCircle className="w-5 h-5 mr-2" />
              {error}
            </div>
          )}
        </div>

        {/* Sonuçlar Alanı */}
        {results && (
          <div className="mt-10 pt-6 border-t border-gray-200">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <CheckCircle className="w-6 h-6 text-green-500 mr-2" />
              En Uygun İş İlanları
            </h2>
            
            <div className="space-y-6">
              {results.map((result, index) => (
                <div key={index} className="p-5 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition duration-200">
                  <h3 className={`text-xl font-bold ${result.general_score > 0.8 ? 'text-blue-700' : 'text-gray-700'}`}>
                    {index + 1}. {result.job_title}
                  </h3>
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                    
                    {/* Skorlar */}
                    <div>
                      <h4 className="font-semibold text-lg mb-2 text-gray-600">Uyum Skorları</h4>
                      <ScoreBar score={result.general_score} label="Genel Uyum Skoru" />
                      <ScoreBar score={result.skill_match} label="Yetenek Eşleşmesi" />
                      <ScoreBar score={result.experience_match} label="Deneyim Uyumu" />
                    </div>

                    {/* Rapor Özeti */}
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                      <h4 className="font-semibold text-lg mb-1 text-gray-600">Önerilen İK Rapor Özeti</h4>
                      <p className="text-sm text-gray-600">{result.report_summary}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
