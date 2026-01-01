'use client'; 

import React, { useState, useCallback } from 'react';
import { FileText, Upload, RefreshCcw, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000'; 

interface MatchResult {
  job_title: string;
  general_score: number;
  skill_match: number;
  experience_match: number;
  report_summary: string;
}

const formatScore = (score: number) => (score * 100).toFixed(0) + '%';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<MatchResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleFileUpload = useCallback(async () => {
    if (!file) {
      setError("Lütfen bir CV dosyası seçin.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', file); 

    try {
      const response = await fetch(`${API_BASE_URL}/api/match_cv`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API Hatası`);
      }

      const data: MatchResult[] = await response.json(); 
      setResults(data);

    } catch (err) {
      console.error(err);
      setError("Sistem hatası. Lütfen tekrar deneyin.");
      setResults(null); 
    } finally {
      setLoading(false);
    }
  }, [file]);

  const ScoreBar = ({ score, label }: { score: number, label: string }) => {
    const color = score > 0.75 ? 'bg-green-600' : score > 0.5 ? 'bg-yellow-500' : 'bg-red-500';
    const width = `${score * 100}%`;

    return (
      <div className="mb-4">
        <div className="flex justify-between mb-1">
          <span className="text-sm font-bold text-gray-700">{label}</span>
          <span className="text-sm font-extrabold text-gray-900">{formatScore(score)}</span>
        </div>
        <div className="w-full bg-gray-300 rounded-full h-3">
          <div className={`h-3 rounded-full transition-all duration-500 ${color}`} style={{ width }}></div>
        </div>
      </div>
    );
  };

  // Büyük Uyarı Kartı
  const WarningCard = ({ message }: { message: string }) => (
    <div className="w-full max-w-2xl mx-auto mt-12 p-8 bg-orange-50 border-2 border-orange-200 rounded-2xl flex flex-col items-center text-center shadow-lg">
      <div className="bg-orange-100 p-4 rounded-full mb-6">
        <AlertTriangle className="w-16 h-16 text-orange-600" />
      </div>
      <h3 className="text-2xl font-bold text-orange-900 mb-4">Belge Analiz Edilemedi</h3>
      <p className="text-orange-800 font-medium text-lg leading-relaxed max-w-lg">
        {message}
      </p>
    </div>
  );

  // --- KONTROL MEKANİZMASI ---
  // Eğer sonuçlar varsa ve ilk sonucun puanı 0 ise bu bir hatadır.
  const isInvalidCV = results && results.length > 0 && results[0].general_score === 0;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-4 sm:p-8 font-inter text-gray-900">
      
      <header className="text-center mb-12 mt-8">
        <h1 className="text-5xl font-extrabold text-gray-900 tracking-tight">LLM Akıllı Eşleştirme</h1>
        <p className="text-lg text-gray-600 mt-3 font-medium">CV&apos;nizi yükleyin, en uygun iş fırsatlarını keşfedin.</p>
      </header>

      <div className="w-full max-w-4xl bg-white shadow-2xl rounded-2xl p-6 sm:p-10 border border-gray-200">
        
        {/* Dosya Yükleme */}
        <div className="flex flex-col sm:flex-row items-center justify-between border-b-2 border-gray-100 pb-8 mb-8">
          <div className="flex items-center space-x-4 mb-4 sm:mb-0">
            <div className="p-3 bg-blue-50 rounded-full">
              <FileText className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500 font-semibold uppercase tracking-wide">Seçilen Dosya</p>
              <span className="text-xl font-bold text-gray-800 block truncate max-w-[200px] sm:max-w-xs">
                {file ? file.name : "Dosya Bekleniyor..."}
              </span>
            </div>
          </div>

          <label htmlFor="file-upload" className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 flex items-center transform hover:-translate-y-0.5">
            <input id="file-upload" type="file" accept=".pdf,.docx,.txt" onChange={handleFileChange} className="hidden" disabled={loading} />
            <Upload className="w-5 h-5 mr-2" />
            <span className="font-bold">CV Yükle</span>
          </label>
        </div>

        <div className="flex flex-col items-center">
          <button
            onClick={handleFileUpload}
            disabled={!file || loading}
            className={`w-full sm:w-auto px-10 py-4 rounded-xl text-lg font-bold text-white transition-all duration-200 shadow-md ${
              !file || loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 active:bg-green-800'
            }`}
          >
            {loading ? (
              <div className="flex items-center"><RefreshCcw className="w-6 h-6 mr-3 animate-spin" />Analiz Yapılıyor...</div>
            ) : "Eşleştir"}
          </button>
          
          {error && (
            <div className="mt-6 flex items-center text-red-700 bg-red-100 p-4 rounded-lg w-full border border-red-200 font-medium">
              <XCircle className="w-6 h-6 mr-3" /> {error}
            </div>
          )}
        </div>

        {/* --- SONUÇ GÖSTERİM MANTIĞI --- */}
        {results && (
          <div className="mt-12 pt-8 border-t-2 border-gray-100">
            
            {/* DURUM 1: GEÇERSİZ CV -> Sadece Uyarı Göster */}
            {isInvalidCV ? (
               <WarningCard message={results[0].report_summary} />
            ) : (
            
            /* DURUM 2: GEÇERLİ CV -> Listeyi Göster */
            <>
                <h2 className="text-3xl font-extrabold text-gray-900 mb-8 flex items-center">
                <CheckCircle className="w-8 h-8 text-green-600 mr-3" />
                Sonuçlar
                </h2>
                
                <div className="space-y-8">
                {results.map((result, index) => (
                    <div key={index} className="p-6 bg-white border-2 border-gray-100 rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300">
                    <div className="flex items-start justify-between mb-4">
                        <h3 className="text-2xl font-bold text-gray-900 leading-tight">
                        <span className="text-blue-600 mr-2">#{index + 1}</span> {result.job_title}
                        </h3>
                        <span className={`px-4 py-1 rounded-full text-sm font-bold ${
                        result.general_score > 0.7 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                        }`}>
                        %{ (result.general_score * 100).toFixed(0) } Uyum
                        </span>
                    </div>

                    <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div className="bg-gray-50 p-5 rounded-xl border border-gray-200">
                        <h4 className="font-bold text-lg mb-4 text-gray-800 border-b border-gray-200 pb-2">Puan Detayı</h4>
                        <ScoreBar score={result.general_score} label="Genel Uyum" />
                        <ScoreBar score={result.skill_match} label="Yetenek" />
                        <ScoreBar score={result.experience_match} label="Deneyim" />
                        </div>
                        <div className="flex flex-col h-full">
                        <div className="bg-blue-50 p-5 rounded-xl border border-blue-100 h-full">
                            <h4 className="font-bold text-lg mb-3 text-blue-900 flex items-center"><FileText className="w-5 h-5 mr-2" />Yapay Zeka Özeti</h4>
                            <p className="text-gray-800 text-base leading-relaxed font-medium">{result.report_summary}</p>
                        </div>
                        </div>
                    </div>
                    </div>
                ))}
                </div>
            </>
            )}
            
          </div>
        )}

      </div>
    </div>
  );
}