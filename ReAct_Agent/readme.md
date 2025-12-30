# ⚖️ Türk İş Hukuku ReAct Ajanı (Local LLM Agent)

Bu proje, **NLP & LLM Uygulamaları** dersi kapsamında geliştirilmiş; Büyük Dil Modellerini (LLM) dış bilgi kaynakları ve muhakeme (reasoning) yeteneği ile güçlendiren otonom bir yapay zeka asistanıdır.

Proje, standart bir RAG (Retrieval-Augmented Generation) sisteminden farklı olarak **ReAct (Reasoning + Acting)** mimarisini kullanır. Ajan, sadece veritabanından veri getirmez; soruyu analiz eder, hangi aracı kullanacağına karar verir, sonuçları değerlendirir ve eksik bilgi varsa tekrar araştırma yapar.

---

## 🚀 Öne Çıkan Özellikler

- **🧠 ReAct Mimarisi:** Ajan "Düşün -> Hareket Et -> Gözlemle -> Cevapla" döngüsü ile çalışır. Karmaşık ve çok adımlı (multi-hop) hukuk sorularını çözebilir.
- **🔒 Tamamen Yerel (Local):** Veri gizliliği ve güvenlik için **Ollama** üzerinden yerel modeller (`Qwen 2.5/3`) kullanılır. Hiçbir veri buluta gönderilmez.
- **📚 Vektör Hafıza:** Türk İş Kanunu maddeleri **FAISS** vektör veritabanında indekslenmiştir.
- **🛡️ Halüsinasyon Kontrolü:** Model, kanunda olmayan bir şeyi (örn: "Öğle uykusu izni") uydurmak yerine, veritabanında bulamadığını raporlayarak dürüst cevap verir.

---

## 🛠️ Teknoloji Yığını

- **LLM:** Qwen 2.5-Coder / Qwen3:4b (Ollama ile çalışır)
- **Orkestrasyon:** Python (Custom ReAct Loop)
- **Veritabanı:** FAISS (Facebook AI Similarity Search)
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Kütüphaneler:** LangChain, Pandas, OpenPyXL

---

## ⚙️ Kurulum Rehberi

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimleri Yükleyin
Python 3.10+ kurulu olduğundan emin olun ve gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```
### 2. Ajan ile Sohbet (Ana Mod)
Ajanla interaktif olarak konuşmak ve hukuk soruları sormak için:
```bash
python manual_agent.py
```
