import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- AYARLAR ---
MODEL_ADI = "qwen3:4b"  # Senin kullandığın model
DB_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TURNS = 5  # Hoca'nın istediği döngü limiti [cite: 52]

print("🧠 ReAct Hukuk Ajanı Başlatılıyor...")

# 1. VERİ KATMANI (The Knowledge) [cite: 100]
# Vektör veritabanını yüklüyoruz.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# 2. ARAÇ KATMANI (The Limbs) [cite: 96]
# PDF Adım 2: RAG Tool Wrapper [cite: 114]
def rag_knowledge_base_tool(sorgu):
    """
    Şirket içi dökümanlarda (İş Kanunu) arama yapar.
    LLM cevabı değil, ham metin (Raw Context) döndürür[cite: 113].
    """
    print(f"\n⚙️ [SİSTEM]: Veritabanında '{sorgu}' aranıyor...")
    results = db.similarity_search(sorgu, k=4)
    # Bulunan metinleri birleştirip ham veri olarak dönüyoruz
    raw_context = "\n".join([doc.page_content for doc in results])
    return raw_context

# 3. ORKESTRASYON KATMANI (The Brain) [cite: 92]
# PDF Adım 3: ReAct Prompt Tasarımı [cite: 120]
SYSTEM_PROMPT = """
Sen bir ReAct ajanısın. Görevin İş Kanunu hakkında sorulan soruları cevaplamaktır.
Aşağıdaki araçlara erişimin var:

1. rag_knowledge_base_tool: İş Kanunu maddelerinde arama yapar. Sadece dökümanlardan bilgi gerektiren sorularda kullan.

Soruyu cevaplamak için şu formatı TİTİZLİKLE takip et:

Soru: Cevaplaman gereken soru.
Düşünce (Thought): Soruya cevap vermek için ne yapmalıyım? Hangi aracı kullanmalıyım?
Hareket (Action): rag_knowledge_base_tool
Hareket Girdisi (Action Input): [Aranacak kısa anahtar kelime]

(BURADA DUR. SİSTEM SANA GÖZLEMİ VERECEK)

Gözlem (Observation): [Araçtan gelen bilgi buraya gelecek]
Düşünce (Thought): Gelen bilgiyi okudum. Cevabı buldum mu?
Son Cevap (Final Answer): Sorunun nihai cevabı (Türkçe).

ÖNEMLİ KURALLAR:
- Asla kendi bilgine güvenme, mutlaka 'rag_knowledge_base_tool' ile kontrol et[cite: 126].
- "Hareket" ve "Hareket Girdisi" yazdıktan sonra DUR. Gözlem kısmını uydurma.
- Her zaman Türkçe düşün ve cevap ver[cite: 55].

Haydi başlayalım!
"""

def chat_loop():
    print(f"⚖️  Ajan Hazır! (Çıkış için 'q')")
    
    while True:
        user_input = input("\nSiz: ")
        if user_input.lower() == "q":
            break
            
        # Sohbet geçmişini başlat
        conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Soru: {user_input}"}
        ]

        
        # ReAct Döngüsü
        for adim in range(MAX_TURNS):
            # 1. Modelin Düşünmesi (Reasoning)
            # STOP Parametresi: Model "Gözlem" yazacağı an otomatik susacak.
            response = ollama.chat(
                    model=MODEL_ADI, 
                    messages=conversation,
                    options={'stop': ['Gözlem (Observation):', 'Gözlem:']}
            )
            cevap = response['message']['content']
            print(f"\n🤖 [ADIM {adim+1}]:\n{cevap}")
            
            # 2. Aksiyon Kontrolü (Acting)
            if "rag_knowledge_base_tool" in cevap:
                # Anahtar kelimeyi ayıkla
                try:
                    satirlar = cevap.split('\n')
                    anahtar_kelime = ""
                    for satir in satirlar:
                        if "Hareket Girdisi" in satir and ":" in satir:
                            anahtar_kelime = satir.split(":", 1)[1].strip()
                            break
                    
                    if not anahtar_kelime:
                        anahtar_kelime = user_input 

                    # 3. Aracı Çalıştır (Tool Execution)
                    arama_sonucu = rag_knowledge_base_tool(anahtar_kelime)
                    
                    # 4. Gözlem (Observation)
                    gozlem_mesaji = f"\nGözlem (Observation): {arama_sonucu}\nŞimdi bu bilgiyle tekrar düşün."
                    print(f"📄 [BULUNAN BİLGİ]: {arama_sonucu[:150]}...") 
                    
                    conversation.append({'role': 'assistant', 'content': cevap})
                    conversation.append({'role': 'user', 'content': gozlem_mesaji})

                    # --- YENİ EKLENEN: Bekleme Mesajı ---
                    print("\n🤔 [SİSTEM]: Model bulunan bilgiyi okuyor ve cevap hazırlıyor... (Lütfen bekleyin)")
                    
                except Exception as e:
                    print(f"Hata: {e}")
                    break

            
            # Eğer model cevabı verdiyse döngüyü bitir 
            elif "Son Cevap" in cevap or "Final Answer" in cevap or "Cevap:" in cevap or "**Cevap**" in cevap:
                print(f"\n✅ [BİTİŞ]: Cevap bulundu, döngü sonlandırılıyor.")
                break

if __name__ == "__main__":
    chat_loop()