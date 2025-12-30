import os
# Buradaki import hatasini cozmek icin en garanti kutuphaneleri kullaniyoruz
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- AYARLAR ---
MODEL_ADI = "qwen2.5-coder:1.5b"  # BURAYI KENDI MODEL ADINLA DEGISTIR
DB_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("🧠 Agent baslatiliyor...")

# 1. FAISS Veritabanini Yukle
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# allow_dangerous_deserialization=True guvenlik uyarisini asmak icin gereklidir (yerel dosya oldugu icin guvenli)
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# 2. Arac (Tool) Fonksiyonu
def kanun_maddesi_ara(query):
    print(f"\n🔎 ARAÇ: '{query}' için kanun taranıyor...")
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

tools = [
    Tool(
        name="kanun_arama_araci",
        func=kanun_maddesi_ara,
        description="Kanun, haklar, tazminat veya yasal sureclerle ilgili sorularda MUTLAKA bu araci kullan."
    )
]

# 3. LLM Baglantisi
llm = ChatOllama(model=MODEL_ADI, temperature=0)

# 4. Prompt (ReAct Mantigi)
template = """
Sen uzman bir Türk Hukuk Müşavirisini.
Sorulan sorulara cevap vermeden önce MUTLAKA 'kanun_arama_araci' kullanarak bilgi topla.
Kafandan cevap uydurma.

Kullanabilecegin Araclar:
{tools}

Soru: {input}

Düşünce (Thought): Ne yapmam gerekiyor?
Hareket (Action): Hangi araci secmeliyim? [{tool_names}]
Hareket Girdisi (Action Input): Aramam gereken kelime nedir?
Gözlem (Observation): Aractan gelen bilgi nedir?
... (Gerekiyorsa tekrar et)
Düşünce (Thought): Cevabi buldum.
Son Cevap (Final Answer): Sorunun cevabi (Turkce).

Haydi Basla!

Soru: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 5. Agent Olusturma
try:
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    print(f"⚖️ AI Avukat ({MODEL_ADI}) Hazir! (Cikmak icin 'q' yaz)")
    
    while True:
        soru = input("\nSiz: ")
        if soru.lower() == 'q':
            break
        response = agent_executor.invoke({"input": soru})
        print(f"\n🤖 Cevap: {response['output']}")

except ImportError:
    print("HATA: Langchain surumlerinde uyumsuzluk var. 'pip install langchain==0.1.20' deneyebilirsin.")
except Exception as e:
    print(f"Bir hata olustu: {e}")