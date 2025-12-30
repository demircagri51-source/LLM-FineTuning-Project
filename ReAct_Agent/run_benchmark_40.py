import pandas as pd
import ollama
from manual_agent import chat_loop, SYSTEM_PROMPT, rag_knowledge_base_tool, MODEL_ADI

# --- 40 SORULUK BENCHMARK SETİ ---
BENCHMARK_QUESTIONS = [
    # --- KATEGORİ 1: KARŞILAŞTIRMA ---
    "5 yıl çalışan birinin ihbar süresi, 6 ay çalışan birinin ihbar süresinden ne kadar fazladır?",
    "Yıllık izin süresi 14 gün olan bir işçinin kıdemi, 20 gün izin hakkı olan birinden az mıdır?",
    "Asgari ücretle çalışan birinin kıdem tazminatı tavanı, brüt ücretinden yüksek midir?",
    "Kadın işçilerin doğum izni süresi, babalık izni süresinden kaç hafta daha uzundur?",
    "10 yıllık bir çalışanın yıllık izni, 3 yıllık bir çalışanın yıllık izninin iki katı mıdır?",
    "Deneme süresi içinde işten ayrılan biri ile 1 yıl sonra ayrılan biri arasındaki ihbar tazminatı farkı nedir?",
    "Bildirim süreleri (ihbar) işçi için mi daha uzundur yoksa işveren için mi, yoksa eşit midir?",
    "Haftalık 45 saati aşan çalışma ücreti ile resmi tatilde yapılan çalışma ücreti oranı aynı mıdır?",
    "Kıdem tazminatı almak için gereken süre, yıllık izin hak etmek için gereken süreden uzun mudur?",
    "İşverenin haklı nedenle fesih süresi, işçinin haklı nedenle fesih süresinden farklı mıdır?",

    # --- KATEGORİ 2: HESAPLAMA ---
    "3.5 yıl çalışan bir işçinin ihbar süresi ile yıllık izin süresi toplam kaç haftadır?",
    "Bir işçi haftada 50 saat çalışırsa, ayda kaç saat fazla mesai yapmış olur?",
    "15 yıl çalışan birinin yıllık izni ile 5 yıl çalışan birinin yıllık izni toplam kaç gündür?",
    "İhbar süresi 8 hafta olan bir işçi, günde 2 saat iş arama izni kullanırsa toplam kaç saat izin kullanmış olur?",
    "7 aylık hamile bir çalışanın doğum öncesi ve sonrası toplam izin süresi kaç gündür?",
    "Günlük 11 saati aşan çalışmalar fazla mesai sayılırsa, haftada 6 gün 12 saat çalışan biri kaç saat fazla mesai alır?",
    "Kıdem tazminatı her yıl için 30 günlük ücretse, 10 yıl 6 ay çalışan biri kaç aylık ücret tutarında tazminat alır?",
    "Yıllık izin süresi cumartesileri de kapsıyorsa, 14 gün izin alan biri kaç pazar günü tatil yapar?",
    "4 hafta ihbar süresi olan biri, bu süreyi çalışmadan peşin ödemek isterse kaç günlük ücret öder?",
    "Ara dinlenmesi 1 saat olan bir iş yerinde günde 9 saat bulunan işçi fiilen kaç saat çalışmış sayılır?",

    # --- KATEGORİ 3: KOŞULLU MANTIK ---
    "Bir işçi iş yerinde kavga çıkarırsa kıdem tazminatı alarak işten ayrılabilir mi?",
    "11 aydır çalışan bir işçi işten çıkarılırsa yıllık izin ücretini talep edebilir mi?",
    "İşveren maaşı 20 gün geciktirirse işçi işi bırakıp tazminat isteyebilir mi?",
    "Hamile olduğu için işten çıkarılan bir kadın işçi işe iade davası açabilir mi?",
    "Deneme süresinin 3. ayında işten çıkarılan işçi ihbar tazminatı isteyebilir mi?",
    "Raporlu olduğu günlerde işten çıkarılan işçinin feshi geçerli sayılır mı?",
    "Kendi isteğiyle (istifa) ayrılan bir işçi, evlilik nedeniyle ayrılmışsa kıdem tazminatı alabilir mi?",
    "Belirli süreli iş sözleşmesi biten bir çalışan işe iade davası açabilir mi?",
    "Günde 7.5 saat çalışan bir işçi gece postasında 8 saat çalıştırılabilir mi?",
    "Yıllık iznini kullanmayan işçi, izin süresinin parasını çalışırken isteyebilir mi?",

    # --- KATEGORİ 4: TERS KÖŞE (OUT OF DISTRIBUTION) ---
    "İş Kanununa göre işçinin 'öğle uykusu izni' kaç saattir?",
    "5 yıl çalışan bir işçinin 'yıpranma tazminatı' İş Kanunu'nun hangi maddesindedir?",
    "İşveren işçiye doğum gününde zorunlu ikramiye vermek zorunda mıdır?",
    "Asgari ücretle çalışan biri işyerinden 'yol parası' almak zorunda mıdır?",
    "Yıllık izin süresi işveren tarafından paraya çevrilip her ay maaşa eklenebilir mi?",
    "İşçi, patronunu sevmediği için 'haklı nedenle' fesih yapabilir mi?",
    "İhbar süresi içinde işçi günde 8 saat 'iş arama izni' kullanabilir mi?",
    "18 yaşından küçük işçiler gece vardiyasında çalıştırılabilir mi?",
    "Erkek işçilere evlendikleri zaman 'çeyiz yardımı' yapılması kanuni zorunluluk mudur?",
    "Hafta sonu tatili (Pazar) çalışması yapan işçi, ertesi gün 2 gün izin hak eder mi?"
]

def run_full_benchmark():
    print(f"🚀 {len(BENCHMARK_QUESTIONS)} Soruluk Benchmark Testi Başlıyor...")
    results = []

    for i, soru in enumerate(BENCHMARK_QUESTIONS):
        print(f"\n[{i+1}/{len(BENCHMARK_QUESTIONS)}] Soru Soruluyor: {soru}")
        
        # Ajanı her soru için sıfırdan başlatıyoruz
        conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Soru: {soru}"}
        ]
        
        final_answer = "CEVAP BULUNAMADI"
        tool_used = "HAYIR"
        reasoning_steps = []

        # ReAct Döngüsü
        for adim in range(5): # Max 5 adım
            try:
                response = ollama.chat(
                    model=MODEL_ADI, 
                    messages=conversation,
                    options={'stop': ['Gözlem (Observation):', 'Gözlem:']}
                )
                cevap = response['message']['content']
                reasoning_steps.append(f"Adım {adim+1}: {cevap[:100]}...") # Log için kısalt

                # Tool Kontrolü
                if "rag_knowledge_base_tool" in cevap:
                    tool_used = "EVET"
                    # Basit parse
                    anahtar_kelime = soru # Fallback
                    if "Hareket Girdisi" in cevap and ":" in cevap:
                        anahtar_kelime = cevap.split(":", 1)[1].strip()
                    
                    # Tool Çalıştır
                    print(f"   ⚙️ Arama Yapılıyor: {anahtar_kelime[:30]}...")
                    arama_sonucu = rag_knowledge_base_tool(anahtar_kelime)
                    
                    conversation.append({'role': 'assistant', 'content': cevap})
                    conversation.append({'role': 'user', 'content': f"\nGözlem (Observation): {arama_sonucu}\n"})
                
                elif "Son Cevap" in cevap or "Final Answer" in cevap or "Cevap:" in cevap or "**Cevap**" in cevap:
                    final_answer = cevap
                    print("   ✅ Cevap Bulundu.")
                    break
                else:
                    conversation.append({'role': 'assistant', 'content': cevap})

            except Exception as e:
                print(f"Hata: {e}")
                break
        
        # Sonucu kaydet
        results.append({
            "Soru_ID": i+1,
            "Soru": soru,
            "Kullanilan_Arac": tool_used,
            "Ajan_Cevabi": final_answer,
            "Adim_Sayisi": len(reasoning_steps)
        })

    # Sonuçları Excel/CSV olarak kaydet
    df = pd.DataFrame(results)
    df.to_excel("Benchmark_Sonuclari_40_Soru.xlsx", index=False)
    print("\n🏁 Benchmark tamamlandı! 'Benchmark_Sonuclari_40_Soru.xlsx' dosyasına bakabilirsin.")

if __name__ == "__main__":
    run_full_benchmark()