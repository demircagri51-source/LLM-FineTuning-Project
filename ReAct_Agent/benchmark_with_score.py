import pandas as pd
import ollama
from manual_agent import chat_loop, SYSTEM_PROMPT, rag_knowledge_base_tool, MODEL_ADI

# --- BENCHMARK VERİ SETİ (SORU + REFERANS GEMINI CEVABI) ---
BENCHMARK_DATA = [
    # --- KATEGORİ 1: KARŞILAŞTIRMA ---
    {"soru": "5 yıl çalışan birinin ihbar süresi, 6 ay çalışan birinin ihbar süresinden ne kadar fazladır?", "ref_cevap": "6 hafta fazladır (8 hafta - 2 hafta)."},
    {"soru": "Yıllık izin süresi 14 gün olan bir işçinin kıdemi, 20 gün izin hakkı olan birinden az mıdır?", "ref_cevap": "Evet, azdır. (14 gün alan 1-5 yıl arası, 20 gün alan 5-15 yıl arasıdır)."},
    {"soru": "Asgari ücretle çalışan birinin kıdem tazminatı tavanı, brüt ücretinden yüksek midir?", "ref_cevap": "Hayır, değildir. Kıdem tazminatı tavanı devlet memuru maaş katsayısına göre belirlenir, asgari ücretle ilişkisi dolaylıdır ama genelde yüksektir."},
    {"soru": "Kadın işçilerin doğum izni süresi, babalık izni süresinden kaç hafta daha uzundur?", "ref_cevap": "Doğum izni 16 hafta, babalık izni 5 gündür. Yaklaşık 15 hafta uzundur."},
    {"soru": "10 yıllık bir çalışanın yıllık izni, 3 yıllık bir çalışanın yıllık izninin iki katı mıdır?", "ref_cevap": "Hayır. 10 yıllık: 20 gün, 3 yıllık: 14 gün. İki katı değildir."},
    {"soru": "Deneme süresi içinde işten ayrılan biri ile 1 yıl sonra ayrılan biri arasındaki ihbar tazminatı farkı nedir?", "ref_cevap": "Deneme süresinde ihbar tazminatı yoktur (0 TL). 1 yıl sonra 4 haftalık ücret kadardır."},
    {"soru": "Bildirim süreleri (ihbar) işçi için mi daha uzundur yoksa işveren için mi, yoksa eşit midir?", "ref_cevap": "Eşittir. İş Kanunu madde 17'ye göre süreler her iki taraf için de aynıdır."},
    {"soru": "Haftalık 45 saati aşan çalışma ücreti ile resmi tatilde yapılan çalışma ücreti oranı aynı mıdır?", "ref_cevap": "Hayır. Fazla mesai %50 zamlı, resmi tatil çalışması %100 (1 günlük yevmiye) zamlı ödenir."},
    {"soru": "Kıdem tazminatı almak için gereken süre, yıllık izin hak etmek için gereken süreden uzun mudur?", "ref_cevap": "Hayır, eşittir. İkisi için de en az 1 yıl çalışmak gerekir."},
    {"soru": "İşverenin haklı nedenle fesih süresi, işçinin haklı nedenle fesih süresinden farklı mıdır?", "ref_cevap": "Hayır, hak düşürücü süreler (6 iş günü) her iki taraf için de aynıdır."},

    # --- KATEGORİ 2: HESAPLAMA ---
    {"soru": "3.5 yıl çalışan bir işçinin ihbar süresi ile yıllık izin süresi toplam kaç haftadır?", "ref_cevap": "İhbar: 6 hafta, Yıllık İzin: 2 hafta (14 gün). Toplam: 8 hafta."},
    {"soru": "Bir işçi haftada 50 saat çalışırsa, ayda kaç saat fazla mesai yapmış olur?", "ref_cevap": "Haftada 5 saat x 4 hafta = Ayda 20 saat fazla mesai."},
    {"soru": "15 yıl çalışan birinin yıllık izni ile 5 yıl çalışan birinin yıllık izni toplam kaç gündür?", "ref_cevap": "15 yıl: 20 gün (veya 26 gün yaşa göre değişir, genelde 20). 5 yıl: 14 gün. Toplam: 34 gün."},
    {"soru": "İhbar süresi 8 hafta olan bir işçi, günde 2 saat iş arama izni kullanırsa toplam kaç saat izin kullanmış olur?", "ref_cevap": "8 hafta x 6 gün (çalışma günü) x 2 saat = 96 saat."},
    {"soru": "7 aylık hamile bir çalışanın doğum öncesi ve sonrası toplam izin süresi kaç gündür?", "ref_cevap": "Toplam 16 hafta x 7 = 112 gündür."},
    {"soru": "Günlük 11 saati aşan çalışmalar fazla mesai sayılırsa, haftada 6 gün 12 saat çalışan biri kaç saat fazla mesai alır?", "ref_cevap": "Günde 1 saat x 6 gün = Haftada 6 saat fazla mesai."},
    {"soru": "Kıdem tazminatı her yıl için 30 günlük ücretse, 10 yıl 6 ay çalışan biri kaç aylık ücret tutarında tazminat alır?", "ref_cevap": "10.5 aylık brüt ücret tutarında alır."},
    {"soru": "Yıllık izin süresi cumartesileri de kapsıyorsa, 14 gün izin alan biri kaç pazar günü tatil yapar?", "ref_cevap": "14 gün içine 2 hafta sonu girer, yani 2 Pazar günü."},
    {"soru": "4 hafta ihbar süresi olan biri, bu süreyi çalışmadan peşin ödemek isterse kaç günlük ücret öder?", "ref_cevap": "4 hafta x 7 gün = 28 günlük ücret (ihbar tazminatı)."},
    {"soru": "Ara dinlenmesi 1 saat olan bir iş yerinde günde 9 saat bulunan işçi fiilen kaç saat çalışmış sayılır?", "ref_cevap": "9 saat - 1 saat ara = 8 saat fiili çalışma."},

    # --- KATEGORİ 3: KOŞULLU MANTIK ---
    {"soru": "Bir işçi iş yerinde kavga çıkarırsa kıdem tazminatı alarak işten ayrılabilir mi?", "ref_cevap": "Hayır, Madde 25/II'ye göre tazminatsız atılır."},
    {"soru": "11 aydır çalışan bir işçi işten çıkarılırsa yıllık izin ücretini talep edebilir mi?", "ref_cevap": "Hayır, yıllık izin hakkı 1 yıl dolunca doğar."},
    {"soru": "İşveren maaşı 20 gün geciktirirse işçi işi bırakıp tazminat isteyebilir mi?", "ref_cevap": "Evet, Madde 24'e göre haklı fesih yapabilir ve kıdem tazminatı alır."},
    {"soru": "Hamile olduğu için işten çıkarılan bir kadın işçi işe iade davası açabilir mi?", "ref_cevap": "Evet, hamilelik geçerli fesih nedeni değildir. İşe iade açabilir."},
    {"soru": "Deneme süresinin 3. ayında işten çıkarılan işçi ihbar tazminatı isteyebilir mi?", "ref_cevap": "Evet, yasal deneme süresi en çok 2 aydır. 3. ayda artık normal çalışandır, ihbar alır."},
    {"soru": "Raporlu olduğu günlerde işten çıkarılan işçinin feshi geçerli sayılır mı?", "ref_cevap": "Hayır, raporluyken yapılan bildirim geçersizdir veya rapor bitiminden sonra hüküm doğurur."},
    {"soru": "Kendi isteğiyle (istifa) ayrılan bir işçi, evlilik nedeniyle ayrılmışsa kıdem tazminatı alabilir mi?", "ref_cevap": "Evet, sadece kadın işçiler evlendikten sonra 1 yıl içinde ayrılırsa kıdem alır."},
    {"soru": "Belirli süreli iş sözleşmesi biten bir çalışan işe iade davası açabilir mi?", "ref_cevap": "Hayır, belirli süreli sözleşmelerde kendiliğinden sona erme durumunda işe iade davası açılamaz."},
    {"soru": "Günde 7.5 saat çalışan bir işçi gece postasında 8 saat çalıştırılabilir mi?", "ref_cevap": "Hayır, gece çalışmaları 7.5 saati geçemez."},
    {"soru": "Yıllık iznini kullanmayan işçi, izin süresinin parasını çalışırken isteyebilir mi?", "ref_cevap": "Hayır, izin parası sadece iş sözleşmesi bittiğinde ödenir. Çalışırken paraya çevrilemez."},

    # --- KATEGORİ 4: TERS KÖŞE ---
    {"soru": "İş Kanununa göre işçinin 'öğle uykusu izni' kaç saattir?", "ref_cevap": "Kanunda böyle bir izin yoktur."},
    {"soru": "5 yıl çalışan bir işçinin 'yıpranma tazminatı' İş Kanunu'nun hangi maddesindedir?", "ref_cevap": "İş Kanunu'nda genel bir yıpranma tazminatı maddesi yoktur (Basın/Deniz iş kanununda vardır)."},
    {"soru": "İşveren işçiye doğum gününde zorunlu ikramiye vermek zorunda mıdır?", "ref_cevap": "Hayır, sözleşmede yoksa yasal bir zorunluluk değildir."},
    {"soru": "Asgari ücretle çalışan biri işyerinden 'yol parası' almak zorunda mıdır?", "ref_cevap": "Hayır, yasal zorunluluk değildir. İşverenin inisiyatifindedir."},
    {"soru": "Yıllık izin süresi işveren tarafından paraya çevrilip her ay maaşa eklenebilir mi?", "ref_cevap": "Hayır, kesinlikle yasaktır. İzin fiilen kullandırılmalıdır."},
    {"soru": "İşçi, patronunu sevmediği için 'haklı nedenle' fesih yapabilir mi?", "ref_cevap": "Hayır, sevmemek haklı fesih nedeni değildir."},
    {"soru": "İhbar süresi içinde işçi günde 8 saat 'iş arama izni' kullanabilir mi?", "ref_cevap": "Hayır, yasal süre günde en az 2 saattir. 8 saat olamaz (işverenin izni yoksa)."},
    {"soru": "18 yaşından küçük işçiler gece vardiyasında çalıştırılabilir mi?", "ref_cevap": "Hayır, sanayi işlerinde 18 yaş altının gece çalıştırılması yasaktır."},
    {"soru": "Erkek işçilere evlendikleri zaman 'çeyiz yardımı' yapılması kanuni zorunluluk mudur?", "ref_cevap": "Hayır, kanunda böyle bir zorunluluk yoktur."},
    {"soru": "Hafta sonu tatili (Pazar) çalışması yapan işçi, ertesi gün 2 gün izin hak eder mi?", "ref_cevap": "Hayır, sadece o günün ücretini zamlı alır veya serbest zaman kullanır. 2 gün izin kuralı yoktur."}
]

def run_scored_benchmark():
    print(f"🚀 Benchmark Başlıyor ({len(BENCHMARK_DATA)} Soru)...")
    print(f"Model: {MODEL_ADI}")
    results = []

    for i, data in enumerate(BENCHMARK_DATA):
        soru = data['soru']
        ref_cevap = data['ref_cevap']
        
        print(f"\n[{i+1}/{len(BENCHMARK_DATA)}] Soru: {soru}")
        
        # Ajanı başlat
        conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Soru: {soru}"}
        ]
        
        agent_answer = "CEVAP BULUNAMADI"
        
        # ReAct Döngüsü
        for adim in range(5): 
            try:
                response = ollama.chat(
                    model=MODEL_ADI, 
                    messages=conversation,
                    options={'stop': ['Gözlem (Observation):', 'Gözlem:']}
                )
                cevap = response['message']['content']

                if "rag_knowledge_base_tool" in cevap:
                    anahtar_kelime = soru
                    if "Hareket Girdisi" in cevap and ":" in cevap:
                        anahtar_kelime = cevap.split(":", 1)[1].strip()
                    
                    print(f"   ⚙️ Arama: {anahtar_kelime[:30]}...")
                    arama_sonucu = rag_knowledge_base_tool(anahtar_kelime)
                    
                    conversation.append({'role': 'assistant', 'content': cevap})
                    conversation.append({'role': 'user', 'content': f"\nGözlem (Observation): {arama_sonucu}\n"})
                
                elif "Son Cevap" in cevap or "Final Answer" in cevap or "Cevap:" in cevap or "**Cevap**" in cevap:
                    agent_answer = cevap
                    print("   ✅ Cevap Alındı.")
                    break
                else:
                    conversation.append({'role': 'assistant', 'content': cevap})

            except Exception as e:
                print(f"Hata: {e}")
                break
        
        # Sonucu listeye ekle
        results.append({
            "Soru_ID": i+1,
            "Soru": soru,
            "Referans_Gemini_Cevabi": ref_cevap,
            "Senin_Ajaninin_Cevabi": agent_answer,
            "Puan_Durumu": "" # Burayı Excel'de sen dolduracaksın
        })

    # Excel'e Kaydet
    df = pd.DataFrame(results)
    dosya_adi = "Benchmark_Karsilastirmali_Sonuc.xlsx"
    df.to_excel(dosya_adi, index=False)
    print(f"\n🏁 BİTTİ! '{dosya_adi}' dosyası oluşturuldu.")
    print("Lütfen Excel'i açıp 'Puan_Durumu' sütununa Doğru için 1, Yanlış için 0 yaz.")

if __name__ == "__main__":
    run_scored_benchmark()