import json
import matplotlib.pyplot as plt
import sys
import os

# --- AYARLAR ---

# 1. DEEP Modelinin Yolu (Senin hatadan kopyalayıp düzelttim):
DEEP_LOG_PATH = r"C:\Users\Çağrı\OneDrive\Masaüstü\nlp\lora_project\CodeGen\models\deep_instruction\checkpoints\checkpoint-step-1689-epoch-4\trainer_state.json"

# 2. DIVERSE Modelinin Yolu (BUNU SENİN BULUP YAPISTIRMAN LAZIM):
# Lütfen diverse modelinin en son checkpoint klasöründeki trainer_state.json yolunu tırnakların içine yapıştır.
# Başına 'r' koymayı unutma!
DIVERSE_LOG_PATH = r"C:\Users\Çağrı\OneDrive\Masaüstü\nlp\lora_project\CodeGen\models\diverse_instruction\checkpoints\checkpoint-step-846-epoch-2\trainer_state.json" 


def ciz(json_path, model_name):
    print(f"Grafik hazırlanıyor: {model_name}...")
    
    # Yol girilmemişse uyarı ver
    if "BURAYA" in json_path:
        print(f"⚠️ UYARI: {model_name} için dosya yolu girilmemiş, bu grafik atlanıyor.")
        return

    if not os.path.exists(json_path):
        print(f"❌ HATA: Dosya bulunamadı -> {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ HATA: Dosya okunamadı. Sebebi: {e}")
        return

    history = data.get('log_history', [])
    steps = []
    losses = []
    
    # Verileri ayıkla
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])

    if not steps:
        print(f"❌ HATA: {model_name} dosyasında 'loss' verisi bulunamadı!")
        return

    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(10, 6))
    
    # Çizgi (Trendi görmek için)
    plt.plot(steps, losses, label='Loss Trendi', color='blue', alpha=0.3)
    
    # Noktalar (Hocanın isteği: Belirli aralıklarla)
    marker_steps = []
    marker_losses = []
    
    for s, l in zip(steps, losses):
        # Her 20 adımda bir veya veri azsa hepsini işaretle
        if s % 20 == 0: 
            marker_steps.append(s)
            marker_losses.append(l)

    # Hiç 20'ye denk gelmezse hepsini işaretle
    if not marker_steps:
        marker_steps = steps
        marker_losses = losses

    plt.scatter(marker_steps, marker_losses, color='red', s=25, zorder=5, label='İşaretli Adımlar (Her 20)')

    plt.title(f'{model_name} Eğitim Kayıp (Loss) Grafiği')
    plt.xlabel('Adım (Step)')
    plt.ylabel('Loss (Hata)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_file = f"{model_name}_loss_grafigi.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Başarılı! Grafik kaydedildi: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Deep modelini çiz
    ciz(DEEP_LOG_PATH, "Deep_Instruction")
    
    print("-" * 30)
    
    # Diverse modelini çiz
    ciz(DIVERSE_LOG_PATH, "Diverse_Instruction")