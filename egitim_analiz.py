import json
import os
import matplotlib.pyplot as plt

# --- AYARLAR ---
OUTPUT_DIR = "qwen-lora-diverse"  # Analiz edilecek klasör (Deep veya Diverse)

def plot_training_history():
    print(f"'{OUTPUT_DIR}' klasöründeki eğitim geçmişi analiz ediliyor...\n")
    
    # 1. trainer_state.json dosyasını bul
    state_file = None
    potential_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
    
    if os.path.exists(potential_path):
        state_file = potential_path
    else:
        # Ana klasörde yoksa son checkpoint'e bak
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            last_checkpoint = checkpoints[-1]
            state_file = os.path.join(OUTPUT_DIR, last_checkpoint, "trainer_state.json")

    if not state_file or not os.path.exists(state_file):
        print("HATA: Log dosyası (trainer_state.json) bulunamadı.")
        return

    # 2. Veriyi Oku
    with open(state_file, 'r') as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    
    # 3. Verileri Ayıkla
    train_steps = []
    train_losses = []
    
    eval_steps = []
    eval_losses = []
    
    for log in log_history:
        # Train Loss Kaydı (Genelde 'loss' anahtarı ile tutulur)
        if "loss" in log and "eval_loss" not in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])
            
        # Eval Loss Kaydı
        if "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_losses.append(log["eval_loss"])

    # 4. Tablo Olarak Yazdır
    print(f"{'ADIM':<10} | {'TRAIN LOSS':<12} | {'EVAL LOSS':<12}")
    print("-" * 40)
    
    # Verileri eşleştirip yazdırma (Adımlar farklı olabilir, yakın olanları gösteririz)
    # Genellikle eval adımları daha seyrektir, o yüzden eval üzerinden döngü kuralım
    if eval_steps:
        for e_step, e_loss in zip(eval_steps, eval_losses):
            # Bu adıma en yakın train loss'u bulalım
            # (Basitçe o adımdaki veya önceki en son train loss)
            try:
                t_loss = next(l for s, l in zip(train_steps, train_losses) if s == e_step)
                print(f"{e_step:<10} | {t_loss:.4f}       | {e_loss:.4f}")
            except StopIteration:
                print(f"{e_step:<10} | {'-':<12} | {e_loss:.4f}")
    else:
        print("Henüz Eval verisi yok, sadece Train verileri listeleniyor:")
        for t_step, t_loss in zip(train_steps, train_losses):
             print(f"{t_step:<10} | {t_loss:.4f}       | -")

    # 5. Grafik Çizdirme (Görsel Analiz)
    plt.figure(figsize=(10, 6))
    
    # Train Loss Çizgisi (Mavi)
    if train_steps:
        plt.plot(train_steps, train_losses, label='Training Loss', color='blue', alpha=0.6)
    
    # Eval Loss Çizgisi (Kırmızı - Kalın)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label='Validation (Eval) Loss', color='red', linewidth=2, marker='o')
    
    plt.title('Eğitim Analizi: Train Loss vs Eval Loss')
    plt.xlabel('Eğitim Adımları (Steps)')
    plt.ylabel('Hata Oranı (Loss)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    print("\nGrafik çiziliyor, pencere açılacak...")
    plt.show()

if __name__ == "__main__":
    plot_training_history()