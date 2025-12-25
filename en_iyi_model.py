import json
import os
import matplotlib.pyplot as plt

# --- AYARLAR ---
OUTPUT_DIR = "qwen-lora-deep"  # Analiz edilecek klasör

def find_best_model():
    print(f"'{OUTPUT_DIR}' klasöründeki eğitim geçmişi aranıyor...\n")
    
    state_file = None
    
    # Önce ana klasöre bak
    potential_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
    if os.path.exists(potential_path):
        state_file = potential_path
    else:
      
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
        if checkpoints:
            # Numaraya göre sırala (checkpoint-100, checkpoint-200...)
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            last_checkpoint = checkpoints[-1]
            state_file = os.path.join(OUTPUT_DIR, last_checkpoint, "trainer_state.json")

    if not state_file or not os.path.exists(state_file):
        print("HATA: 'trainer_state.json' dosyası bulunamadı. Eğitim çok erken kesilmiş olabilir.")
        return

    print(f"Log dosyası bulundu: {state_file}")

    
    with open(state_file, 'r') as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    
    
    eval_steps = []
    eval_losses = []
    
    print(f"\n{'ADIM (STEP)':<10} | {'EVAL LOSS':<10}")
    print("-" * 25)
    
    best_loss = float('inf')
    best_step = -1
    
    for log in log_history:
        if "eval_loss" in log:
            step = log["step"]
            loss = log["eval_loss"]
            eval_steps.append(step)
            eval_losses.append(loss)
            
            print(f"{step:<10} | {loss:.4f}")
            
            if loss < best_loss:
                best_loss = loss
                best_step = step

   
    if best_step != -1:
        print("-" * 25)
        print(f"\n🏆 EN İYİ MODEL (Best Checkpoint): checkpoint-{best_step}")
        print(f"📉 En Düşük Hata (Loss): {best_loss:.4f}")
        print(f"\nBu checkpoint'i kullanmak için yolunuz: {OUTPUT_DIR}/checkpoint-{best_step}")
    else:
        print("\nHenüz hiç değerlendirme (eval) yapılmamış.")

if __name__ == "__main__":
    find_best_model()