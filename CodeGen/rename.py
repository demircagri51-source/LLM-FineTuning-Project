import os
import shutil

# Klasör yolları
BASE_DIR = "models"
MODEL_DIRS = ["deep_instruction", "diverse_instruction"]

# Epoch hesabı (Yaklaşık: Her 563 adım 1 epoch ediyordu)
STEPS_PER_EPOCH = 563

def rename_folders():
    print("--- Checkpoint İsim Düzeltme Başlıyor ---")
    
    for model_dir in MODEL_DIRS:
        path = os.path.join(BASE_DIR, model_dir)
        if not os.path.exists(path):
            print(f"UYARI: {path} bulunamadı! Klasör ismini doğru yaptınız mı?")
            continue
            
        print(f"\n📂 {model_dir} taranıyor...")
        
        for folder_name in os.listdir(path):
            # Sadece 'checkpoint-' ile başlayanları al, ama zaten düzeltilmişleri alma
            if folder_name.startswith("checkpoint-") and "step" not in folder_name:
                try:
                    # 'checkpoint-200' -> step=200
                    step = int(folder_name.split("-")[-1])
                    
                    # Epoch hesapla
                    epoch = (step // STEPS_PER_EPOCH) + 1
                    
                    # Yeni isim: checkpoint-step-200-epoch-1
                    new_name = f"checkpoint-step-{step}-epoch-{epoch}"
                    
                    old_path = os.path.join(path, folder_name)
                    new_path = os.path.join(path, new_name)
                    
                    os.rename(old_path, new_path)
                    print(f"   ✅ Değişti: {folder_name} -> {new_name}")
                    
                except Exception as e:
                    print(f"   ❌ Hata ({folder_name}): {e}")

    print("\n--- İşlem Tamamlandı ---")

if __name__ == "__main__":
    rename_folders()