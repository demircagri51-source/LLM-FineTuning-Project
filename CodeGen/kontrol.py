import os

print("--- KLASÖR KONTROLÜ ---")
if os.path.exists("models"):
    print("✅ 'models' klasörü var.")
    for model in os.listdir("models"):
        model_path = os.path.join("models", model)
        if os.path.isdir(model_path):
            print(f"\n📂 Model: {model}")
            files = os.listdir(model_path)
            checkpoints = [f for f in files if "checkpoint" in f]
            
            if not checkpoints:
                print("   ❌ HATA: İçinde hiç checkpoint klasörü yok!")
                print(f"   (İçindekiler: {files[:5]}...)")
            else:
                print(f"   Bulunan Checkpoint Sayısı: {len(checkpoints)}")
                print(f"   Örnek İsim: {checkpoints[0]}")
                
                # İsim kontrolü
                if "step" in checkpoints[0] and "epoch" in checkpoints[0]:
                    print("   ✅ İsim formatı DOĞRU.")
                else:
                    print("   ❌ İsim formatı YANLIŞ! (İçinde 'step' ve 'epoch' geçmiyor)")
else:
    print("❌ 'models' klasörü BULUNAMADI!")