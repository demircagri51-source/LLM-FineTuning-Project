import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# --- AYARLAR ---
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def chat_with_base_model():
    print(f"\n{'='*50}")
    print(f"BASE MODEL YÜKLENİYOR... (Lütfen Bekleyin)")
    print(f"{'='*50}")

    # 1. Model ve Tokenizer'ı Yükle (Sadece 1 kere)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16, 
            device_map="auto",         
            trust_remote_code=True
        )
    except Exception as e:
        print(f"HATA: Model yüklenirken sorun oluştu: {e}")
        return

    print("\n>>> MODEL HAZIR! Sorularınızı yazabilirsiniz.")
    print(">>> Çıkmak için 'q' veya 'exit' yazıp Enter'a basın.\n")

    # 2. Sonsuz Döngü (Soru-Cevap)
    while True:
        try:
            # Kullanıcıdan Girdi Al
            soru = input("\nSORUNUZ: ")
            
            # Çıkış Kontrolü
            if soru.lower() in ['q', 'exit', 'cikis', 'çıkış']:
                print("Programdan çıkılıyor...")
                break
            
            if not soru.strip():
                continue

            messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": soru}
            ]

            # Prompt Hazırlama
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            print("Thinking... (Cevap üretiliyor)")

            # Cevap Üretme
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,       
                temperature=0.7       
            )

            # Sadece cevabı ayıklama
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"\n--- BASE MODEL CEVABI ---\n{response}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nİşlem durduruldu.")
            break
        except Exception as e:
            print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    chat_with_base_model()