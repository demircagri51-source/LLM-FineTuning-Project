import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
import sys

# --- AYARLAR ---
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEEP_MODEL_PATH = "qwen-lora-deep/final_model"
DIVERSE_MODEL_PATH = "qwen-lora-diverse/final_model"

def get_single_response(model_path, model_name, soru):
    """Tek bir modelden cevap alır ve belleği temizler"""
    print(f"   -> {model_name} yükleniyor...")
    
    # Base Model Yükle
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA Adaptörünü Yükle
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception as e:
        return f"HATA: Model yüklenemedi. ({e})"

    # Cevap Üret
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. Provide only the solution code."},
        {"role": "user", "content": soru}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        do_sample=False, 
        temperature=0.0
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Temizlik (VRAM şişmemesi için kritik)
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return response

def main_loop():
    print(f"\n{'='*60}")
    print(" MODEL KARŞILAŞTIRMA ARACI (DEEP vs DIVERSE)")
    print(f"{'='*60}")
    print("Çıkmak için 'q' yazın.\n")

    while True:
        try:
            soru = input("\nSORUNUZ: ")
            
            if soru.lower() in ['q', 'exit', 'cikis']:
                print("Çıkış yapılıyor...")
                break
            
            if not soru.strip():
                continue

            print("\nAnaliz Başlıyor (Bu işlem modelleri yüklediği için biraz sürebilir)...")
            
            # 1. DEEP Model Cevabı
            deep_cevap = get_single_response(DEEP_MODEL_PATH, "DEEP MODEL", soru)
            
            # 2. DIVERSE Model Cevabı
            diverse_cevap = get_single_response(DIVERSE_MODEL_PATH, "DIVERSE MODEL", soru)

            # Sonuçları Yazdır
            print(f"\n{'-'*30}")
            print("🟦 DEEP MODEL CEVABI:")
            print(f"{'-'*30}\n{deep_cevap}")
            
            print(f"\n{'-'*30}")
            print("🟧 DIVERSE MODEL CEVABI:")
            print(f"{'-'*30}\n{diverse_cevap}")
            print(f"{'='*60}")

        except KeyboardInterrupt:
            print("\nProgram durduruldu.")
            break

if __name__ == "__main__":
    main_loop()