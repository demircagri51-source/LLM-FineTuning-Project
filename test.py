from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"   

print("Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model yüklendi ✓")

def ask(question):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


# Test soruları (PDF istediği için 3 örnek veriyoruz)
test_questions = [
    "Python'da listeyi nasıl sıralarım?",
    "Binary search algoritması nasıl çalışır?",
    "Fibonacci sayıları için recursive bir fonksiyon yaz."
]

print("\n--- MODEL TEST SONUÇLARI ---\n")

for q in test_questions:
    print(f"Soru: {q}")
    print("Cevap:")
    print(ask(q))
    print("\n" + "-"*50 + "\n")
