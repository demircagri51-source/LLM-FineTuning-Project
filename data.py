from datasets import load_dataset
import os


deep_file_path = os.path.join("data", "deep_solution_only.jsonl")
diverse_file_path = os.path.join("data", "diverse_solution_only.jsonl")

# 1. Datasetleri Lokal Dosyalardan Yükleme
# "json" formatını kullanıyoruz çünkü dosyalarınız .jsonl uzantılı
print(f"Yükleniyor: {deep_file_path}")
deep_dataset = load_dataset("json", data_files=deep_file_path, split="train")

print(f"Yükleniyor: {diverse_file_path}")
diverse_dataset = load_dataset("json", data_files=diverse_file_path, split="train")


print("\n--- Veri Kontrolü ---")
print("Sütun İsimleri:", deep_dataset.column_names)
print("Örnek Veri:", deep_dataset[0])



def format_data_code_only(example):
   
    soru = example.get('input') or example.get('instruction') or example.get('problem')
    cevap = example.get('solution') or example.get('code') or example.get('output')
    
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. Provide only the solution code."},
            {"role": "user", "content": soru},
            {"role": "assistant", "content": cevap}
        ]
    }

# Datasetleri dönüştürme
deep_dataset_formatted = deep_dataset.map(format_data_code_only)
diverse_dataset_formatted = diverse_dataset.map(format_data_code_only)

print("Veri hazırlığı tamamlandı!")