import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import os
import gc


BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_DIR = "data"  


def format_data_code_only(example):
    """
    Qwen Chat formatına uygun veri hazırlar.
    Sadece 'solution' alanını (temiz kod) hedef olarak kullanır.
    """
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. Provide only the solution code."},
            {"role": "user", "content": example['input']},
            {"role": "assistant", "content": example['solution']}
        ]
    }


def run_training(file_name, output_dir_name):
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"\n{'='*40}")
    print(f"BAŞLIYOR: {output_dir_name} (Veri: {file_name})")
    print(f"{'='*40}")

    
    dataset = load_dataset("json", data_files=file_path, split="train")
    
    # Test için %10 ayırdııgm yer ezber yapmaması için
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train'].map(format_data_code_only)
    eval_dataset = dataset['test'].map(format_data_code_only)

    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 

  
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    
    peft_config = LoraConfig(
        r=32,                    
        lora_alpha=64,           
        lora_dropout=0.1,
        target_modules="all-linear", 
        bias="none",
        task_type="CAUSAL_LM",
    )

   # out of cuda hatası aldııgm ıçn burdaki bazı şeyleri degiştirdim
   
    training_args = TrainingArguments(
        output_dir=output_dir_name,
        per_device_train_batch_size=1, # 2 den bire çektim 
        gradient_accumulation_steps=16,  # 4 den 8 e çektim
        
        
        per_device_eval_batch_size=1,   # Eval sırasında da batch size 1 olsun
        eval_accumulation_steps=1,     
        
        
        learning_rate=5e-5,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=200, # 200 adımda bir checkpoint kaydet, arttırık
        eval_strategy="steps",
        eval_steps=200, 
        logging_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=0, 
        gradient_checkpointing=True,
    )
    

  
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,  
        args=training_args,
    )
    

    # Eğitimi Başlat
    trainer.train()

    # Final Modeli Kaydet
    final_path = os.path.join(output_dir_name, "final_model")
    trainer.save_model(final_path)
    print(f"Eğitim tamamlandı! Model kaydedildi: {final_path}")

    # Bellek Temizliği (Sıradaki eğitim için)
    del model
    del trainer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# --- ANA ÇALIŞMA ---
if __name__ == "__main__":
    # 1. Eğitim: DEEP Dataset
    # run_training("deep_solution_only.jsonl", "qwen-lora-deep")

    # 2. Eğitim: DIVERSE Dataset
    run_training("diverse_solution_only.jsonl", "qwen-lora-diverse")