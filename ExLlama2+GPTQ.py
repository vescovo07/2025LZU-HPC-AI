# !pip install -q exllamav2 huggingface_hub pandas psutil tqdm transformers

import sys
import os
import time
import json
import re
import pandas as pd
import psutil
import torch
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

def load_gsm8k_csv(num_samples=500):
    print("Loading GSM8K dataset...")
    target_path = "/kaggle/input/grade-school-math-8k-q-a/main_test.csv"
    if not os.path.exists(target_path):
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if "test" in file.lower() and file.endswith(".csv"):
                    target_path = os.path.join(root, file)
                    break
            if target_path and os.path.exists(target_path): break
            
    if not target_path: raise FileNotFoundError("Cannot find GSM8K test CSV file!")
    
    df = pd.read_csv(target_path)
    df.columns = [c.lower() for c in df.columns]
    dataset = []
    for _, row in df.iterrows():
        q = row.get('question') or row.get('q')
        a = row.get('answer') or row.get('a')
        if q and a: dataset.append({'question': str(q), 'answer': str(a)})
    return dataset[:num_samples]

def is_correct(pred, truth):
    def extract(text):
        if "####" in text: return text.split("####")[-1].strip().replace(",", "")
        nums = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
        return nums[-1] if nums else None
    p, t = extract(pred), extract(truth)
    if p is None or t is None: return False
    try: return float(p) == float(t)
    except: return str(p) == str(t)

def run_exllamav2_inference():
    BASELINE_THROUGHPUT = 17.64
    NUM_SAMPLES = 500
    
    MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
    print(f"Downloading model: {MODEL_REPO} ...")
    model_dir = snapshot_download(repo_id=MODEL_REPO)
    
    hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("Initializing ExLlamaV2 model...")
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    config.max_seq_len = 2048 
    
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, batch_size=1) 
    model.load_autosplit(cache)
    
    ex_tokenizer = ExLlamaV2Tokenizer(config)
    
    eos_id = ex_tokenizer.single_id("<|im_end|>")
    print(f"Finding EOS Token ID: {eos_id}")

    generator = ExLlamaV2BaseGenerator(model, cache, ex_tokenizer)
    
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.0
    settings.top_k = 1
    settings.top_p = 1.0
    settings.eos_token_id = eos_id 

    dataset = load_gsm8k_csv(NUM_SAMPLES)
    
    correct_count = 0
    total_tokens = 0
    start_time = time.time()
    
    print(f"Starting inference (ExLlamaV2, Total={NUM_SAMPLES})...")
    
    pbar = tqdm(dataset, total=NUM_SAMPLES)
    
    for i, item in enumerate(pbar):
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Please solve the problem step by step and put the final answer after '####'."},
            {"role": "user", "content": item['question']}
        ]
        prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = generator.generate_simple(
            prompt, 
            settings, 
            num_tokens=512, 
            seed=42
        )
        
        response_text = output[len(prompt):]
        
        if "<|im_end|>" in response_text:
            response_text = response_text.split("<|im_end|>")[0]
        
        ids = ex_tokenizer.encode(response_text)
        token_count = ids.shape[-1]
        total_tokens += token_count
        
        if is_correct(response_text, item['answer']):
            correct_count += 1
            
        elapsed = time.time() - start_time
        spd = total_tokens / elapsed if elapsed > 0 else 0
        acc = correct_count / (i + 1)
        pbar.set_postfix({"Acc": f"{acc:.1%}", "Spd": f"{spd:.1f}t/s"})

    total_time = time.time() - start_time
    throughput = total_tokens / total_time
    
    print("\n" + "="*40)
    print(f"Final result: (ExLlamaV2+GPTQ)")
    print(f"Token throughput: {throughput:.2f} t/s")
    print(f"Increasement: {throughput/BASELINE_THROUGHPUT:.2f}x")
    print(f"Accuracy: {correct_count/NUM_SAMPLES:.2%}")
    print("="*40)
    
    with open("optimized_benchmark_exllama_final.json", "w") as f:
        json.dump({"throughput": throughput, "accuracy": correct_count/NUM_SAMPLES}, f)

if __name__ == "__main__":
    run_exllamav2_inference()