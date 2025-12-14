# !pip install -q accelerate datasets nvidia-ml-py psutil tqdm pandas

import time
import json
import re
import os
import threading
import psutil
import pynvml
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

def load_gsm8k_csv(num_samples=500):
    print("Loading gsm8k dataset...")
    target_path = "/kaggle/input/grade-school-math-8k-q-a/main_test.csv"
    
    if not os.path.exists(target_path):
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if "test" in file.lower() and file.endswith(".csv"):
                    target_path = os.path.join(root, file)
                    break
    
    if not target_path: raise FileNotFoundError("ERROR")
    
    df = pd.read_csv(target_path)
    df.columns = [c.lower() for c in df.columns]
    dataset = []
    for _, row in df.iterrows():
        q = row.get('question') or row.get('q')
        a = row.get('answer') or row.get('a')
        if q and a: dataset.append({'question': str(q), 'answer': str(a)})
    return dataset[:num_samples]

class ResourceMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.metrics = {"peak_gpu_ram": 0.0}
        try: pynvml.nvmlInit()
        except: pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

    def _loop(self):
        while self.running:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.metrics["peak_gpu_ram"] = max(self.metrics["peak_gpu_ram"], mem.used / 1024**3)
            except: pass
            time.sleep(1.0)

def is_correct(pred, truth):
    def extract(text):
        if "####" in text: return text.split("####")[-1].strip().replace(",", "")
        nums = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
        return nums[-1] if nums else None
    p, t = extract(pred), extract(truth)
    if p is None or t is None: return False
    try: return float(p) == float(t)
    except: return str(p) == str(t)

def run_benchmark():
    BASELINE_THROUGHPUT = 17.64
    NUM_SAMPLES = 500
    
    def find_model_path(keyword):
        for root, _, files in os.walk("/kaggle/input"):
            if keyword in root and "config.json" in files: return root
        return None
    local_path = find_model_path("3b")
    MODEL_ID = local_path if local_path else "Qwen/Qwen2.5-3B-Instruct"

    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        print("\nFP16 + Torch Compile (JIT) + Dynamic Cache")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )

        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        dataset = load_gsm8k_csv(NUM_SAMPLES)
        
        print("Preheating...")
        warmup_prompts = ["1+1=", "Time complexity of quicksort is"]
        for p in warmup_prompts:
            inputs = tokenizer([p], return_tensors="pt").to("cuda:0")
            model.generate(**inputs, max_new_tokens=20)
        print("Preheat done. Starting session...")

        correct = 0
        total_tokens = 0
        start_time = time.time()
        
        pbar = tqdm(dataset, total=NUM_SAMPLES)
        
        for i, item in enumerate(pbar):
            msgs = [{"role": "system", "content": "You are a helpful math assistant. Please solve the problem step by step and put the final answer after '####'."}, {"role": "user", "content": item['question']}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
            in_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            new_tokens = outputs[0][in_len:]
            gen_len = len(new_tokens)
            total_tokens += gen_len
            
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if is_correct(pred, item['answer']): correct += 1
                
            elapsed = time.time() - start_time
            speed = total_tokens / elapsed if elapsed > 0 else 0
            acc = correct / (i + 1)
            pbar.set_postfix({"Acc": f"{acc:.1%}", "Spd": f"{speed:.1f}t/s"})

        total_time = time.time() - start_time
        monitor.stop()
        
        throughput = total_tokens / total_time
        
        print("\n" + "="*40)
        print(f"Final results (FP16 + Compile)")
        print(f"Throughput: {throughput:.2f} t/s")
        print(f"Increasement: {throughput/BASELINE_THROUGHPUT:.2f}x")
        print(f"Accurancy: {correct/NUM_SAMPLES:.2%}")
        print("="*40)
        
        with open("optimized_benchmark_final.json", "w") as f:
            json.dump({"throughput": throughput, "accuracy": correct/NUM_SAMPLES}, f)

    except Exception as e:
        monitor.stop()
        print(f"ERROR {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()