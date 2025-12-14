# !pip install -U -q transformers accelerate datasets nvidia-ml-py psutil pandas

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
    target_file = None
    
    specific_path = "/kaggle/input/grade-school-math-8k-q-a/main_test.csv"
    if os.path.exists(specific_path):
        target_file = specific_path
    else:
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if "test" in file.lower() and file.endswith(".csv"):
                    target_file = os.path.join(root, file)
                    break
            if target_file: break

    if not target_file:
        raise FileNotFoundError("Cannot find gsm8k CSV file in /kaggle/input")

    print(f"Found: {target_file}")
    df = pd.read_csv(target_file)
    df.columns = [c.lower() for c in df.columns]
    
    dataset = []
    for _, row in df.iterrows():
        q = row.get('question') or row.get('q') or row.get('problem')
        a = row.get('answer') or row.get('a')
        if q and a:
            dataset.append({'question': str(q), 'answer': str(a)})
            
    return dataset[:num_samples]

class ResourceMonitor:
    def __init__(self, output_file="optimized_metrics.json"):
        self.running = False
        self.thread = None
        self.metrics = {"peak_gpu_ram_gb": 0.0}
        try: pynvml.nvmlInit()
        except: pass

    def _monitor_loop(self):
        while self.running:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.metrics["peak_gpu_ram_gb"] = max(self.metrics["peak_gpu_ram_gb"], mem.used / (1024**3))
            except: pass
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()

def is_correct(pred, truth):
    def extract(text):
        if "####" in text: return text.split("####")[-1].strip().replace(",", "")
        nums = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
        return nums[-1] if nums else None
    p, t = extract(pred), extract(truth)
    if p is None or t is None: return False
    try: return float(p) == float(t)
    except: return str(p) == str(t)

def run_static_cache_inference():
    BASELINE_THROUGHPUT = 17.64
    NUM_SAMPLES = 500
    BATCH_SIZE = 1
    
    def find_model_path(keyword):
        for root, _, files in os.walk("/kaggle/input"):
            if keyword in root and "config.json" in files: return root
        return None

    local_path = find_model_path("3b")
    TARGET_MODEL = local_path if local_path else "Qwen/Qwen2.5-3B-Instruct"
    print(f"Model path: {TARGET_MODEL}")

    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        print(f"\nFP16 + Static KV Cache + SDPA Attention (BS={BATCH_SIZE})")

        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        
        model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa", 
            trust_remote_code=True
        )

        model.generation_config.cache_implementation = "static"

        dataset = load_gsm8k_csv(NUM_SAMPLES)
        
        print("Preheating...")
        dummy_input = tokenizer("Warmup run", return_tensors="pt").to("cuda:0")
        model.generate(**dummy_input, max_new_tokens=20)
        print("Preheat done.\n")

        correct_count = 0
        total_tokens = 0
        start_time = time.time()
        
        pbar = tqdm(dataset, total=NUM_SAMPLES, unit="sample")
        
        for i, item in enumerate(pbar):
            msgs = [
                {"role": "system", "content": "You are a helpful math assistant. Please solve the problem step by step and put the final answer after '####'."},
                {"role": "user", "content": item['question']}
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
            input_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            new_tokens = outputs[0][input_len:]
            gen_len = len(new_tokens)
            total_tokens += gen_len
            
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if is_correct(response, item['answer']):
                correct_count += 1
            
            elapsed = time.time() - start_time
            spd = total_tokens / elapsed if elapsed > 0 else 0
            acc = correct_count / (i + 1)
            pbar.set_postfix({"Acc": f"{acc:.1%}", "Spd": f"{spd:.1f}t/s"})

        total_time = time.time() - start_time
        monitor.stop()
        
        throughput = total_tokens / total_time
        speedup = throughput / BASELINE_THROUGHPUT
        acc_final = correct_count / NUM_SAMPLES
        
        print("\n" + "="*40)
        print(f"Static Cache optimization done (BS=1)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Token throughput: {throughput:.2f} t/s")
        print(f"Acceleration ratio: {speedup:.2f}x (Baseline {BASELINE_THROUGHPUT} t/s)")
        print(f"Accurancy: {acc_final:.2%}")
        print("="*40)
        
        with open("optimized_benchmark_static.json", "w") as f:
            json.dump({
                "method": "FP16 + Static Cache + SDPA",
                "throughput": throughput,
                "speedup": speedup,
                "accuracy": acc_final,
                "gpu_mem": monitor.metrics["peak_gpu_ram_gb"]
            }, f, indent=4)

    except Exception as e:
        monitor.stop()
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_static_cache_inference()