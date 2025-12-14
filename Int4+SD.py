# !pip install -U transformers accelerate
# !pip install -q datasets nvidia-ml-py psutil accelerate
# !pip install -q bitsandbytes accelerate datasets nvidia-ml-py psutil

import time
import json
import re
import threading
import psutil
import pynvml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ResourceMonitor:
    def __init__(self, output_file="optimized_metrics.json"):
        self.output_file = output_file
        self.running = False
        self.thread = None
        self.metrics = {
            "peak_cpu_percent": 0.0,
            "peak_ram_gb": 0.0,
            "peak_gpu_ram_gb": {},
            "peak_gpu_util": {}
        }
        self.gpu_handles = []
        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                self.gpu_handles.append((i, pynvml.nvmlDeviceGetHandleByIndex(i)))
                self.metrics["peak_gpu_ram_gb"][f"gpu_{i}"] = 0.0
                self.metrics["peak_gpu_util"][f"gpu_{i}"] = 0.0
        except Exception:
            pass

    def _monitor_loop(self):
        psutil.cpu_percent(interval=None)
        while self.running:
            # CPU
            self.metrics["peak_cpu_percent"] = max(self.metrics["peak_cpu_percent"], psutil.cpu_percent(interval=None))
            # RAM
            self.metrics["peak_ram_gb"] = max(self.metrics["peak_ram_gb"], psutil.virtual_memory().used / (1024**3))
            # GPU
            for idx, handle in self.gpu_handles:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    key = f"gpu_{idx}"
                    self.metrics["peak_gpu_ram_gb"][key] = max(self.metrics["peak_gpu_ram_gb"][key], mem.used / (1024**3))
                    self.metrics["peak_gpu_util"][key] = max(self.metrics["peak_gpu_util"][key], util.gpu)
                except:
                    pass
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.gpu_handles: 
            try: pynvml.nvmlShutdown() 
            except: pass

def extract_answer(text):
    if "####" in text: return text.split("####")[-1].strip().replace(",", "")
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    return numbers[-1] if numbers else None

def is_correct(pred, truth):
    p, t = extract_answer(pred), extract_answer(truth)
    if p is None or t is None: return False
    try: return float(p) == float(t)
    except: return str(p) == str(t)

def run_optimized_inference():
    # Baseline Data (FP16 without optimizations)
    BASELINE_THROUGHPUT = 17.64
    BASELINE_ACCURACY = 0.7020

    TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    NUM_SAMPLES = 500
    
    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        print("Optimition: INT4 + Speculative Decoding...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

        print(f"Loading target model: {TARGET_MODEL} (INT4)...")
        model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"Loading draft model: {DRAFT_MODEL} (FP16)...")
        assistant_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"Loading GSM8K dataset ({NUM_SAMPLES} samples)...")
        dataset = load_dataset("gsm8k", "main", split="test").select(range(NUM_SAMPLES))
        
        correct_count = 0
        total_tokens = 0
        start_time = time.time()
        
        print(f"Starting (Batch Size=1, Total={NUM_SAMPLES})...")
        
        for i, item in enumerate(dataset):
            question = item['question']
            ground_truth = item['answer']
            
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Please solve the problem step by step and put the final answer after '####'."},
                {"role": "user", "content": question}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            input_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    assistant_model=assistant_model,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            new_tokens = outputs[0][input_len:]
            total_tokens += len(new_tokens)
            
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if is_correct(response_text, ground_truth):
                correct_count += 1
            
            if (i+1) % 50 == 0:
                print(f"process: {i+1}/{NUM_SAMPLES}")

        total_time = time.time() - start_time
        monitor.stop()
        
        throughput_samples = NUM_SAMPLES / total_time
        throughput_tokens = total_tokens / total_time
        accuracy = correct_count / NUM_SAMPLES
        
        speedup_ratio = throughput_tokens / BASELINE_THROUGHPUT
        acc_diff = accuracy - BASELINE_ACCURACY
        
        metrics = {
            "method": "Int4 Quantization + Speculative Decoding",
            "model": TARGET_MODEL,
            "assistant_model": DRAFT_MODEL,
            "time_seconds": round(total_time, 2),
            "accuracy": round(accuracy, 4),
            "accuracy_change": round(acc_diff, 4),
            "throughput_tokens_per_sec": round(throughput_tokens, 2),
            "throughput_samples_per_sec": round(throughput_samples, 2),
            "speedup_ratio": round(speedup_ratio, 2),
            "hardware_peak": monitor.metrics
        }
        
        with open("optimized_benchmark_hf.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        print("\n" + "="*40)
        print("Optimization done (Transformers original benchmark)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Token throughput: {throughput_tokens:.2f} tokens/s (baseline: {BASELINE_THROUGHPUT})")
        print(f"Speedup: {speedup_ratio:.2f}x")
        print(f"Accuracy: {accuracy:.2%} (baseline: {BASELINE_ACCURACY:.2%}, change: {acc_diff:+.2%})")
        print("="*40)
        print("Peak:", json.dumps(monitor.metrics, indent=4))

    except Exception as e:
        monitor.stop()
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimized_inference()