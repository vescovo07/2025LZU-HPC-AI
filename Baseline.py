# !pip install -q datasets nvidia-ml-py psutil accelerate

import time
import json
import re
import threading
import psutil
import pynvml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class ResourceMonitor:
    def __init__(self, output_file="evaluation_metrics.json"):
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
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append((i, handle))
                self.metrics["peak_gpu_ram_gb"][f"gpu_{i}"] = 0.0
                self.metrics["peak_gpu_util"][f"gpu_{i}"] = 0.0
            print(f"There are {device_count} GPU, monitoring ready.")
        except Exception as e:
            print(f"Failed to initialize GPU{e}")

    def _monitor_loop(self):
        psutil.cpu_percent(interval=None)
        while self.running:
            # CPU & RAM
            cpu = psutil.cpu_percent(interval=None)
            if cpu > self.metrics["peak_cpu_percent"]:
                self.metrics["peak_cpu_percent"] = cpu
            
            ram = psutil.virtual_memory().used / (1024**3)
            if ram > self.metrics["peak_ram_gb"]:
                self.metrics["peak_ram_gb"] = round(ram, 2)
            
            # GPU
            for idx, handle in self.gpu_handles:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    mem_gb = mem_info.used / (1024**3)
                    if mem_gb > self.metrics["peak_gpu_ram_gb"][f"gpu_{idx}"]:
                        self.metrics["peak_gpu_ram_gb"][f"gpu_{idx}"] = round(mem_gb, 2)
                        
                    if util_info.gpu > self.metrics["peak_gpu_util"][f"gpu_{idx}"]:
                        self.metrics["peak_gpu_util"][f"gpu_{idx}"] = util_info.gpu
                except:
                    pass
            
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.gpu_handles:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

def extract_answer_number(text):
    if "####" in text:
        target = text.split("####")[-1].strip()
        return target.replace(",", "")
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

def is_correct(model_output, ground_truth):
    pred = extract_answer_number(model_output)
    truth = extract_answer_number(ground_truth)
    
    if pred is None or truth is None:
        return False
    
    try:
        return float(pred) == float(truth)
    except ValueError:
        return str(pred) == str(truth)

def run_evaluation():
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    NUM_SAMPLES = 500
    BATCH_SIZE = 1
    
    monitor = ResourceMonitor()
    monitor.start()
    
    try:
        print(f"Loading {NUM_SAMPLES} datas of gsm8k datasets...")
        dataset = load_dataset("gsm8k", "main", split="test")
        dataset = dataset.select(range(NUM_SAMPLES))
        
        print("Loading model Qwen2.5-3B...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        correct_count = 0
        total_tokens_generated = 0
        
        print(f"Session begin (Total: {NUM_SAMPLES}, Batch Size: {BATCH_SIZE})...")
        
        start_time_global = time.time()
        
        for i, item in enumerate(dataset):
            question = item['question']
            ground_truth = item['answer']

            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Please solve the problem step by step and put the final answer after '####'."},
                {"role": "user", "content": question}
            ]
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
            
            input_len = model_inputs.input_ids.shape[1]
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False
                )
            
            output_len = generated_ids.shape[1] - input_len
            total_tokens_generated += output_len
            
            response_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
            
            if is_correct(response_text, ground_truth):
                correct_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{NUM_SAMPLES} samples...")

        end_time_global = time.time()
        
        total_time = end_time_global - start_time_global
        throughput_samples = NUM_SAMPLES / total_time
        throughput_tokens = total_tokens_generated / total_time
        accuracy = correct_count / NUM_SAMPLES
        
        monitor.stop()

        final_metrics = monitor.metrics
        final_metrics.update({
            "model_name": MODEL_ID,
            "dataset": "gsm8k",
            "num_samples": NUM_SAMPLES,
            "batch_size": BATCH_SIZE,
            "total_time_seconds": round(total_time, 2),
            "inference_metrics": {
                "throughput_samples_per_sec": round(throughput_samples, 2),
                "throughput_tokens_per_sec": round(throughput_tokens, 2),
                "accuracy": round(accuracy, 4),                             
                "correct_samples": correct_count
            }
        })
        
        output_filename = "inference_benchmark_gsm8k.json"
        with open(output_filename, "w") as f:
            json.dump(final_metrics, f, indent=4)
            
        print("\n" + "="*40)
        print("Finished!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Sample throughput: {throughput_samples:.2f} samples/s")
        print(f"Token throughput: {throughput_tokens:.2f} tokens/s")
        print(f"Accurancy {accuracy:.2%}")
        print("="*40)
        print("Peaks:")
        print(json.dumps(monitor.metrics, indent=4))
        
    except Exception as e:
        monitor.stop()
        print(f"Runtime Error! {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()