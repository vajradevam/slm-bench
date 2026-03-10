import os
import json
import re
import subprocess
import requests
import psutil
import platform
import cpuinfo
import sys
import time
import signal
from tqdm import tqdm
from statistics import mean
from rouge_score import rouge_scorer

# ==========================================
# CONFIGURATION
# ==========================================

CONFIG = {
    "runs_per_prompt": 1,
    "max_tokens": 128,
    "threads": psutil.cpu_count(logical=False) or psutil.cpu_count(),
    "ctx": 2048,
    "temperature": 0.0,
    "seed": 42,
    "timeout": 120,
    "server_port": 8080,
    "host": "127.0.0.1"
}

# ==========================================
# DATA & PROMPTS
# ==========================================

PROMPTS = {
    "R1": "If a train travels at 60 km/h for 3 hours how far does it travel?",
    "R2": "A shop sells apples for $2 each. If someone buys 7 apples how much do they pay?",
    "R3": "Rectangle length 8 width 5. What is the area?",
    "R4": "John has 12 marbles gives 5 away then buys 8 more. How many now?",
    "R5": "Car travels 150 km with 10 liters fuel. Km per liter?",
    "R6": "3 workers build wall in 6 hours. Time for 6 workers?",
    
    "C1": "Write python function factorial.",
    "C2": "Write python function palindrome check.",
    "C3": "Write python function largest number in list.",
    "C4": "Write python function count vowels.",
    "C5": "Write python function remove duplicates from list.",
    "C6": "Write python function fibonacci sequence.",
    
    "S1": "Summarize: Artificial intelligence simulates human intelligence in machines. Machine learning learns from data.",
    "S2": "Summarize: Climate change refers to warming due to greenhouse gases.",
    "S3": "Summarize: Internet enables global communication and information access.",
    "S4": "Summarize: Open source software allows collaborative code development.",
    
    "I1": "List five machine learning uses in healthcare.",
    "I2": "Explain overfitting in machine learning.",
    "I3": "List advantages of renewable energy.",
    "I4": "Explain database index."
}

GROUND_TRUTH = {
    "R1": {"type": "exact", "answer": "180"},
    "R2": {"type": "exact", "answer": "14"},
    "R3": {"type": "exact", "answer": "40"},
    "R4": {"type": "exact", "answer": "15"},
    "R5": {"type": "exact", "answer": "15"},
    "R6": {"type": "exact", "answer": "3"},
    
    "S1": {"type": "summary", "reference": "Artificial intelligence simulates intelligence and machine learning learns from data."},
    "S2": {"type": "summary", "reference": "Climate change is warming caused by greenhouse gases."},
    "S3": {"type": "summary", "reference": "Internet enables global communication and information sharing."},
    "S4": {"type": "summary", "reference": "Open source allows collaborative software development."},
    
    "I1": {"type": "keywords", "keywords": ["diagnosis", "imaging", "drug", "monitoring"]},
    "I2": {"type": "keywords", "keywords": ["overfitting", "training", "generalization"]},
    "I3": {"type": "keywords", "keywords": ["clean", "renewable", "sustainable"]},
    "I4": {"type": "keywords", "keywords": ["index", "database", "query"]}
}

# ==========================================
# UTILITIES
# ==========================================

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def system_info():
    try:
        cpu_brand = cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        cpu_brand = platform.processor()
        
    return {
        "cpu": cpu_brand,
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": platform.system(),
        "python_version": platform.python_version()
    }

def clean_output(text, prompt_text):
    if not text: return ""
    if text.startswith(prompt_text):
        text = text[len(prompt_text):]
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()

# ==========================================
# EVALUATION LOGIC
# ==========================================

def eval_reasoning(output, answer):
    if not output: return 0
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    for n in numbers:
        try:
            if float(n) == float(answer): return 1
        except ValueError: continue
    return 0

def eval_summary(output, reference):
    if not output: return 0.0
    score = scorer.score(reference, output)
    return round(score["rougeL"].fmeasure, 4)

def eval_keywords(output, keywords):
    if not output: return 0.0
    output_lower = output.lower()
    hits = sum(1 for k in keywords if k.lower() in output_lower)
    return round(hits / len(keywords), 4)

# ==========================================
# SERVER MANAGEMENT
# ==========================================

def wait_for_server(url, timeout=60):
    start_time = time.time()
    health_url = f"{url}/health"
    print(f"Waiting for server at {url}...")
    
    while time.time() - start_time < timeout:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
            
    return False

def run_llama_bench(model_path):
    cmd = [
        "llama-bench",
        "-m", model_path,
        "-t", str(CONFIG["threads"]),
        "-p", "512",
        "-n", "128",
        "-o", "json"
    ]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8')
        out = proc.stdout
    except Exception as e:
        print(f"Bench error: {e}")
        return 0.0, 0.0

    prompt_tps = 0.0
    gen_tps = 0.0
    
    try:
        json_start = out.find('[')
        if json_start != -1:
            data = json.loads(out[json_start:])
            for item in data:
                tname = item.get("test_name", item.get("test", ""))
                avg_ts = item.get("avg_ts", 0)
                if "pp" in tname: prompt_tps = avg_ts
                elif "tg" in tname: gen_tps = avg_ts
    except:
        pass
        
    return prompt_tps, gen_tps

def query_server(url, prompt):
    payload = {
        "prompt": prompt,
        "n_predict": CONFIG["max_tokens"],
        "temperature": CONFIG["temperature"],
        "seed": CONFIG["seed"],
        "stop": ["User:"], 
        "cache_prompt": False 
    }
    
    try:
        r = requests.post(f"{url}/completion", json=payload, timeout=CONFIG["timeout"])
        if r.status_code == 200:
            return r.json().get("content", "")
        else:
            return f"ERROR: Server returned {r.status_code}"
    except Exception as e:
        return f"ERROR: Request failed {e}"

# ==========================================
# MAIN BENCHMARK LOGIC
# ==========================================

def download_model(url, path):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".part"
    downloaded = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    try:
        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0)) + downloaded
            mode = 'ab' if downloaded else 'wb'
            with open(temp_path, mode) as f:
                with tqdm(total=total, initial=downloaded, unit='B', unit_scale=True, desc=os.path.basename(path)) as pbar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        os.rename(temp_path, path)
    except Exception as e:
        print(f"Download failed: {e}")
        raise

def load_models():
    if not os.path.exists("models.json"): return []
    with open("models.json", "r") as f: return json.load(f).get("models", [])

def load_results():
    if not os.path.exists("results.json"): return {}
    try:
        with open("results.json", "r") as f: return json.load(f)
    except: return {}

def save_results(data):
    with open("results.json", "w") as f: json.dump(data, f, indent=2)

def benchmark_model(model_path, model_name):
    print(f"[1/3] Running Performance Benchmark (llama-bench)...")
    prompt_tps, gen_tps = run_llama_bench(model_path)
    
    print(f"[2/3] Starting llama-server...")
    server_url = f"http://{CONFIG['host']}:{CONFIG['server_port']}"
    
    cmd = [
        "llama-server",
        "-m", model_path,
        "--port", str(CONFIG["server_port"]),
        "--host", CONFIG["host"],
        "-c", str(CONFIG["ctx"]),
        "-t", str(CONFIG["threads"])
    ]
    
    server_proc = None
    try:
        server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if not wait_for_server(server_url):
            stderr_output = server_proc.stderr.read() if server_proc.stderr else "Unknown error"
            raise Exception(f"Server failed to start.\nServer Log: {stderr_output[:500]}")

        print(f"[3/3] Running Quality Benchmark (HTTP API)...")
        quality_metrics = {"reasoning": [], "summary": [], "instruction": []}
        detailed_results = {}
        
        for pid, prompt in tqdm(PROMPTS.items(), desc="Testing Prompts"):
            raw_output = query_server(server_url, prompt)
            clean_text = clean_output(raw_output, prompt)
            
            gt = GROUND_TRUTH.get(pid)
            score = 0.0
            score_type = "unknown"
            
            if gt:
                if gt["type"] == "exact":
                    score = eval_reasoning(clean_text, gt["answer"])
                    quality_metrics["reasoning"].append(score)
                    score_type = "accuracy"
                elif gt["type"] == "summary":
                    score = eval_summary(clean_text, gt["reference"])
                    quality_metrics["summary"].append(score)
                    score_type = "rougeL"
                elif gt["type"] == "keywords":
                    score = eval_keywords(clean_text, gt["keywords"])
                    quality_metrics["instruction"].append(score)
                    score_type = "recall"
                    
            detailed_results[pid] = {
                "output": clean_text[:250] + "..." if len(clean_text) > 250 else clean_text,
                "score": score,
                "type": score_type
            }
            
        avg_reasoning = mean(quality_metrics["reasoning"]) if quality_metrics["reasoning"] else 0
        avg_summary = mean(quality_metrics["summary"]) if quality_metrics["summary"] else 0
        avg_instruction = mean(quality_metrics["instruction"]) if quality_metrics["instruction"] else 0
        
        return {
            "performance": {
                "prompt_tps": round(prompt_tps, 2),
                "generation_tps": round(gen_tps, 2)
            },
            "quality": {
                "reasoning_accuracy": round(avg_reasoning, 4),
                "summary_rouge": round(avg_summary, 4),
                "instruction_recall": round(avg_instruction, 4)
            },
            "detailed_results": detailed_results
        }
        
    finally:
        if server_proc:
            print("Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

def main():
    print("--- SLM Laptop Benchmark Tool (Server Mode) ---")
    print("System Info:", json.dumps(system_info(), indent=2))
    
    models = load_models()
    results = load_results()
    
    for m in models:
        name = m.get("name")
        url = m.get("url")
        filename = m.get("file")
        
        if not name or not url or not filename: continue
        
        # CHECK FOR EXISTING RESULTS
        if name in results:
            print(f"Skipping {name} (already in results.json).")
            continue
            
        print(f"\n========================================")
        print(f"Model: {name}")
        print("========================================")
        
        path = os.path.join("models", filename)
        
        try:
            download_model(url, path)
            
            start_time = time.time()
            data = benchmark_model(path, name)
            duration = round(time.time() - start_time, 2)
            
            results[name] = {
                "meta": {
                    "params": m.get("params"),
                    "quant": m.get("quant"),
                    "url": url
                },
                "system": system_info(),
                "benchmark_duration_sec": duration,
                "results": data
            }
            save_results(results)
            
        except KeyboardInterrupt:
            print("Benchmark interrupted by user.")
            # Cleanup partially processed model on interrupt
            if os.path.exists(path):
                try: os.remove(path)
                except: pass
            break
            
        except Exception as e:
            print(f"Critical error: {e}")
            results[name] = {"error": str(e)}
            save_results(results)
        
        finally:
            # CLEANUP: Delete model file to save space
            if os.path.exists(path):
                print(f"Cleaning up model file: {path}")
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Error deleting file: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()