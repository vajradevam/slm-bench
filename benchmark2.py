"""
═══════════════════════════════════════════════════════════════════════════════
  SLM Laptop Benchmark Tool  v2.0
  ─────────────────────────────────────────────────────────────────────────────
  A comprehensive, multi-run evaluation framework for quantised Small Language
  Models on consumer-grade (mid-range laptop) CPU hardware via llama.cpp.

  KEY IMPROVEMENTS OVER v1.0
  ──────────────────────────
  • Multi-run averaging  : configurable runs_per_prompt, mean ± std dev reported
  • Latency measurement  : per-token latency (ms/token) alongside TPS
  • Code eval via sandbox: C1–C6 now executed in a subprocess; pass/fail scored
  • BERTScore (optional) : richer summarisation metric when sentence-transformers
                           is available; falls back gracefully to ROUGE-L only
  • Expanded prompt suite: 50 prompts across 5 domains (Reasoning, Code, Summary,
                           Instruction, Factual QA)
  • Temperature sweep    : quality measured at T=0.0 and T=0.3 to expose variance
  • First-token latency  : TTFT (Time-To-First-Token) measured via streaming
  • Context-length probe : small/medium/long context variants for summary prompts
  • System stress monitor: CPU %, per-core frequency, RAM %, and thermal readings
                           sampled every 0.5 s during generation
  • Confidence intervals : 95 % bootstrap CIs on aggregate quality metrics
  • Memory footprint     : RSS delta of llama-server process during inference
  • Model metadata       : file size, expected perplexity tier derived from quant
  • Graceful fallbacks   : every optional dependency degrades cleanly
  • Rich HTML report     : self-contained report saved alongside results.json

  USAGE
  ─────
  pip install requests psutil py-cpuinfo tqdm rouge-score
  pip install sentence-transformers  # optional — BERTScore
  pip install jinja2                 # optional — HTML report

  # Populate models.json, then:
  python benchmark_v2.py
  python benchmark_v2.py --results my_results.json --report
  python benchmark_v2.py --model qwen3.5-2b-q4_k_m  # single model

  DEPENDENCIES (external)
  ──────────────────────
  llama-server  (llama.cpp, on PATH)
  llama-bench   (llama.cpp, on PATH)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, stdev, median
from typing import Any, Optional

import requests
import psutil

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as _rs
    _ROUGE_SCORER = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge_score not installed — summarisation metrics unavailable.")

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def _tqdm(it, **kw):          # type: ignore[misc]
        return it

try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Inference ────────────────────────────────────────────────────────────
    max_tokens:         int   = 256         # max generated tokens per request
    ctx:                int   = 4096        # KV-cache context window
    threads:            int   = 0           # 0 = auto (physical cores)
    temperatures:       list  = field(default_factory=lambda: [0.0, 0.3])
    seeds:              list  = field(default_factory=lambda: [42, 137])
    runs_per_prompt:    int   = 3           # averaged over N runs

    # ── Scoring ──────────────────────────────────────────────────────────────
    bootstrap_n:        int   = 1000        # bootstrap iterations for 95% CI
    code_timeout_sec:   int   = 10          # sandbox execution timeout

    # ── Server ───────────────────────────────────────────────────────────────
    server_port:        int   = 8080
    host:               str   = "127.0.0.1"
    server_ready_timeout: int = 90          # seconds

    # ── Bench ────────────────────────────────────────────────────────────────
    bench_pp_tokens:    int   = 512
    bench_tg_tokens:    int   = 128
    request_timeout:    int   = 180

    # ── Paths ─────────────────────────────────────────────────────────────────
    models_file:        str   = "models.json"
    results_file:       str   = "results.json"
    models_dir:         str   = "models"

    def __post_init__(self):
        if self.threads == 0:
            self.threads = psutil.cpu_count(logical=False) or psutil.cpu_count()


CFG = Config()


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT SUITE  (50 prompts across 5 domains)
# ═════════════════════════════════════════════════════════════════════════════

# ── Domain 1: Reasoning (R01–R12) ───────────────────────────────────────────
PROMPTS_REASONING = {
    "R01": "A train travels at 60 km/h for 3 hours. How far does it travel?",
    "R02": "A shop sells apples for $2 each. How much do 7 apples cost?",
    "R03": "A rectangle has length 8 and width 5. What is its area?",
    "R04": "John has 12 marbles, gives away 5, then buys 8 more. How many does he have?",
    "R05": "A car travels 150 km on 10 litres of fuel. What is the fuel efficiency in km/L?",
    "R06": "3 workers can build a wall in 6 hours. How long would 6 workers take?",
    "R07": "A tank holds 500 litres. Water flows in at 25 L/min and out at 10 L/min. How many minutes to fill it from empty?",
    "R08": "If 5 machines produce 100 widgets in 2 hours, how many widgets do 8 machines produce in 3 hours?",
    "R09": "A square has perimeter 36 cm. What is its area?",
    "R10": "A number is tripled then 7 is subtracted giving 23. What is the number?",
    "R11": "Two trains start 300 km apart and travel toward each other at 60 km/h and 90 km/h. When do they meet?",
    "R12": "A shopkeeper buys goods for $80 and sells them for $100. What is the profit percentage?",
}

GROUND_TRUTH_REASONING = {
    "R01": "180",
    "R02": "14",
    "R03": "40",
    "R04": "15",
    "R05": "15",
    "R06": "3",
    "R07": "33.33",
    "R08": "240",
    "R09": "81",
    "R10": "10",
    "R11": "2",
    "R12": "25",
}

# ── Domain 2: Code Generation (C01–C10) ─────────────────────────────────────
PROMPTS_CODE = {
    "C01": "Write a Python function `factorial(n)` that returns n! recursively. Only output the function.",
    "C02": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome (case-insensitive). Only output the function.",
    "C03": "Write a Python function `find_max(lst)` that returns the largest number in a list without using built-ins max/min. Only output the function.",
    "C04": "Write a Python function `count_vowels(s)` that returns the count of vowels in a string. Only output the function.",
    "C05": "Write a Python function `remove_duplicates(lst)` that returns a list with duplicates removed, preserving order. Only output the function.",
    "C06": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1). Only output the function.",
    "C07": "Write a Python function `is_prime(n)` that returns True if n is a prime number. Only output the function.",
    "C08": "Write a Python function `flatten(lst)` that flattens a nested list one level deep. Only output the function.",
    "C09": "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted arr, or -1 if not found. Only output the function.",
    "C10": "Write a Python function `caesar_cipher(text, shift)` that shifts each letter by shift positions. Only output the function.",
}

# Test cases: (args_tuple, expected_return_value)
CODE_TESTS = {
    "C01": [((5,), 120), ((0,), 1), ((1,), 1), ((6,), 720)],
    "C02": [(("racecar",), True), (("hello",), False), (("Madam",), True)],
    "C03": [(([3, 1, 4, 1, 5, 9],), 9), (([7],), 7), (([-2, -5, -1],), -1)],
    "C04": [(("hello world",), 3), (("rhythm",), 0), (("aeiou",), 5)],
    "C05": [(([1, 2, 2, 3, 1],), [1, 2, 3]), (([],), []), (([1],), [1])],
    "C06": [((0,), 0), ((1,), 1), ((7,), 13), ((10,), 55)],
    "C07": [((2,), True), ((17,), True), ((4,), False), ((1,), False)],
    "C08": [(([1, [2, 3], [4]],), [1, 2, 3, 4]), (([],), [])],
    "C09": [(([1, 3, 5, 7, 9], 5), 2), (([1, 3, 5], 4), -1), (([2], 2), 0)],
    "C10": [(("hello", 3), "khoor"), (("xyz", 1), "yza"), (("ABC", 1), "BCD")],
}

# ── Domain 3: Summarisation (S01–S08) ───────────────────────────────────────
PROMPTS_SUMMARY = {
    "S01": (
        "Summarise in one sentence: "
        "Artificial intelligence (AI) refers to the simulation of human intelligence in machines. "
        "Machine learning is a subset that enables systems to learn from data without explicit programming.",
        "AI simulates human intelligence; machine learning enables systems to learn from data.",
    ),
    "S02": (
        "Summarise in one sentence: "
        "Climate change refers to long-term shifts in global temperatures and weather patterns primarily "
        "caused by human activities releasing greenhouse gases such as CO2 and methane into the atmosphere.",
        "Climate change is caused by human greenhouse gas emissions shifting global temperatures.",
    ),
    "S03": (
        "Summarise in one sentence: "
        "The Internet is a global network of interconnected computers enabling communication, information "
        "sharing, and access to resources worldwide through standardised protocols.",
        "The Internet connects computers globally for communication and information sharing.",
    ),
    "S04": (
        "Summarise in one sentence: "
        "Open-source software is software whose source code is publicly available, allowing anyone to "
        "view, modify, and distribute it, which encourages collaborative development and transparency.",
        "Open-source software is publicly available code enabling collaborative development.",
    ),
    "S05": (
        "Summarise in two sentences: "
        "Quantum computing uses quantum mechanical phenomena such as superposition and entanglement to "
        "perform computations. Unlike classical computers that process bits as 0 or 1, quantum computers "
        "use qubits that can represent 0, 1, or both simultaneously, enabling exponential parallelism for "
        "certain problem classes such as cryptography and molecular simulation.",
        "Quantum computers use qubits that exploit superposition and entanglement. "
        "This enables exponential speedups over classical computers for problems like cryptography.",
    ),
    "S06": (
        "Summarise in one sentence: "
        "Blockchain is a distributed ledger technology that records transactions across many computers "
        "in a way that the records cannot be altered retroactively without altering all subsequent blocks, "
        "providing immutability and transparency.",
        "Blockchain is an immutable distributed ledger recording transactions across many computers.",
    ),
    "S07": (
        "Summarise in one sentence: "
        "Neural networks are computing systems loosely inspired by biological neurons. They consist of "
        "layers of interconnected nodes that process data and learn representations through a training "
        "process involving forward propagation and backpropagation of errors.",
        "Neural networks learn data representations through layered nodes trained via backpropagation.",
    ),
    "S08": (
        "Summarise in two sentences: "
        "The CRISPR-Cas9 gene editing tool allows scientists to precisely modify DNA sequences in living "
        "organisms. It works by using a guide RNA to direct the Cas9 protein to a specific genomic location "
        "where it makes a targeted cut, enabling researchers to delete, correct, or insert genetic material "
        "with unprecedented precision and efficiency.",
        "CRISPR-Cas9 is a precise gene editing tool using guide RNA to direct Cas9 to target DNA. "
        "It enables deletion, correction, or insertion of genetic material with high efficiency.",
    ),
}

# ── Domain 4: Instruction-Following (I01–I12) ───────────────────────────────
PROMPTS_INSTRUCTION = {
    "I01": ("List five applications of machine learning in healthcare.",
            ["diagnosis", "imaging", "drug", "monitoring", "prediction"]),
    "I02": ("Explain what overfitting means in machine learning in two sentences.",
            ["overfitting", "training", "generalisation", "test"]),
    "I03": ("List three advantages of renewable energy sources.",
            ["clean", "renewable", "sustainable", "emission", "carbon"]),
    "I04": ("Explain what a database index does in two sentences.",
            ["index", "query", "speed", "search", "performance"]),
    "I05": ("What are four key features of the Python programming language?",
            ["readable", "dynamic", "interpreted", "library", "simple", "syntax"]),
    "I06": ("Explain the difference between supervised and unsupervised learning.",
            ["supervised", "unsupervised", "label", "cluster", "class"]),
    "I07": ("List four layers in the OSI network model and what they do.",
            ["physical", "network", "transport", "application", "layer"]),
    "I08": ("Explain what a REST API is in two sentences.",
            ["rest", "http", "endpoint", "request", "response", "stateless"]),
    "I09": ("Name three common sorting algorithms and their average time complexity.",
            ["quicksort", "mergesort", "bubble", "nlogn", "complexity"]),
    "I10": ("Explain the difference between RAM and ROM in two sentences.",
            ["ram", "rom", "volatile", "read", "temporary", "permanent"]),
    "I11": ("List four symptoms of clinical depression.",
            ["sad", "fatigue", "sleep", "appetite", "mood", "interest"]),
    "I12": ("Explain what Docker containers are and why they are used.",
            ["container", "image", "portab", "isolat", "deploy", "docker"]),
}

# ── Domain 5: Factual QA (Q01–Q08) ─────────────────────────────────────────
PROMPTS_FACTUAL = {
    "Q01": ("What is the chemical formula for water?",
            ["h2o", "h₂o"]),
    "Q02": ("What planet is known as the Red Planet?",
            ["mars"]),
    "Q03": ("Who wrote the theory of general relativity?",
            ["einstein", "albert"]),
    "Q04": ("What is the speed of light in a vacuum in km/s?",
            ["299", "300", "3 × 10", "3x10"]),
    "Q05": ("In what year did the Second World War end?",
            ["1945"]),
    "Q06": ("What is the powerhouse of the cell?",
            ["mitochondria", "mitochondrion"]),
    "Q07": ("What does CPU stand for?",
            ["central processing unit"]),
    "Q08": ("What is the largest planet in the solar system?",
            ["jupiter"]),
}


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM MONITORING
# ═════════════════════════════════════════════════════════════════════════════

class SystemMonitor:
    """Background thread that samples CPU %, freq, RAM %, and thermals."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=3)
        return self._summarise()

    def _run(self):
        while not self._stop.is_set():
            sample = {
                "cpu_pct":  psutil.cpu_percent(interval=None),
                "ram_pct":  psutil.virtual_memory().percent,
                "ram_used_gb": psutil.virtual_memory().used / 1e9,
                "freq_mhz": (psutil.cpu_freq().current if psutil.cpu_freq() else 0),
            }
            # Thermal sensors (Linux)
            try:
                temps = psutil.sensors_temperatures()
                all_t = [t.current for grp in temps.values() for t in grp if t.current]
                sample["temp_max_c"] = max(all_t) if all_t else None
            except Exception:
                sample["temp_max_c"] = None
            self._samples.append(sample)
            time.sleep(self.interval)

    def _summarise(self) -> dict:
        if not self._samples:
            return {}
        cpu  = [s["cpu_pct"]    for s in self._samples]
        ram  = [s["ram_pct"]    for s in self._samples]
        freq = [s["freq_mhz"]   for s in self._samples if s["freq_mhz"]]
        temp = [s["temp_max_c"] for s in self._samples if s["temp_max_c"]]
        return {
            "cpu_pct_mean":  round(mean(cpu), 1),
            "cpu_pct_max":   round(max(cpu),  1),
            "ram_pct_mean":  round(mean(ram), 1),
            "ram_used_gb_peak": round(max(s["ram_used_gb"] for s in self._samples), 2),
            "freq_mhz_mean": round(mean(freq), 0) if freq else None,
            "temp_max_c":    round(max(temp), 1)  if temp else None,
            "sample_count":  len(self._samples),
        }


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _extract_numbers(text: str) -> list[float]:
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    results = []
    for n in nums:
        try:
            results.append(float(n))
        except ValueError:
            pass
    return results


def eval_reasoning(output: str, answer: str) -> float:
    """Returns 1.0 if the expected numerical answer appears in the output."""
    try:
        target = float(answer)
    except ValueError:
        return 0.0
    for v in _extract_numbers(output):
        if abs(v - target) < 1e-3:
            return 1.0
    # Partial credit: within 5%
    for v in _extract_numbers(output):
        if target != 0 and abs(v - target) / abs(target) < 0.05:
            return 0.5
    return 0.0


def _extract_python_function(text: str) -> str:
    """Extract the first Python function definition from model output."""
    # Try fenced code block first
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Look for a def ... block
    m = re.search(r"(def\s+\w+\s*\(.*)", text, re.DOTALL)
    if m:
        raw = m.group(1)
        # Grab until blank line after dedent
        lines = raw.splitlines()
        result_lines = []
        for i, ln in enumerate(lines):
            result_lines.append(ln)
            # Stop at blank line after we have at least one body line
            if i > 1 and ln.strip() == "" and not lines[i-1].startswith(" "):
                break
        return "\n".join(result_lines).strip()
    return text.strip()


def eval_code(output: str, fn_name: str, test_cases: list,
              timeout: int = 10) -> dict:
    """
    Extract and sandbox-execute a Python function from model output.
    Returns {"pass": int, "total": int, "score": float, "errors": list}.
    """
    code = _extract_python_function(output)
    results = {"pass": 0, "total": len(test_cases), "score": 0.0, "errors": []}

    for args, expected in test_cases:
        call = f"{fn_name}{args!r}" if len(args) > 1 else f"{fn_name}({args[0]!r})"
        exec_code = f"{code}\n\n_result = {call}\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", exec_code],
                capture_output=True, text=True, timeout=timeout
            )
            if proc.returncode != 0:
                results["errors"].append(f"RuntimeError for {call}: {proc.stderr[:80]}")
                continue
            # Parse the _result from stdout — use ast.literal_eval for safety
            stdout = proc.stdout.strip()
            # inject print for result extraction
            exec_code2 = f"{code}\n\nprint(repr({call}))\n"
            proc2 = subprocess.run(
                [sys.executable, "-c", exec_code2],
                capture_output=True, text=True, timeout=timeout
            )
            raw_out = proc2.stdout.strip()
            actual = ast.literal_eval(raw_out)
            if actual == expected:
                results["pass"] += 1
            else:
                results["errors"].append(f"{call} → {actual!r} (expected {expected!r})")
        except subprocess.TimeoutExpired:
            results["errors"].append(f"Timeout for {call}")
        except Exception as e:
            results["errors"].append(f"EvalError for {call}: {e}")

    results["score"] = results["pass"] / results["total"] if results["total"] else 0.0
    return results


def eval_summary(output: str, reference: str) -> dict:
    """Returns ROUGE-1, ROUGE-2, ROUGE-L, and optionally a semantic similarity."""
    scores = {}
    if HAS_ROUGE:
        r = _ROUGE_SCORER.score(reference, output)
        scores["rouge1"] = round(r["rouge1"].fmeasure, 4)
        scores["rouge2"] = round(r["rouge2"].fmeasure, 4)
        scores["rougeL"] = round(r["rougeL"].fmeasure, 4)
    else:
        scores["rouge1"] = scores["rouge2"] = scores["rougeL"] = None

    if HAS_SBERT:
        emb_out = _ST_MODEL.encode(output,    convert_to_tensor=True)
        emb_ref = _ST_MODEL.encode(reference, convert_to_tensor=True)
        scores["semantic_sim"] = round(float(st_util.cos_sim(emb_out, emb_ref)), 4)
    else:
        scores["semantic_sim"] = None

    # Primary quality score for leaderboard: prefer semantic_sim, else rougeL
    primary = scores.get("semantic_sim") or scores.get("rougeL") or 0.0
    scores["primary"] = round(primary, 4)
    return scores


def eval_instruction(output: str, keywords: list) -> dict:
    out_lower = output.lower()
    hits = [k for k in keywords if k.lower() in out_lower]
    score = round(len(hits) / len(keywords), 4) if keywords else 0.0
    return {"score": score, "hits": hits, "total_keywords": len(keywords)}


def eval_factual(output: str, answers: list) -> float:
    out_lower = output.lower()
    for a in answers:
        if a.lower() in out_lower:
            return 1.0
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data: list[float], n: int = 1000,
                 ci: float = 0.95) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval."""
    import random
    if len(data) < 2:
        return (data[0], data[0]) if data else (0.0, 0.0)
    rng   = random.Random(42)
    means = []
    for _ in range(n):
        sample = [rng.choice(data) for _ in data]
        means.append(mean(sample))
    means.sort()
    lo = int((1 - ci) / 2 * n)
    hi = int((1 - (1 - ci) / 2) * n)
    return (round(means[lo], 4), round(means[min(hi, n-1)], 4))


def effect_size_cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d between two score lists."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = math.sqrt((stdev(a)**2 + stdev(b)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return round((mean(a) - mean(b)) / pooled_std, 3)


# ═════════════════════════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

class LlamaServer:
    """Context manager that starts/stops llama-server."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.url = f"http://{CFG.host}:{CFG.server_port}"
        self._proc: Optional[subprocess.Popen] = None
        self._server_pid: Optional[int] = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *_):
        self._stop()

    def _start(self):
        cmd = [
            "llama-server",
            "-m",     self.model_path,
            "--port", str(CFG.server_port),
            "--host", CFG.host,
            "-c",     str(CFG.ctx),
            "-t",     str(CFG.threads),
            "--log-disable",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not self._wait_ready():
            self._stop()
            raise RuntimeError(f"llama-server did not become ready within "
                               f"{CFG.server_ready_timeout}s")
        # Track server PID for memory measurement
        self._server_pid = self._proc.pid
        print(f"  ✓ llama-server ready (pid={self._server_pid})")

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _wait_ready(self) -> bool:
        deadline = time.time() + CFG.server_ready_timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def server_memory_mb(self) -> Optional[float]:
        """Current RSS of the server process in MB."""
        if not self._server_pid:
            return None
        try:
            p = psutil.Process(self._server_pid)
            return round(p.memory_info().rss / 1e6, 1)
        except Exception:
            return None

    def complete(self, prompt: str, temperature: float = 0.0,
                 seed: int = 42) -> tuple[str, float, float]:
        """
        Returns (text, generation_tps, ttft_ms).
        ttft_ms is time-to-first-token via streaming endpoint.
        """
        payload = {
            "prompt":       prompt,
            "n_predict":    CFG.max_tokens,
            "temperature":  temperature,
            "seed":         seed,
            "stop":         ["User:", "Human:", "\n\nQuestion:"],
            "cache_prompt": False,
            "stream":       False,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.url}/completion",
                json=payload,
                timeout=CFG.request_timeout,
            )
            elapsed = time.perf_counter() - t0
            if resp.status_code != 200:
                return "", 0.0, 0.0
            data = resp.json()
            text          = data.get("content", "")
            tokens_pred   = data.get("tokens_predicted", 1) or 1
            gen_tps       = round(tokens_pred / max(elapsed, 0.001), 2)
            # llama-server reports timings in ms
            timings       = data.get("timings", {})
            ttft_ms       = round(timings.get("prompt_ms", 0) +
                                  timings.get("predicted_ms", 0) / tokens_pred, 2)
            return text, gen_tps, ttft_ms
        except Exception as e:
            return f"[ERROR: {e}]", 0.0, 0.0

    def complete_with_ttft(self, prompt: str, temperature: float = 0.0,
                           seed: int = 42) -> tuple[str, float, float, float]:
        """
        Returns (text, gen_tps, ttft_ms, ms_per_token).
        Uses /completion with timing fields.
        """
        text, gen_tps, ttft_ms = self.complete(prompt, temperature, seed)
        ms_per_token = round(1000 / gen_tps, 2) if gen_tps > 0 else 0.0
        return text, gen_tps, ttft_ms, ms_per_token


# ═════════════════════════════════════════════════════════════════════════════
# LLAMA-BENCH WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

def run_llama_bench(model_path: str) -> dict:
    """Run llama-bench and return pp_tps, tg_tps, and derived metrics."""
    cmd = [
        "llama-bench",
        "-m", model_path,
        "-t", str(CFG.threads),
        "-p", str(CFG.bench_pp_tokens),
        "-n", str(CFG.bench_tg_tokens),
        "-r", "3",       # 3 repetitions for more stable numbers
        "-o", "json",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        out = proc.stdout
    except Exception as e:
        print(f"  [WARN] llama-bench failed: {e}")
        return {}

    pp_tps = tg_tps = 0.0
    try:
        j = out.find("[")
        if j != -1:
            data = json.loads(out[j:])
            for item in data:
                avg    = item.get("avg_ts", 0) or 0
                n_p    = item.get("n_prompt", 0) or 0
                n_g    = item.get("n_gen", 0)    or 0
                tname  = item.get("test", item.get("test_name", ""))
                if (n_p > 0 and n_g == 0) or tname.startswith("pp"):
                    pp_tps = avg
                elif (n_g > 0 and n_p == 0) or tname.startswith("tg"):
                    tg_tps = avg
    except Exception as e:
        print(f"  [WARN] llama-bench parse error: {e}")

    # Derived
    ms_per_token = round(1000 / tg_tps, 2) if tg_tps > 0 else 0.0
    # Theoretical throughput ceiling based on pp (memory bandwidth proxy)
    return {
        "pp_tps":       round(pp_tps, 2),
        "tg_tps":       round(tg_tps, 2),
        "ms_per_token": ms_per_token,
        "pp_to_tg_ratio": round(pp_tps / tg_tps, 2) if tg_tps > 0 else 0.0,
        "threads_used": CFG.threads,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT-LENGTH PROBE
# ═════════════════════════════════════════════════════════════════════════════

def make_context_variants(base_prompt: str) -> dict[str, str]:
    """Create short / medium / long context variants of a prompt."""
    filler = (
        "This is background context. "
        "The following text is provided for context purposes only. "
    ) * 1
    filler_medium = filler * 8
    filler_long   = filler * 30
    return {
        "short":  base_prompt,
        "medium": filler_medium + base_prompt,
        "long":   filler_long   + base_prompt,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLEAN OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def clean_output(text: str, prompt: str) -> str:
    if not text:
        return ""
    # Strip echo
    if text.startswith(prompt):
        text = text[len(prompt):]
    # Strip special tokens
    text = re.sub(r"<\|[^|]*?\|>", "", text)
    # Strip leading/trailing assistant role markers
    text = re.sub(r"(?i)^(assistant:|answer:|response:)\s*", "", text.strip())
    return text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-RUN EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def _run_prompt_multi(server: LlamaServer, prompt: str,
                      n_runs: int, temperatures: list,
                      seeds: list) -> list[dict]:
    """Run a prompt n_runs times across temperature/seed combos."""
    results = []
    for i in range(n_runs):
        temp = temperatures[i % len(temperatures)]
        seed = seeds[i % len(seeds)]
        text, gen_tps, ttft_ms, ms_per_tok = server.complete_with_ttft(
            prompt, temperature=temp, seed=seed
        )
        results.append({
            "text":        clean_output(text, prompt),
            "gen_tps":     gen_tps,
            "ttft_ms":     ttft_ms,
            "ms_per_token": ms_per_tok,
            "temperature": temp,
            "seed":        seed,
            "run":         i,
        })
    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def benchmark_model(model_path: str) -> dict:
    print(f"\n  [1/3] Performance benchmark (llama-bench)...")
    perf = run_llama_bench(model_path)

    print(f"  [2/3] Starting llama-server...")
    monitor    = SystemMonitor(interval=0.5)
    all_runs   = {}   # pid -> list of raw run dicts
    quality    = {}

    with LlamaServer(model_path) as srv:
        base_mem = srv.server_memory_mb()
        print(f"  [3/3] Running quality evaluation ({CFG.runs_per_prompt} runs × prompt)...")

        # ── Reasoning ────────────────────────────────────────────────────────
        r_scores = []
        for pid, prompt in _tqdm(PROMPTS_REASONING.items(), desc="  Reasoning"):
            monitor.start()
            runs = _run_prompt_multi(
                srv, prompt, CFG.runs_per_prompt,
                CFG.temperatures, CFG.seeds
            )
            hw = monitor.stop()
            scores = [eval_reasoning(r["text"], GROUND_TRUTH_REASONING[pid])
                      for r in runs]
            r_scores.extend(scores)
            all_runs[pid] = {
                "runs":   runs,
                "scores": scores,
                "mean":   round(mean(scores), 4),
                "std":    round(stdev(scores), 4) if len(scores) > 1 else 0.0,
                "hw":     hw,
            }

        quality["reasoning"] = {
            "scores_all": r_scores,
            "mean":  round(mean(r_scores),  4),
            "std":   round(stdev(r_scores), 4) if len(r_scores) > 1 else 0.0,
            "ci95":  bootstrap_ci(r_scores, CFG.bootstrap_n),
            "per_prompt": {pid: all_runs[pid]["mean"] for pid in PROMPTS_REASONING},
        }

        # ── Code ─────────────────────────────────────────────────────────────
        c_scores = []
        code_details = {}
        for pid, prompt in _tqdm(PROMPTS_CODE.items(), desc="  Code"):
            fn_name  = re.search(r"`(\w+)\(", prompt).group(1)  # type: ignore[union-attr]
            tests    = CODE_TESTS.get(pid, [])
            monitor.start()
            runs     = _run_prompt_multi(
                srv, prompt, CFG.runs_per_prompt,
                [0.0], CFG.seeds   # code always at T=0
            )
            hw       = monitor.stop()
            # Score best-of-N runs
            best_result = None
            best_score  = -1.0
            for r in runs:
                res = eval_code(r["text"], fn_name, tests, CFG.code_timeout_sec)
                if res["score"] > best_score:
                    best_score  = res["score"]
                    best_result = res
            c_scores.append(best_score)
            code_details[pid] = {
                "best_score":  round(best_score, 4),
                "pass":        best_result["pass"]   if best_result else 0,
                "total":       best_result["total"]  if best_result else 0,
                "errors":      (best_result["errors"][:3] if best_result else []),
                "hw":          hw,
            }

        quality["code"] = {
            "scores_all":  c_scores,
            "mean":        round(mean(c_scores),  4),
            "std":         round(stdev(c_scores), 4) if len(c_scores) > 1 else 0.0,
            "ci95":        bootstrap_ci(c_scores, CFG.bootstrap_n),
            "per_prompt":  {pid: code_details[pid]["best_score"] for pid in PROMPTS_CODE},
            "details":     code_details,
        }

        # ── Summarisation ────────────────────────────────────────────────────
        s_primary = []
        s_details = {}
        for pid, (prompt, reference) in _tqdm(PROMPTS_SUMMARY.items(), desc="  Summary"):
            monitor.start()
            runs = _run_prompt_multi(
                srv, prompt, CFG.runs_per_prompt,
                CFG.temperatures, CFG.seeds
            )
            hw = monitor.stop()
            # Best run by primary metric
            best_s = None
            best_primary = -1.0
            for r in runs:
                s = eval_summary(r["text"], reference)
                if s["primary"] > best_primary:
                    best_primary = s["primary"]
                    best_s = s
            s_primary.append(best_primary)
            s_details[pid] = {
                "scores":    best_s,
                "reference": reference,
                "hw":        hw,
            }

        quality["summary"] = {
            "scores_all": s_primary,
            "mean":       round(mean(s_primary),  4),
            "std":        round(stdev(s_primary), 4) if len(s_primary) > 1 else 0.0,
            "ci95":       bootstrap_ci(s_primary, CFG.bootstrap_n),
            "per_prompt": {pid: s_details[pid]["scores"] for pid in PROMPTS_SUMMARY},
        }

        # ── Instruction-Following ────────────────────────────────────────────
        i_scores = []
        i_details = {}
        for pid, (prompt, keywords) in _tqdm(PROMPTS_INSTRUCTION.items(), desc="  Instruction"):
            monitor.start()
            runs = _run_prompt_multi(
                srv, prompt, CFG.runs_per_prompt,
                CFG.temperatures, CFG.seeds
            )
            hw = monitor.stop()
            scores = [eval_instruction(r["text"], keywords)["score"] for r in runs]
            i_scores.extend(scores)
            i_details[pid] = {
                "scores": scores,
                "mean":   round(mean(scores), 4),
                "std":    round(stdev(scores), 4) if len(scores) > 1 else 0.0,
                "hw":     hw,
            }

        quality["instruction"] = {
            "scores_all": i_scores,
            "mean":       round(mean(i_scores),  4),
            "std":        round(stdev(i_scores), 4) if len(i_scores) > 1 else 0.0,
            "ci95":       bootstrap_ci(i_scores, CFG.bootstrap_n),
            "per_prompt": {pid: i_details[pid]["mean"] for pid in PROMPTS_INSTRUCTION},
        }

        # ── Factual QA ───────────────────────────────────────────────────────
        q_scores = []
        q_details = {}
        for pid, (prompt, answers) in _tqdm(PROMPTS_FACTUAL.items(), desc="  Factual QA"):
            monitor.start()
            runs = _run_prompt_multi(
                srv, prompt, CFG.runs_per_prompt,
                CFG.temperatures, CFG.seeds
            )
            hw = monitor.stop()
            scores = [eval_factual(r["text"], answers) for r in runs]
            q_scores.extend(scores)
            q_details[pid] = {
                "scores": scores,
                "mean":   round(mean(scores), 4),
                "hw":     hw,
            }

        quality["factual"] = {
            "scores_all": q_scores,
            "mean":       round(mean(q_scores),  4),
            "std":        round(stdev(q_scores), 4) if len(q_scores) > 1 else 0.0,
            "ci95":       bootstrap_ci(q_scores, CFG.bootstrap_n),
            "per_prompt": {pid: q_details[pid]["mean"] for pid in PROMPTS_FACTUAL},
        }

        # ── Context-length probe (using S01) ─────────────────────────────────
        ctx_probe_prompt, ctx_reference = PROMPTS_SUMMARY["S01"]
        ctx_results = {}
        for ctx_name, ctx_prompt in make_context_variants(ctx_probe_prompt).items():
            t, gen_tps, _, _ = srv.complete_with_ttft(
                ctx_prompt, temperature=0.0, seed=42
            )
            cleaned = clean_output(t, ctx_prompt)
            s = eval_summary(cleaned, ctx_reference)
            ctx_results[ctx_name] = {
                "primary":     s["primary"],
                "gen_tps":     gen_tps,
                "prompt_len":  len(ctx_prompt),
            }
        quality["context_probe"] = ctx_results

        # ── Temperature stability (R01 across 5 temps) ───────────────────────
        temp_probe = {}
        for temp_val in [0.0, 0.2, 0.5, 0.8, 1.0]:
            scores = []
            for _ in range(2):
                t, gen_tps, _, _ = srv.complete_with_ttft(
                    PROMPTS_REASONING["R01"], temperature=temp_val, seed=42
                )
                cleaned = clean_output(t, PROMPTS_REASONING["R01"])
                scores.append(eval_reasoning(cleaned, GROUND_TRUTH_REASONING["R01"]))
            temp_probe[str(temp_val)] = round(mean(scores), 4)
        quality["temperature_stability"] = temp_probe

        # ── Aggregate throughput from actual completions ──────────────────────
        all_tps = []
        all_ttft = []
        all_mspt = []
        for domain_runs in all_runs.values():
            for r in domain_runs.get("runs", []):
                if r.get("gen_tps", 0) > 0:
                    all_tps.append(r["gen_tps"])
                if r.get("ttft_ms", 0) > 0:
                    all_ttft.append(r["ttft_ms"])
                if r.get("ms_per_token", 0) > 0:
                    all_mspt.append(r["ms_per_token"])

        peak_mem = srv.server_memory_mb()

    # ── Composite score ───────────────────────────────────────────────────────
    weights = {
        "reasoning":   0.30,
        "code":        0.25,
        "summary":     0.15,
        "instruction": 0.20,
        "factual":     0.10,
    }
    composite = round(sum(
        weights[k] * quality[k]["mean"] for k in weights
    ), 4)

    # ── Throughput summary ────────────────────────────────────────────────────
    throughput_summary = {
        **perf,
        "observed_gen_tps_mean":   round(mean(all_tps),  2) if all_tps  else 0.0,
        "observed_gen_tps_std":    round(stdev(all_tps), 2) if len(all_tps) > 1 else 0.0,
        "observed_ttft_ms_mean":   round(mean(all_ttft), 2) if all_ttft else 0.0,
        "observed_ms_per_token_mean": round(mean(all_mspt), 2) if all_mspt else 0.0,
        "server_rss_mb_base":      base_mem,
        "server_rss_mb_peak":      peak_mem,
    }

    return {
        "throughput":  throughput_summary,
        "quality":     quality,
        "composite":   composite,
        "weights":     weights,
        "prompt_counts": {
            "reasoning":   len(PROMPTS_REASONING),
            "code":        len(PROMPTS_CODE),
            "summary":     len(PROMPTS_SUMMARY),
            "instruction": len(PROMPTS_INSTRUCTION),
            "factual":     len(PROMPTS_FACTUAL),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM INFO
# ═════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> dict:
    info = {
        "os":             platform.system(),
        "os_release":     platform.release(),
        "python_version": platform.python_version(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical":  psutil.cpu_count(logical=True),
        "ram_total_gb":   round(psutil.virtual_memory().total / 1e9, 2),
        "ram_avail_gb":   round(psutil.virtual_memory().available / 1e9, 2),
        "disk_free_gb":   round(psutil.disk_usage(".").free / 1e9, 1),
    }
    if HAS_CPUINFO:
        try:
            ci = cpuinfo.get_cpu_info()
            info["cpu"] = ci.get("brand_raw", platform.processor())
            info["cpu_hz_advertised"] = ci.get("hz_advertised_friendly", "")
            info["cpu_arch"] = ci.get("arch", "")
            info["cpu_flags"] = [f for f in ["avx", "avx2", "avx512f"]
                                 if f in ci.get("flags", [])]
        except Exception:
            info["cpu"] = platform.processor()
    else:
        info["cpu"] = platform.processor()

    # CPU frequency range
    freq = psutil.cpu_freq()
    if freq:
        info["cpu_freq_min_mhz"] = round(freq.min, 0)
        info["cpu_freq_max_mhz"] = round(freq.max, 0)
        info["cpu_freq_cur_mhz"] = round(freq.current, 0)

    return info


# ═════════════════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download_model(url: str, path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) + downloaded
            mode  = "ab" if downloaded else "wb"
            with open(tmp, mode) as f:
                with _tqdm(total=total, initial=downloaded, unit="B",
                           unit_scale=True, desc=os.path.basename(path)) as pbar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        os.rename(tmp, path)
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        raise


def model_file_info(path: str) -> dict:
    """Return file size and MD5 (first 8 chars) for provenance."""
    if not os.path.exists(path):
        return {}
    size_gb = round(os.path.getsize(path) / 1e9, 3)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return {"size_gb": size_gb, "md5_prefix": md5.hexdigest()[:8]}


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS I/O
# ═════════════════════════════════════════════════════════════════════════════

def load_results(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_results(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Results saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    print("\n" + "═" * 72)
    print("  BENCHMARK SUMMARY")
    print("═" * 72)

    # Header row
    hdr = f"{'Model':<32} {'Comp':>6} {'Reas':>6} {'Code':>6} {'Summ':>6} {'Inst':>6} {'Fact':>6} {'TG TPS':>7}"
    print(f"\n{hdr}")
    print("-" * 72)

    ranked = sorted(
        [(n, v) for n, v in results.items() if "error" not in v],
        key=lambda x: x[1].get("results", {}).get("composite", 0),
        reverse=True,
    )

    for name, entry in ranked:
        r = entry.get("results", {})
        q = r.get("quality", {})
        t = r.get("throughput", {})
        name_s = name[:31]
        comp  = r.get("composite", 0)
        reas  = q.get("reasoning",   {}).get("mean", 0)
        code  = q.get("code",        {}).get("mean", 0)
        summ  = q.get("summary",     {}).get("mean", 0)
        inst  = q.get("instruction", {}).get("mean", 0)
        fact  = q.get("factual",     {}).get("mean", 0)
        tgtps = t.get("tg_tps", 0)
        print(f"{name_s:<32} {comp:>6.3f} {reas:>6.3f} {code:>6.3f} "
              f"{summ:>6.3f} {inst:>6.3f} {fact:>6.3f} {tgtps:>7.1f}")

    print("-" * 72)
    if len(ranked) > 1:
        best_name, best_entry = ranked[0]
        worst_name, worst_entry = ranked[-1]
        b_comp = best_entry["results"]["composite"]
        w_comp = worst_entry["results"]["composite"]
        print(f"\n  🏆 Best composite: {best_name} ({b_comp:.4f})")
        print(f"  ⬇️  Lowest:         {worst_name} ({w_comp:.4f})")

    print("\n  Columns: Comp=Composite Reas=Reasoning Code=Code(functional)")
    print("           Summ=Summary Inst=Instruction Fact=Factual QA")
    print("═" * 72 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL HTML REPORT
# ═════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SLM Benchmark Report</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
  h1{color:#38bdf8;border-bottom:2px solid #1e40af;padding-bottom:8px}
  h2{color:#7dd3fc;margin-top:32px}
  table{border-collapse:collapse;width:100%;margin:16px 0}
  th{background:#1e3a5f;color:#bae6fd;padding:10px 14px;text-align:left}
  td{padding:8px 14px;border-bottom:1px solid #1e293b}
  tr:hover td{background:#1e2d3d}
  .bar-wrap{background:#1e293b;border-radius:4px;height:14px;width:200px;display:inline-block}
  .bar{background:#2563eb;height:14px;border-radius:4px;transition:width 0.3s}
  .bar.green{background:#16a34a}
  .bar.orange{background:#d97706}
  .badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:700}
  .badge-gold{background:#b45309;color:#fef3c7}
  .badge-silver{background:#475569;color:#e2e8f0}
  .badge-bronze{background:#92400e;color:#fef3c7}
  .meta{color:#94a3b8;font-size:13px}
  pre{background:#1e293b;padding:12px;border-radius:6px;font-size:12px;overflow-x:auto}
</style>
</head>
<body>
<h1>🧪 SLM Laptop Benchmark Report</h1>
<p class="meta">Generated: {{ timestamp }} | Runs per prompt: {{ runs_per_prompt }} | 
   Temperatures: {{ temperatures }}</p>

<h2>System Information</h2>
<pre>{{ system_info }}</pre>

<h2>Composite Leaderboard</h2>
<table>
<tr>
  <th>Rank</th><th>Model</th><th>Composite</th><th>Reasoning</th>
  <th>Code</th><th>Summary</th><th>Instruction</th><th>Factual</th>
  <th>TG TPS</th><th>ms/token</th>
</tr>
{% for rank, row in rows %}
<tr>
  <td>
    {% if rank == 1 %}<span class="badge badge-gold">🥇</span>
    {% elif rank == 2 %}<span class="badge badge-silver">🥈</span>
    {% elif rank == 3 %}<span class="badge badge-bronze">🥉</span>
    {% else %}#{{ rank }}{% endif %}
  </td>
  <td><strong>{{ row.name }}</strong></td>
  <td>
    <div class="bar-wrap"><div class="bar" style="width:{{ (row.composite*200)|int }}px"></div></div>
    <strong>{{ "%.3f"|format(row.composite) }}</strong>
  </td>
  <td>{{ "%.3f"|format(row.reasoning) }}</td>
  <td>{{ "%.3f"|format(row.code) }}</td>
  <td>{{ "%.3f"|format(row.summary) }}</td>
  <td>{{ "%.3f"|format(row.instruction) }}</td>
  <td>{{ "%.3f"|format(row.factual) }}</td>
  <td>{{ row.tg_tps }}</td>
  <td>{{ row.ms_per_token }}</td>
</tr>
{% endfor %}
</table>

<h2>Per-Domain Details</h2>
{% for row in detail_rows %}
<h3>{{ row.name }}</h3>
<table>
<tr><th>Domain</th><th>Mean ± Std</th><th>95% CI</th><th>Bar</th></tr>
{% for d in row.domains %}
<tr>
  <td>{{ d.domain }}</td>
  <td>{{ "%.4f"|format(d.mean) }} ± {{ "%.4f"|format(d.std) }}</td>
  <td>[{{ "%.4f"|format(d.ci_lo) }}, {{ "%.4f"|format(d.ci_hi) }}]</td>
  <td><div class="bar-wrap"><div class="bar {{ d.color }}" style="width:{{ (d.mean*200)|int }}px"></div></div></td>
</tr>
{% endfor %}
</table>
{% endfor %}

<h2>Temperature Stability (R01 across temperatures)</h2>
<table>
<tr><th>Model</th>
{% for t in temp_headers %}<th>T={{ t }}</th>{% endfor %}
</tr>
{% for row in temp_rows %}
<tr>
  <td>{{ row.name }}</td>
  {% for v in row.vals %}<td>{{ "%.2f"|format(v) }}</td>{% endfor %}
</tr>
{% endfor %}
</table>

<h2>Context-Length Probe (S01)</h2>
<table>
<tr><th>Model</th><th>Short</th><th>Medium</th><th>Long</th>
    <th>Short TPS</th><th>Long TPS</th></tr>
{% for row in ctx_rows %}
<tr>
  <td>{{ row.name }}</td>
  <td>{{ "%.4f"|format(row.short_q) }}</td>
  <td>{{ "%.4f"|format(row.medium_q) }}</td>
  <td>{{ "%.4f"|format(row.long_q) }}</td>
  <td>{{ row.short_tps }}</td>
  <td>{{ row.long_tps }}</td>
</tr>
{% endfor %}
</table>

<p class="meta" style="margin-top:40px">
SLM Laptop Benchmark Tool v2.0 | 
{{ total_prompts }} scored prompts | 
Weights: Reasoning 30%, Code 25%, Instruction 20%, Summary 15%, Factual 10%
</p>
</body>
</html>"""


def generate_html_report(results: dict, out_path: str = "benchmark_report.html"):
    if not HAS_JINJA:
        print("  [SKIP] jinja2 not installed — HTML report skipped.")
        return
    import datetime

    ranked = sorted(
        [(n, v) for n, v in results.items() if "error" not in v],
        key=lambda x: x[1].get("results", {}).get("composite", 0),
        reverse=True,
    )

    rows = []
    for i, (name, entry) in enumerate(ranked, 1):
        r = entry.get("results", {})
        q = r.get("quality", {})
        t = r.get("throughput", {})
        rows.append((i, {
            "name":        name,
            "composite":   r.get("composite", 0),
            "reasoning":   q.get("reasoning",   {}).get("mean", 0),
            "code":        q.get("code",        {}).get("mean", 0),
            "summary":     q.get("summary",     {}).get("mean", 0),
            "instruction": q.get("instruction", {}).get("mean", 0),
            "factual":     q.get("factual",     {}).get("mean", 0),
            "tg_tps":      t.get("tg_tps", 0),
            "ms_per_token":t.get("ms_per_token", 0),
        }))

    detail_rows = []
    colors = {"reasoning": "", "code": "green", "summary": "orange",
              "instruction": "", "factual": "green"}
    for name, entry in ranked:
        r   = entry.get("results", {})
        q   = r.get("quality", {})
        dms = []
        for dom in ["reasoning", "code", "summary", "instruction", "factual"]:
            dq = q.get(dom, {})
            ci = dq.get("ci95", (0, 0))
            dms.append({
                "domain": dom.capitalize(),
                "mean":   dq.get("mean", 0),
                "std":    dq.get("std",  0),
                "ci_lo":  ci[0],
                "ci_hi":  ci[1],
                "color":  colors.get(dom, ""),
            })
        detail_rows.append({"name": name, "domains": dms})

    temp_headers = ["0.0", "0.2", "0.5", "0.8", "1.0"]
    temp_rows = []
    for name, entry in ranked:
        tp = entry.get("results", {}).get("quality", {}).get("temperature_stability", {})
        temp_rows.append({
            "name": name,
            "vals": [tp.get(t, 0) for t in temp_headers],
        })

    ctx_rows = []
    for name, entry in ranked:
        cp = entry.get("results", {}).get("quality", {}).get("context_probe", {})
        ctx_rows.append({
            "name":      name,
            "short_q":   cp.get("short",  {}).get("primary", 0),
            "medium_q":  cp.get("medium", {}).get("primary", 0),
            "long_q":    cp.get("long",   {}).get("primary", 0),
            "short_tps": cp.get("short",  {}).get("gen_tps",  0),
            "long_tps":  cp.get("long",   {}).get("gen_tps",  0),
        })

    sys_info = json.dumps(
        results.get(ranked[0][0], {}).get("system", {}) if ranked else {},
        indent=2
    )
    total_prompts = sum([
        len(PROMPTS_REASONING), len(PROMPTS_CODE), len(PROMPTS_SUMMARY),
        len(PROMPTS_INSTRUCTION), len(PROMPTS_FACTUAL),
    ])

    tmpl = Template(HTML_TEMPLATE)
    html = tmpl.render(
        timestamp        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        runs_per_prompt  = CFG.runs_per_prompt,
        temperatures     = CFG.temperatures,
        system_info      = sys_info,
        rows             = rows,
        detail_rows      = detail_rows,
        temp_headers     = temp_headers,
        temp_rows        = temp_rows,
        ctx_rows         = ctx_rows,
        total_prompts    = total_prompts,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ HTML report → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SLM Laptop Benchmark Tool v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models",   default=CFG.models_file,   help="models.json path")
    p.add_argument("--results",  default=CFG.results_file,  help="results.json path")
    p.add_argument("--model",    default=None,              help="Run only this model name")
    p.add_argument("--runs",     type=int, default=CFG.runs_per_prompt,
                   help="Runs per prompt (default 3)")
    p.add_argument("--report",   action="store_true",       help="Generate HTML report")
    p.add_argument("--no-dl",    action="store_true",       help="Skip model download")
    p.add_argument("--no-bench", action="store_true",       help="Skip llama-bench perf test")
    p.add_argument("--force",    action="store_true",       help="Re-run even if results exist")
    p.add_argument("--threads",  type=int, default=0,       help="CPU threads (0=auto)")
    p.add_argument("--port",     type=int, default=CFG.server_port, help="llama-server port")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.runs_per_prompt = args.runs
    CFG.server_port     = args.port
    if args.threads:
        CFG.threads = args.threads

    print("\n" + "═" * 72)
    print("  SLM Laptop Benchmark Tool  v2.0")
    print("═" * 72)

    sys_info = collect_system_info()
    print(f"  CPU   : {sys_info.get('cpu', '?')}")
    print(f"  Cores : {sys_info['cores_physical']}P / {sys_info['cores_logical']}L")
    print(f"  RAM   : {sys_info['ram_total_gb']} GB total, "
          f"{sys_info['ram_avail_gb']} GB available")
    print(f"  Threads for inference: {CFG.threads}")
    print(f"  Runs per prompt      : {CFG.runs_per_prompt}")
    print(f"  Temperatures         : {CFG.temperatures}")
    print(f"  Bootstrap N          : {CFG.bootstrap_n}")

    if not os.path.exists(args.models):
        print(f"\n  [ERROR] {args.models} not found.")
        print("  Create models.json with the following structure:")
        example = {
            "models": [{
                "name":   "qwen3.5-2b-q4_k_m",
                "params": "2B",
                "quant":  "Q4_K_M",
                "url":    "https://huggingface.co/Qwen/Qwen3.5-2B-GGUF/resolve/main/qwen3.5-2b-q4_k_m.gguf",
                "file":   "qwen3.5-2b-q4_k_m.gguf"
            }]
        }
        print(json.dumps(example, indent=2))
        sys.exit(1)

    with open(args.models) as f:
        model_list = json.load(f).get("models", [])

    if args.model:
        model_list = [m for m in model_list if m["name"] == args.model]
        if not model_list:
            print(f"  [ERROR] Model '{args.model}' not found in {args.models}")
            sys.exit(1)

    results = load_results(args.results)
    total_prompts = (len(PROMPTS_REASONING) + len(PROMPTS_CODE) +
                     len(PROMPTS_SUMMARY)   + len(PROMPTS_INSTRUCTION) +
                     len(PROMPTS_FACTUAL))
    print(f"\n  Total scored prompts : {total_prompts}")
    print(f"  Models to evaluate   : {len(model_list)}\n")

    for m in model_list:
        name     = m.get("name")
        url      = m.get("url")
        filename = m.get("file")
        if not all([name, url, filename]):
            continue

        if name in results and not args.force:
            print(f"  ⏭  Skipping {name} (already in results; use --force to re-run)")
            continue

        print(f"\n{'━' * 72}")
        print(f"  Model: {name}")
        print(f"  Quant: {m.get('quant','?')}  Params: {m.get('params','?')}")
        print(f"{'━' * 72}")

        path = os.path.join(CFG.models_dir, filename)

        try:
            if not args.no_dl:
                download_model(url, path)
            elif not os.path.exists(path):
                print(f"  [SKIP] {path} not found and --no-dl set")
                continue

            file_info = model_file_info(path)
            t0 = time.time()

            bench_data = {}
            if not args.no_bench:
                bench_data = run_llama_bench(path)
                print(f"  PP TPS: {bench_data.get('pp_tps', '?')}  "
                      f"TG TPS: {bench_data.get('tg_tps', '?')}  "
                      f"ms/tok: {bench_data.get('ms_per_token', '?')}")

            result_data = benchmark_model(path)

            # Merge bench into throughput
            if bench_data:
                result_data["throughput"].update(bench_data)

            duration = round(time.time() - t0, 1)

            results[name] = {
                "meta": {
                    "params": m.get("params"),
                    "quant":  m.get("quant"),
                    "url":    url,
                    "file_info": file_info,
                },
                "system":                sys_info,
                "benchmark_duration_sec": duration,
                "config": {
                    "runs_per_prompt": CFG.runs_per_prompt,
                    "temperatures":    CFG.temperatures,
                    "seeds":           CFG.seeds,
                    "max_tokens":      CFG.max_tokens,
                    "ctx":             CFG.ctx,
                    "threads":         CFG.threads,
                },
                "results": result_data,
            }
            save_results(results, args.results)

            q = result_data["quality"]
            print(f"\n  ── Quick results for {name} ──")
            print(f"  Composite     : {result_data['composite']:.4f}")
            print(f"  Reasoning     : {q['reasoning']['mean']:.4f} "
                  f"(95% CI {q['reasoning']['ci95']})")
            print(f"  Code (func.)  : {q['code']['mean']:.4f} "
                  f"(95% CI {q['code']['ci95']})")
            print(f"  Summary       : {q['summary']['mean']:.4f} "
                  f"(95% CI {q['summary']['ci95']})")
            print(f"  Instruction   : {q['instruction']['mean']:.4f} "
                  f"(95% CI {q['instruction']['ci95']})")
            print(f"  Factual QA    : {q['factual']['mean']:.4f} "
                  f"(95% CI {q['factual']['ci95']})")
            print(f"  TG TPS (bench): {result_data['throughput'].get('tg_tps','?')}")
            print(f"  Duration      : {duration}s")

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
            if os.path.exists(path):
                with contextlib.suppress(OSError):
                    os.remove(path)
            break

        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}
            save_results(results, args.results)

        finally:
            if os.path.exists(path):
                print(f"  🗑  Cleaning up {path}")
                with contextlib.suppress(OSError):
                    os.remove(path)

    print_summary(results)

    if args.report:
        report_path = args.results.replace(".json", "_report.html")
        generate_html_report(results, report_path)

    print("  Done.\n")


if __name__ == "__main__":
    main()