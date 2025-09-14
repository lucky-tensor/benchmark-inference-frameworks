#!/usr/bin/env python3
import os
import sys
import argparse
import time
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set model download directory and enable JIT
os.environ['TINYGRAD_DOWNLOAD_CACHE'] = os.path.expanduser('~/models')
os.environ['TINYGRAD_JIT'] = '1'

# Import GPU memory utilities
from tinygrad.helpers import GlobalCounters

# Import chat interface for interactive mode
from chat_interface import (
    ChatSession, ChatMessage, MessageRole, ResponseStats,
    LLaMA3ChatInterface, GPT2ChatInterface, create_chat_interface,
    stream_generate_tokens
)

@dataclass
class InferenceMetrics:
    """Comprehensive inference metrics collection"""
    model_name: str
    timestamp: str

    # Model loading metrics
    model_load_start: float = 0.0
    model_load_end: float = 0.0
    model_load_duration: float = 0.0
    model_was_cached: bool = False

    # Prompt processing metrics
    prompt_length: int = 0
    prompt_processing_start: float = 0.0
    prompt_processing_end: float = 0.0
    prompt_processing_duration: float = 0.0

    # Token generation metrics
    generation_start: float = 0.0
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    time_to_first_token: float = 0.0  # TTFT
    time_per_token: float = 0.0       # TPT
    total_tokens: int = 0
    tokens_per_second: float = 0.0

    # System metrics
    gpu_memory_before: float = 0.0
    gpu_memory_after: float = 0.0
    gpu_memory_peak: float = 0.0

    # Infrastructure metrics
    device_info: str = ""
    shard_count: int = 1
    quantization: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging"""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'model_loading': {
                'duration_ms': round(self.model_load_duration * 1000, 2),
                'was_cached': self.model_was_cached
            },
            'prompt_processing': {
                'prompt_length': self.prompt_length,
                'duration_ms': round(self.prompt_processing_duration * 1000, 2)
            },
            'token_generation': {
                'total_tokens': self.total_tokens,
                'time_to_first_token_ms': round(self.time_to_first_token * 1000, 2),
                'time_per_token_ms': round(self.time_per_token * 1000, 2),
                'tokens_per_second': round(self.tokens_per_second, 2)
            },
            'memory': {
                'before_mb': round(self.gpu_memory_before, 1),
                'after_mb': round(self.gpu_memory_after, 1),
                'peak_mb': round(self.gpu_memory_peak, 1)
            },
            'infrastructure': {
                'device_info': self.device_info,
                'shard_count': self.shard_count,
                'quantization': self.quantization or 'none'
            }
        }

class VerboseLogger:
    """Verbose logging system for inference operations"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None

    def log_step(self, step_name: str, details: str = ""):
        """Log a step with timing information"""
        now = time.time()
        elapsed = now - self.start_time

        if self.current_step:
            step_duration = now - self.step_times[self.current_step]
            if self.verbose:
                print(f"  ✓ {self.current_step} completed in {step_duration:.3f}s")

        self.current_step = step_name
        self.step_times[step_name] = now

        if self.verbose:
            status = f"[{elapsed:.3f}s] {step_name}"
            if details:
                status += f" - {details}"
            print(status)

    def log_info(self, message: str, indent: bool = True):
        """Log an informational message"""
        if self.verbose:
            prefix = "  " if indent else ""
            print(f"{prefix}{message}")

    def log_substep(self, message: str):
        """Log a sub-step with additional indentation"""
        if self.verbose:
            print(f"    → {message}")

    def log_timing(self, operation: str, duration: float):
        """Log timing information for an operation"""
        if self.verbose:
            print(f"  ⏱️  {operation}: {duration:.3f}s")

    def log_metrics(self, metrics: InferenceMetrics):
        """Log comprehensive metrics summary"""
        if not self.verbose:
            return

        print("\n" + "="*60)
        print("INFERENCE METRICS SUMMARY")
        print("="*60)

        # Model info
        print(f"Model: {metrics.model_name}")
        print(f"Timestamp: {metrics.timestamp}")
        print(f"Device: {metrics.device_info}")
        if metrics.shard_count > 1:
            print(f"Shards: {metrics.shard_count} GPUs")
        if metrics.quantization:
            print(f"Quantization: {metrics.quantization}")

        print("\nTIMING BREAKDOWN:")
        print(f"  Model Loading:     {metrics.model_load_duration*1000:8.2f}ms {'(cached)' if metrics.model_was_cached else ''}")
        print(f"  Prompt Processing: {metrics.prompt_processing_duration*1000:8.2f}ms ({metrics.prompt_length} tokens)")
        print(f"  Time to First Token: {metrics.time_to_first_token*1000:6.2f}ms")
        print(f"  Token Generation:  {(metrics.last_token_time - metrics.first_token_time)*1000:8.2f}ms ({metrics.total_tokens} tokens)")

        print("\nPERFORMANCE METRICS:")
        print(f"  Tokens per Second: {metrics.tokens_per_second:8.2f} tok/s")
        print(f"  Time per Token:    {metrics.time_per_token*1000:8.2f}ms")

        print("\nMEMORY USAGE:")
        print(f"  Before Loading:    {metrics.gpu_memory_before:8.1f}MB")
        print(f"  After Loading:     {metrics.gpu_memory_after:8.1f}MB")
        print(f"  Peak Usage:        {metrics.gpu_memory_peak:8.1f}MB")

        total_time = metrics.last_token_time - metrics.model_load_start
        print(f"\nTOTAL INFERENCE TIME: {total_time:.3f}s")
        print("="*60)

# Global logger and metrics instances
logger = VerboseLogger()
current_metrics = None

class ModelMemoryTracker:
    """Track model memory usage using tinygrad's GlobalCounters"""
    def __init__(self, cache_file="~/.tinygrad_mem_cache.json"):
        self.cache_file = Path(cache_file).expanduser()
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass

    def get_current_memory_mb(self):
        """Get current GPU memory usage in MB from tinygrad"""
        return GlobalCounters.mem_used / (1024 * 1024)

    def is_model_loaded(self, model_name, tolerance_mb=50):
        """Check if model appears to be loaded based on memory signature"""
        if model_name not in self.cache:
            return False

        cached_mem = self.cache[model_name]['memory_mb']
        current_mem = self.get_current_memory_mb()

        # Check if current memory usage suggests model is loaded
        memory_diff = abs(current_mem - cached_mem)
        is_loaded = memory_diff <= tolerance_mb and current_mem >= cached_mem * 0.8

        if is_loaded:
            age_hours = (time.time() - self.cache[model_name]['timestamp']) / 3600
            print(f"Model {model_name} appears to be loaded ({current_mem:.0f}MB, expected: {cached_mem:.0f}MB, age: {age_hours:.1f}h)")

        return is_loaded

    def record_model_load(self, model_name, memory_before_mb, memory_after_mb):
        """Record model loading statistics"""
        delta_mb = memory_after_mb - memory_before_mb
        self.cache[model_name] = {
            'memory_mb': memory_after_mb,
            'delta_mb': delta_mb,
            'timestamp': time.time()
        }
        self._save_cache()
        print(f"Recorded {model_name}: {delta_mb:.0f}MB delta, total: {memory_after_mb:.0f}MB")

# Global tracker instance
memory_tracker = ModelMemoryTracker()

# Global model cache to keep models loaded in memory
_model_cache = {}

def run_llama3(model_name: str, **kwargs) -> None:
    """Run LLaMA 3 model inference with comprehensive metrics and logging"""
    from llama3 import MODEL_PARAMS, build_transformer, Tokenizer, prefill
    from tinygrad import Tensor, Device
    from pathlib import Path
    import time

    global current_metrics

    # Initialize metrics collection
    current_metrics = InferenceMetrics(
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        shard_count=kwargs.get('shard', 1),
        quantization=kwargs.get('quantize', ''),
        device_info=f"CUDA ({Device.DEFAULT})"
    )

    logger.log_step("INITIALIZING INFERENCE", f"model={model_name}, shards={current_metrics.shard_count}")
    current_metrics.gpu_memory_before = memory_tracker.get_current_memory_mb()
    logger.log_info(f"Initial GPU memory: {current_metrics.gpu_memory_before:.1f}MB")

    # Map friendly names to actual model sizes
    size_map = {
        'llama3-1b': '1B',
        'llama3-8b': '8B',
        'llama3-70b': '70B',
        'llama3-405b': '405B'
    }

    if model_name not in size_map:
        print(f"Error: Unknown LLaMA model '{model_name}'. Available: {list(size_map.keys())}")
        sys.exit(1)

    model_size = size_map[model_name]
    logger.log_info(f"Model size: {model_size} parameters")

    # Get model path (simplified - assumes models are downloaded)
    if model_size == "1B":
        model_path = Path.home() / "models" / "llama3-1b-instruct" / "Llama-3.2-1B-Instruct-Q6_K.gguf"
        tokenizer_path = Path.home() / "models" / "llama3-1b-instruct" / "tokenizer.model"
    elif model_size == "8B":
        model_path = Path.home() / "models" / "llama3-8b-sfr" / "model.safetensors.index.json"
        tokenizer_path = Path.home() / "models" / "llama3-8b-sfr" / "tokenizer.model"
    elif model_size == "70B":
        model_path = Path.home() / "models" / "DeepSeek-R1-Distill-Llama-70B" / "model.safetensors.index.json"
        tokenizer_path = Path.home() / "models" / "DeepSeek-R1-Distill-Llama-70B" / "tokenizer.model"
    else:
        print(f"Model size {model_size} not yet configured for download")
        sys.exit(1)

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print(f"Please download the model first using: python llama3.py --download_model --size {model_size}")
        sys.exit(1)

    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    # Set temperature and other globals needed by llama3.py
    import llama3
    llama3.TEMPERATURE = kwargs['temperature']
    llama3.TOP_K = 50
    llama3.TOP_P = 0.95
    llama3.ALPHA_F = 1.0
    llama3.ALPHA_P = 0.0

    if kwargs.get('seed') is not None:
        Tensor.manual_seed(kwargs['seed'])

    # ========== MODEL LOADING PHASE ==========
    print("\n" + "="*80)
    print("🔄 MODEL LOADING PHASE STARTING")
    print("="*80)

    # Check if model is already cached in memory
    logger.log_step("MODEL LOADING", "Checking model cache")
    shard_info = f"_shard{kwargs.get('shard', 1)}" if kwargs.get('shard', 1) > 1 else ""
    quant_info = f"_q{kwargs.get('quantize')}" if kwargs.get('quantize') else ""
    cache_key = f"{model_name}_{model_size}{shard_info}{quant_info}"

    current_metrics.model_load_start = time.time()

    if cache_key in _model_cache:
        logger.log_info("Model found in memory cache - reusing loaded model")
        current_metrics.model_was_cached = True
        model, tokenizer, device = _model_cache[cache_key]
        current_metrics.model_load_end = time.time()
    else:
        logger.log_info("Model not in cache - loading from disk")
        current_metrics.model_was_cached = False

        # Record memory before loading
        memory_before = memory_tracker.get_current_memory_mb()
        logger.log_info(f"GPU memory before model load: {memory_before:.1f}MB")

        # Setup device for sharding
        device = tuple(f"{Device.DEFAULT}:{i}" for i in range(kwargs.get('shard', 1))) if kwargs.get('shard', 1) > 1 else Device.DEFAULT
        logger.log_info(f"Device configuration: {device}")

        # Set the global device variable that prefill() expects
        llama3.device = device[0] if isinstance(device, tuple) else device

        # Load model with sharding and quantization support
        logger.log_substep("Loading transformer model")
        model = build_transformer(model_path, model_size=model_size, device=device, quantize=kwargs.get('quantize'))

        logger.log_substep("Loading tokenizer")
        tokenizer = Tokenizer(str(tokenizer_path))

        # Cache the loaded model
        _model_cache[cache_key] = (model, tokenizer, device)

        current_metrics.model_load_end = time.time()

        # Record memory after loading
        memory_after = memory_tracker.get_current_memory_mb()
        logger.log_info(f"GPU memory after model load: {memory_after:.1f}MB")
        memory_tracker.record_model_load(model_name, memory_before, memory_after)

    current_metrics.model_load_duration = current_metrics.model_load_end - current_metrics.model_load_start
    logger.log_timing("Model loading", current_metrics.model_load_duration)

    print("="*80)
    print("✅ MODEL LOADING PHASE COMPLETED")
    print(f"⏱️  Duration: {current_metrics.model_load_duration:.3f}s ({'from cache' if current_metrics.model_was_cached else 'from disk'})")
    print("="*80)

    # ========== JIT COMPILATION PHASE ==========
    print("\n" + "="*80)
    print("⚡ JIT COMPILATION PHASE STARTING")
    print("   (First model call will compile kernels - this may take some time)")
    print("="*80)

    # Show JIT cache location
    cache_dir = Path.home() / ".cache" / "tinygrad"
    print(f"🔧 TinyGrad JIT cache: {cache_dir}")
    if cache_dir.exists():
        cache_db = cache_dir / "cache.db"
        if cache_db.exists():
            cache_size = cache_db.stat().st_size / (1024 * 1024)  # MB
            print(f"📦 Cache database: {cache_db} ({cache_size:.1f} MB)")
    print()

    # Capture cache state before JIT compilation
    def get_cache_contents(cache_dir):
        """Get detailed cache contents for comparison."""
        if not cache_dir.exists():
            return {"exists": False, "files": 0, "total_size": 0, "db_entries": [], "db_summary": {}}

        cache_files = list(cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        cache_db = cache_dir / "cache.db"
        db_entries = []
        db_summary = {"total_entries": 0, "kernel_types": {}, "recent_entries": []}

        if cache_db.exists():
            try:
                conn = sqlite3.connect(str(cache_db))
                cursor = conn.cursor()

                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if tables:
                    # Assume the main table is 'cache' or similar - try common names
                    table_candidates = ['cache', 'kernels', 'compiled_kernels']
                    main_table = None

                    for candidate in table_candidates:
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (candidate,))
                        if cursor.fetchone():
                            main_table = candidate
                            break

                    if not main_table and tables:
                        # Use first table if no standard name found
                        main_table = tables[0][0]

                    if main_table:
                        # Get table schema
                        cursor.execute(f"PRAGMA table_info({main_table});")
                        columns = cursor.fetchall()
                        column_names = [col[1] for col in columns]

                        # Get all entries
                        cursor.execute(f"SELECT * FROM {main_table};")
                        rows = cursor.fetchall()

                        db_summary["total_entries"] = len(rows)

                        # Analyze entries for patterns
                        for row in rows:
                            entry = dict(zip(column_names, row))

                            # Try to identify kernel type from key/name patterns
                            kernel_type = "unknown"
                            key_text = ""

                            # Look for identifying info in various fields
                            for field in ['key', 'name', 'kernel_name', 'code']:
                                if field in entry and entry[field]:
                                    key_text += str(entry[field]).lower()

                            # Classify kernel types
                            if 'matmul' in key_text or 'gemm' in key_text or 'dot' in key_text:
                                kernel_type = "matmul"
                            elif 'conv' in key_text:
                                kernel_type = "convolution"
                            elif 'reduce' in key_text or 'sum' in key_text or 'max' in key_text:
                                kernel_type = "reduce"
                            elif 'elementwise' in key_text or 'add' in key_text or 'mul' in key_text:
                                kernel_type = "elementwise"
                            elif 'reshape' in key_text or 'transpose' in key_text:
                                kernel_type = "reshape"

                            db_summary["kernel_types"][kernel_type] = db_summary["kernel_types"].get(kernel_type, 0) + 1

                            # Store first few entries as samples
                            if len(db_entries) < 5:
                                db_entries.append(entry)

                conn.close()
            except Exception as e:
                db_summary["error"] = str(e)

        return {
            "exists": True,
            "files": len(cache_files),
            "total_size": total_size,
            "db_entries": db_entries,
            "db_summary": db_summary
        }

    cache_before = get_cache_contents(cache_dir)
    print(f"📊 Cache state before JIT: {cache_before['files']} files, {cache_before['total_size']/(1024*1024):.1f} MB total")

    # Display cache contents summary
    if cache_before.get('exists', False) and cache_before.get('db_summary', {}).get('total_entries', 0) > 0:
        summary = cache_before['db_summary']
        print(f"🗄️  Database contains {summary['total_entries']} cached kernels:")
        for kernel_type, count in summary.get('kernel_types', {}).items():
            print(f"   • {kernel_type}: {count} kernels")

        # Show sample entries
        if cache_before.get('db_entries', []):
            print("📝 Sample cache entries:")
            for i, entry in enumerate(cache_before['db_entries'][:3]):
                # Find the most informative field to display
                key_info = ""
                for field in ['key', 'name', 'kernel_name', 'hash']:
                    if field in entry and entry[field]:
                        key_info = f"{field}={str(entry[field])[:60]}..."
                        break
                if key_info:
                    print(f"   {i+1}. {key_info}")
    else:
        print("🆕 No existing cache found - first run will compile all kernels")

    # Start JIT timing
    jit_start_time = time.time()

    # Helper functions for encoding
    def encode_role(role: str):
        return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")

    def encode_message(role: str, content: str):
        return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

    # Prepare prompt with timing
    logger.log_step("PROMPT PROCESSING", "Encoding user prompt")
    current_metrics.prompt_processing_start = time.time()

    prompt_tokens = [tokenizer.bos_id] + encode_message("user", kwargs['prompt']) + encode_role("assistant")
    current_metrics.prompt_length = len(prompt_tokens)
    logger.log_info(f"Prompt encoded to {current_metrics.prompt_length} tokens")

    # Record memory usage after prompt encoding but before prefill
    current_metrics.gpu_memory_after = memory_tracker.get_current_memory_mb()

    # Add tinygrad cache information
    cache_info = f"~/.cache/tinygrad/cache.db ({os.path.getsize(os.path.expanduser('~/.cache/tinygrad/cache.db')) / 1024 / 1024:.1f}MB)"
    logger.log_info(f"Tinygrad JIT cache: {cache_info}")

    # Run prefill with timing
    logger.log_substep("Running prefill (prompt processing)")
    prefill_start = time.time()
    start_pos = prefill(model, prompt_tokens[:-1])
    prefill_end = time.time()

    current_metrics.prompt_processing_end = prefill_end
    current_metrics.prompt_processing_duration = current_metrics.prompt_processing_end - current_metrics.prompt_processing_start
    logger.log_timing("Prompt processing", current_metrics.prompt_processing_duration)

    # ========== JIT COMPILATION PHASE COMPLETED ==========
    jit_end_time = time.time()
    jit_duration = jit_end_time - jit_start_time

    # Capture cache state after JIT compilation
    cache_after = get_cache_contents(cache_dir)

    print("\n" + "="*80)
    print("✅ JIT COMPILATION PHASE COMPLETED")
    print(f"⏱️  Duration: {jit_duration:.3f}s")
    print("="*80)

    # Show cache changes
    if cache_after.get('exists', False):
        before_files = cache_before.get('files', 0)
        before_size = cache_before.get('total_size', 0)
        files_added = cache_after.get('files', 0) - before_files
        size_added = (cache_after.get('total_size', 0) - before_size) / (1024 * 1024)
        print(f"📊 Cache changes: +{files_added} files, +{size_added:.1f} MB")

        # Compare kernel counts
        after_summary = cache_after.get('db_summary', {})
        before_summary = cache_before.get('db_summary', {})

        if after_summary.get('total_entries', 0) > 0:
            kernels_added = after_summary.get('total_entries', 0) - before_summary.get('total_entries', 0)
            print(f"🔧 Kernels compiled: +{kernels_added} new kernels")

            # Show breakdown of new kernel types
            if kernels_added > 0:
                print("📋 Kernel type breakdown:")
                for kernel_type, after_count in after_summary.get('kernel_types', {}).items():
                    before_count = before_summary.get('kernel_types', {}).get(kernel_type, 0)
                    added_count = after_count - before_count
                    if added_count > 0:
                        print(f"   • {kernel_type}: +{added_count} kernels (total: {after_count})")

        else:
            print("ℹ️  Cache analysis: Unable to read kernel details from database")

        # Special message for first runs
        if before_summary.get('total_entries', 0) == 0:
            print("🆕 First run detected - all kernels compiled from scratch")
    else:
        print("⚠️  Cache directory not found after JIT compilation")

    print("="*80)

    last_tok = prompt_tokens[-1]

    # Generation phase with comprehensive timing metrics
    logger.log_step("TOKEN GENERATION", f"Generating up to {kwargs['count']} tokens")
    current_metrics.generation_start = time.time()

    generated_text = ""
    first_token_generated = False
    tokens_generated = 0

    for i in range(kwargs['count']):
        # Time each token generation
        token_start = time.time()
        tok = model(Tensor([[last_tok]], device=device), start_pos, llama3.TEMPERATURE, llama3.TOP_K, llama3.TOP_P, llama3.ALPHA_F, llama3.ALPHA_P).item()
        token_end = time.time()

        # Record first token timing (TTFT)
        if not first_token_generated:
            current_metrics.first_token_time = token_end
            current_metrics.time_to_first_token = current_metrics.first_token_time - current_metrics.generation_start
            logger.log_timing("Time to First Token (TTFT)", current_metrics.time_to_first_token)
            first_token_generated = True

        start_pos += 1
        last_tok = tok
        tokens_generated += 1

        if tok in tokenizer.stop_tokens:
            logger.log_info(f"Hit stop token, generation complete after {tokens_generated} tokens")
            break

        decoded = tokenizer.decode([tok])
        generated_text += decoded
        if not kwargs.get('noshow'):
            print(decoded, end="", flush=True)

    # Record final metrics
    current_metrics.last_token_time = time.time()
    current_metrics.total_tokens = tokens_generated

    # Calculate comprehensive metrics
    total_generation_time = current_metrics.last_token_time - current_metrics.generation_start
    if tokens_generated > 1:
        time_after_first_token = current_metrics.last_token_time - current_metrics.first_token_time
        current_metrics.time_per_token = time_after_first_token / (tokens_generated - 1)
        current_metrics.tokens_per_second = tokens_generated / total_generation_time
    else:
        current_metrics.time_per_token = total_generation_time
        current_metrics.tokens_per_second = 1.0 / total_generation_time if total_generation_time > 0 else 0.0

    # Final memory measurement
    current_metrics.gpu_memory_peak = memory_tracker.get_current_memory_mb()

    # Log comprehensive metrics summary
    logger.log_step("INFERENCE COMPLETE", "Performance Summary")
    logger.log_info(f"Model: {model_name} ({model_size} params)")
    logger.log_info(f"Prompt length: {current_metrics.prompt_length} tokens")
    logger.log_info(f"Generated: {current_metrics.total_tokens} tokens")
    logger.log_info(f"Model load time: {current_metrics.model_load_duration:.3f}s (cached: {current_metrics.model_was_cached})")
    logger.log_info(f"Prompt processing: {current_metrics.prompt_processing_duration:.3f}s")
    logger.log_info(f"TTFT: {current_metrics.time_to_first_token:.3f}s")
    logger.log_info(f"Time per token: {current_metrics.time_per_token:.3f}s")
    logger.log_info(f"Tokens/sec: {current_metrics.tokens_per_second:.1f}")
    logger.log_info(f"GPU Memory: {current_metrics.gpu_memory_before:.1f}MB -> {current_metrics.gpu_memory_peak:.1f}MB (+{current_metrics.gpu_memory_peak - current_metrics.gpu_memory_before:.1f}MB)")

    if not kwargs.get('noshow'):
        print()  # Final newline

def run_gpt2(model_name: str, **kwargs) -> None:
    """Run GPT-2 model inference"""
    from gpt2 import MODEL_PARAMS, GPT2
    from tinygrad import Tensor

    if model_name not in MODEL_PARAMS and not model_name.startswith("gpt2_gguf_"):
        print(f"Error: Unknown GPT-2 model '{model_name}'. Available: {list(MODEL_PARAMS.keys())}")
        sys.exit(1)

    if kwargs.get('seed') is not None:
        Tensor.manual_seed(kwargs['seed'])

    # Check if model is already cached in memory
    if model_name in _model_cache:
        print(f"Model {model_name} found in memory cache - reusing loaded model")
        gpt2 = _model_cache[model_name]
    else:
        print(f"Model {model_name} not in cache - loading from disk")

        # Record memory before loading
        memory_before = memory_tracker.get_current_memory_mb()

        gpt2 = GPT2.build_gguf(model_name) if model_name.startswith("gpt2_gguf_") else GPT2.build(model_name)

        # Cache the loaded model
        _model_cache[model_name] = gpt2

        # Record memory after loading
        memory_after = memory_tracker.get_current_memory_mb()
        memory_tracker.record_model_load(model_name, memory_before, memory_after)

    texts = gpt2.generate(
        kwargs['prompt'],
        kwargs['count'],
        kwargs['temperature'],
        timing=kwargs['timing'],
        batch_size=kwargs['batch_size']
    )

    if len(texts) == 1:
        print(texts[0])
    else:
        for i, text in enumerate(texts):
            print(f"Response {i}: {text}")

def run_interactive_chat(model_name: str, **kwargs) -> None:
    """Run interactive chat mode using the generic chat interface"""
    import os
    import time
    from pathlib import Path

    print(f"🚀 Starting interactive chat with {model_name}")
    print("💡 Enhanced with real-time performance statistics")
    print("📝 Type 'quit', 'exit', or 'q' to end the session")

    # Show JIT cache location
    cache_dir = Path.home() / ".cache" / "tinygrad"
    print(f"🔧 TinyGrad JIT cache: {cache_dir}")
    if cache_dir.exists():
        cache_db = cache_dir / "cache.db"
        if cache_db.exists():
            cache_size = cache_db.stat().st_size / (1024 * 1024)  # MB
            print(f"📦 Cache database: {cache_db} ({cache_size:.1f} MB)")
    print()

    # Determine model type and create appropriate interface

    # Capture cache state before JIT compilation
    def get_cache_contents(cache_dir):
        """Get detailed cache contents for comparison."""
        if not cache_dir.exists():
            return {"exists": False, "files": 0, "total_size": 0, "db_entries": [], "db_summary": {}}

        cache_files = list(cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        cache_db = cache_dir / "cache.db"
        db_entries = []
        db_summary = {"total_entries": 0, "kernel_types": {}, "recent_entries": []}

        if cache_db.exists():
            try:
                conn = sqlite3.connect(str(cache_db))
                cursor = conn.cursor()

                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                if tables:
                    # Assume the main table is 'cache' or similar - try common names
                    table_candidates = ['cache', 'kernels', 'compiled_kernels']
                    main_table = None

                    for candidate in table_candidates:
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (candidate,))
                        if cursor.fetchone():
                            main_table = candidate
                            break

                    if not main_table and tables:
                        # Use first table if no standard name found
                        main_table = tables[0][0]

                    if main_table:
                        # Get table schema
                        cursor.execute(f"PRAGMA table_info({main_table});")
                        columns = cursor.fetchall()
                        column_names = [col[1] for col in columns]

                        # Get all entries
                        cursor.execute(f"SELECT * FROM {main_table};")
                        rows = cursor.fetchall()

                        db_summary["total_entries"] = len(rows)

                        # Analyze entries for patterns
                        for row in rows:
                            entry = dict(zip(column_names, row))

                            # Try to identify kernel type from key/name patterns
                            key_field = None
                            for field in ['key', 'name', 'kernel_name', 'hash']:
                                if field in column_names:
                                    key_field = entry.get(field, '')
                                    break

                            if key_field:
                                # Extract operation type from key
                                key_str = str(key_field)
                                if 'matmul' in key_str.lower() or 'gemm' in key_str.lower():
                                    op_type = 'matmul'
                                elif 'conv' in key_str.lower():
                                    op_type = 'conv'
                                elif 'reduce' in key_str.lower():
                                    op_type = 'reduce'
                                elif 'elementwise' in key_str.lower() or 'alu' in key_str.lower():
                                    op_type = 'elementwise'
                                else:
                                    # Try to extract from first part of hash-like strings
                                    op_type = 'other'

                                db_summary["kernel_types"][op_type] = db_summary["kernel_types"].get(op_type, 0) + 1

                            # Store first few entries for inspection
                            if len(db_entries) < 5:
                                db_entries.append(entry)

                conn.close()

            except Exception as e:
                db_summary["error"] = str(e)

        return {
            "exists": True,
            "files": len(cache_files),
            "total_size": total_size,
            "db_entries": db_entries,
            "db_summary": db_summary
        }

    cache_before = get_cache_contents(cache_dir)
    print(f"📊 Cache state before JIT: {cache_before['files']} files, {cache_before['total_size']/(1024*1024):.1f} MB total")

    # Display cache contents summary
    if cache_before.get('exists', False) and cache_before.get('db_summary', {}).get('total_entries', 0) > 0:
        summary = cache_before['db_summary']
        print(f"🗄️  Database contains {summary['total_entries']} cached kernels:")
        for kernel_type, count in summary.get('kernel_types', {}).items():
            print(f"   • {kernel_type}: {count} kernels")

        # Show sample entries
        if cache_before.get('db_entries', []):
            print("📝 Sample cache entries:")
            for i, entry in enumerate(cache_before['db_entries'][:3]):
                # Find the most informative field to display
                key_info = ""
                for field in ['key', 'name', 'kernel_name', 'hash']:
                    if field in entry and entry[field]:
                        key_info = f"{field}={str(entry[field])[:60]}..."
                        break
                if key_info:
                    print(f"   {i+1}. {key_info}")
    else:
        print("🆕 No existing cache found - first run will compile all kernels")

    # Start JIT timing
    jit_start_time = time.time()

    # Determine model type and create appropriate interface
    if model_name.startswith('llama3'):
        # Load LLaMA 3 model using existing function
        from llama3 import MODEL_PARAMS, build_transformer, Tokenizer, prefill
        from tinygrad import Tensor, Device

        # Map friendly names to actual model sizes
        size_map = {'llama3-1b': '1B', 'llama3-8b': '8B', 'llama3-70b': '70B', 'llama3-405b': '405B'}
        if model_name not in size_map:
            print(f"Error: Unknown LLaMA model '{model_name}'")
            return

        model_size = size_map[model_name]

        # Get model paths (simplified - assumes models are downloaded)
        if model_size == "1B":
            model_path = Path.home() / "models" / "llama3-1b-instruct" / "Llama-3.2-1B-Instruct-Q6_K.gguf"
            tokenizer_path = Path.home() / "models" / "llama3-1b-instruct" / "tokenizer.model"
        elif model_size == "8B":
            model_path = Path.home() / "models" / "llama3-8b-sfr" / "model.safetensors.index.json"
            tokenizer_path = Path.home() / "models" / "llama3-8b-sfr" / "tokenizer.model"
        elif model_size == "70B":
            model_path = Path.home() / "models" / "DeepSeek-R1-Distill-Llama-70B" / "model.safetensors.index.json"
            tokenizer_path = Path.home() / "models" / "DeepSeek-R1-Distill-Llama-70B" / "tokenizer.model"
        else:
            print(f"Model size {model_size} not configured for interactive mode")
            return

        if not model_path.exists() or not tokenizer_path.exists():
            print(f"Model files not found. Please ensure model is downloaded.")
            return

        # Setup device and load model
        device = tuple(f"{Device.DEFAULT}:{i}" for i in range(kwargs.get('shard', 1))) if kwargs.get('shard', 1) > 1 else Device.DEFAULT
        model = build_transformer(model_path, model_size=model_size, device=device, quantize=kwargs.get('quantize'))
        tokenizer = Tokenizer(str(tokenizer_path))

        # Create chat interface
        chat_interface = LLaMA3ChatInterface(tokenizer)
        session = ChatSession([], "You are a helpful assistant.")

        # Setup initial context
        initial_tokens = chat_interface.encode_chat_session(session)
        start_pos = prefill(model, initial_tokens, device_param=device)

    elif model_name.startswith('gpt2'):
        # Load GPT-2 model using existing function
        from gpt2 import GPT2

        gpt2 = GPT2.build_gguf(model_name) if model_name.startswith("gpt2_gguf_") else GPT2.build(model_name)

        # Create chat interface
        chat_interface = GPT2ChatInterface(gpt2.tokenizer)
        session = ChatSession([], "You are a helpful assistant.")

        # GPT-2 doesn't need initial prefill
        initial_tokens = []
        start_pos = 0

    else:
        print(f"Error: Unsupported model type for '{model_name}'")
        return

    # Initialize variables for the chat loop - these will be set based on model type
    llama_model = None
    llama_device = None
    gpt2_model = None

    # Set model-specific variables after loading
    if model_name.startswith('llama3'):
        llama_model = model
        llama_device = device
    elif model_name.startswith('gpt2'):
        gpt2_model = gpt2

    # JIT compilation is complete after model loading and initial setup
    jit_end_time = time.time()
    jit_duration = jit_end_time - jit_start_time

    # Compare cache state after JIT compilation
    cache_after = get_cache_contents(cache_dir)

    print("="*80)
    print("✅ JIT COMPILATION PHASE COMPLETED")
    print(f"⏱️  Duration: {jit_duration:.3f}s")

    # Show detailed cache changes
    files_added = cache_after['files'] - cache_before['files']
    size_added = (cache_after['total_size'] - cache_before['total_size']) / (1024 * 1024)  # MB

    # Compare kernel counts
    before_kernels = cache_before['db_summary']['total_entries']
    after_kernels = cache_after['db_summary']['total_entries']
    kernels_added = after_kernels - before_kernels

    if kernels_added > 0 or files_added > 0 or size_added > 0.1:
        print(f"📈 Cache changes: +{kernels_added} kernels, +{files_added} files, +{size_added:.1f} MB")
        print("   🔄 New kernels compiled and cached")

        # Show breakdown of new kernel types
        if kernels_added > 0:
            print("🆕 New kernel types compiled:")
            before_types = cache_before['db_summary']['kernel_types']
            after_types = cache_after['db_summary']['kernel_types']

            for kernel_type in after_types:
                before_count = before_types.get(kernel_type, 0)
                after_count = after_types[kernel_type]
                if after_count > before_count:
                    new_count = after_count - before_count
                    print(f"   • {kernel_type}: +{new_count} kernels (total: {after_count})")
    else:
        print("📋 Cache unchanged - all kernels already compiled")
        print("   ⚡ Using existing JIT cache for optimal performance")
        if after_kernels > 0:
            print(f"   📊 Reusing {after_kernels} cached kernels")

    print("="*80)

    # Show additional cache information
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*"))
        print(f"📁 Cache contains {len(cache_files)} files:")
        for cache_file in sorted(cache_files)[:5]:  # Show first 5 files
            if cache_file.is_file():
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                print(f"   {cache_file.name}: {size_mb:.1f} MB")
        if len(cache_files) > 5:
            print(f"   ... and {len(cache_files) - 5} more files")
    print()

    # Interactive chat loop
    while True:
        try:
            user_input = input(chat_interface.format_interactive_prompt())
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            # Add user message to session
            session.add_message(MessageRole.USER, user_input)

            # Generate response with statistics tracking
            stats = chat_interface.create_response_stats()
            stats.start_generation()

            if model_name.startswith('llama3'):
                # LLaMA 3 token-by-token generation
                context_tokens, expected_role = chat_interface.prepare_generation_context(session)

                # Run prefill for new context
                if len(context_tokens) > len(initial_tokens):
                    new_tokens = context_tokens[len(initial_tokens):]
                    start_pos = prefill(llama_model, new_tokens, start_pos=start_pos, device_param=llama_device)
                    initial_tokens = context_tokens

                last_tok = context_tokens[-1]
                response_text = ""
                first_token = True

                # Generate response tokens
                while True:
                    tok = llama_model(Tensor([[last_tok]], device=llama_device), start_pos, kwargs.get('temperature', 0.7), 50, 0.95, 1.0, 0.0).item()

                    if first_token:
                        stats.record_first_token()
                        first_token = False
                    stats.record_token()

                    start_pos += 1
                    last_tok = tok

                    if tok in chat_interface.get_stop_tokens():
                        break

                    decoded = chat_interface.decode_tokens([tok])
                    response_text += decoded
                    print(decoded, end="", flush=True)

                # Update initial_tokens for next iteration
                initial_tokens.extend(chat_interface.encode_message(ChatMessage(MessageRole.USER, user_input)))
                initial_tokens.extend(chat_interface.encode_message(ChatMessage(MessageRole.ASSISTANT, response_text)))

            else:
                # GPT-2 batch generation with proper TTFT timing
                context_tokens, expected_role = chat_interface.prepare_generation_context(session)
                context_text = chat_interface.decode_tokens(context_tokens)

                # Manually calculate TTFT for batch generation
                import time
                generation_start = time.time()

                response_texts = gpt2_model.generate(context_text, kwargs.get('count', 100), kwargs.get('temperature', 0.85), timing=False, batch_size=1)

                # Calculate TTFT manually since GPT-2 does batch generation
                generation_end = time.time()
                stats.time_to_first_token = generation_end - generation_start
                stats.first_token_recorded = True

                if response_texts:
                    full_response = response_texts[0]
                    if full_response.startswith(context_text):
                        response_text = full_response[len(context_text):].strip()
                    else:
                        response_text = full_response.strip()

                    # Count tokens for statistics
                    response_tokens = gpt2_model.tokenizer.encode(response_text)
                    for _ in response_tokens:
                        stats.record_token()

                    print(response_text)
                else:
                    response_text = ""

            # Finalize statistics and display
            stats.finalize()
            print(f"\n{stats.format_stats()}\n")

            # Add assistant response to session
            session.add_message(MessageRole.ASSISTANT, response_text, stats)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

def main():
    # Standard defaults optimized for quality and performance
    defaults = {
        'prompt': "What is the answer to life, the universe, and everything?",
        'count': 100,
        'temperature': 0.85,
        'timing': False,
        'seed': None,
        'batch_size': 1,
        'benchmark': -1,
        'noshow': False,
    }

    # Available models
    available_models = {
        # LLaMA 3 models
        'llama3-1b': 'LLaMA 3 1B parameters',
        'llama3-8b': 'LLaMA 3 8B parameters',
        'llama3-70b': 'LLaMA 3 70B parameters',
        'llama3-405b': 'LLaMA 3 405B parameters',
        # GPT-2 models
        'gpt2': 'GPT-2 124M parameters',
        'gpt2-medium': 'GPT-2 350M parameters',
        'gpt2-large': 'GPT-2 774M parameters',
        'gpt2-xl': 'GPT-2 1.5B parameters',
        # GPT-2 GGUF quantized models
        'gpt2_gguf_q4_0': 'GPT-2 Q4_0 quantized',
        'gpt2_gguf_q8_0': 'GPT-2 Q8_0 quantized',
    }

    parser = argparse.ArgumentParser(
        description='Unified inference for LLaMA 3 and GPT-2 models with multi-GPU support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"Available models:\n" + "\n".join([f"  {k}: {v}" for k, v in available_models.items()]) +
               "\n\nExamples:\n" +
               "  python inference.py --model llama3-1b                      # Interactive chat with LLaMA 3 1B\n" +
               "  python inference.py --model gpt2-medium                    # Interactive chat with GPT-2 Medium\n" +
               "  python inference.py --model llama3-1b --prompt \"Hello\"      # Single generation\n" +
               "  python inference.py --model llama3-8b --shard 2            # 8B model on 2 GPUs\n" +
               "  python inference.py --model llama3-70b --shard 4 --quantize int8  # 70B model on 4 GPUs with quantization"
    )

    parser.add_argument('--model', type=str, required=True, help="Model name to use for inference")
    parser.add_argument('--prompt', type=str, default=defaults['prompt'], help="Text prompt for single generation (if not provided, starts interactive chat)")
    parser.add_argument('--count', type=int, default=defaults['count'], help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=defaults['temperature'], help="Sampling temperature (0.0 = deterministic)")
    parser.add_argument('--timing', action='store_true', help="Show timing information per token")
    parser.add_argument('--seed', type=int, help="Random seed for reproducible generation")
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help="Batch size for generation")
    parser.add_argument('--benchmark', type=int, default=defaults['benchmark'], help="Benchmark mode with N tokens (GPT-2 only)")
    parser.add_argument('--noshow', action='store_true', help="Don't display the generated text")
    parser.add_argument('--shard', type=int, default=1, help="Shard the model across multiple devices")
    parser.add_argument('--quantize', choices=['int8', 'nf4', 'float16'], help="Quantization method")
    parser.add_argument('--list-models', action='store_true', help="List all available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model, desc in available_models.items():
            print(f"  {model:20} - {desc}")
        sys.exit(0)

    if args.model not in available_models:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Use --list-models to see available options")
        sys.exit(1)

    # Convert args to dict for easy passing
    kwargs = vars(args)

    # Validate GPU count for sharding
    if args.shard > 1:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
            gpu_count = len([line for line in result.stdout.strip().split('\n') if line.startswith('GPU')])
            if args.shard > gpu_count:
                print(f"Error: Requested {args.shard} GPUs but only {gpu_count} available")
                print(f"Available GPUs: {gpu_count}")
                sys.exit(1)
            print(f"Using {args.shard} of {gpu_count} available GPUs")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not detect GPU count. Proceeding with sharding...")

    print(f"Loading {args.model}...")

    try:
        # Default to interactive chat mode when no custom prompt is provided
        if args.prompt == defaults['prompt']:
            run_interactive_chat(args.model, **kwargs)
        elif args.model.startswith('llama3'):
            run_llama3(args.model, **kwargs)
        elif args.model.startswith('gpt2'):
            if kwargs.get('shard', 1) > 1 or kwargs.get('quantize'):
                print("Warning: GPT-2 models don't support sharding or quantization in this implementation")
            run_gpt2(args.model, **kwargs)
        else:
            print(f"Error: Unsupported model type for '{args.model}'")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()