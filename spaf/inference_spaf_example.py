#!/usr/bin/env python3
# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
Example inference script for SpAF model.

This script demonstrates how to use a trained SpAF model for text generation
with speculative decoding.
"""

import os
import json
import time
import torch
import logging
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gumiho.model.spaf_model import SpAFModel
from gumiho.inference.spaf_generate import spaf_generate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_generation(model, input_ids, num_runs=3, **generation_kwargs):
    """
    Benchmark SpAF generation performance.
    
    Args:
        model: SpAFModel instance.
        input_ids: Input token IDs.
        num_runs: Number of runs for averaging.
        **generation_kwargs: Additional generation arguments.
        
    Returns:
        Average generation time and tokens per second.
    """
    logger.info("Benchmarking generation performance...")
    
    times = []
    total_tokens = 0
    
    for i in range(num_runs):
        start_time = time.time()
        
        output_ids = spaf_generate(
            model=model,
            input_ids=input_ids,
            **generation_kwargs
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        num_new_tokens = output_ids.size(1) - input_ids.size(1)
        total_tokens += num_new_tokens
        
        logger.info(f"Run {i+1}: {elapsed:.2f}s, {num_new_tokens} tokens, "
                   f"{num_new_tokens/elapsed:.2f} tokens/s")
    
    avg_time = sum(times) / len(times)
    avg_tokens = total_tokens / len(times)
    tokens_per_sec = avg_tokens / avg_time
    
    logger.info(f"\nAverage: {avg_time:.2f}s, {avg_tokens:.1f} tokens, "
               f"{tokens_per_sec:.2f} tokens/s")
    
    return avg_time, tokens_per_sec


def compare_with_baseline(model, input_ids, max_new_tokens=100):
    """
    Compare SpAF generation with baseline autoregressive generation.
    
    Args:
        model: SpAFModel instance.
        input_ids: Input token IDs.
        max_new_tokens: Maximum new tokens to generate.
    """
    logger.info("\n" + "="*60)
    logger.info("Comparing SpAF with Baseline")
    logger.info("="*60)
    
    # Baseline generation
    logger.info("\n[Baseline] Autoregressive generation:")
    start = time.time()
    with torch.no_grad():
        baseline_output = model.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    baseline_time = time.time() - start
    baseline_tokens = baseline_output.size(1) - input_ids.size(1)
    baseline_tps = baseline_tokens / baseline_time
    
    logger.info(f"Time: {baseline_time:.2f}s")
    logger.info(f"Tokens: {baseline_tokens}")
    logger.info(f"Speed: {baseline_tps:.2f} tokens/s")
    
    # SpAF generation
    logger.info("\n[SpAF] Speculative generation:")
    start = time.time()
    spaf_output = spaf_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=5,
        temperature=0.0,
    )
    spaf_time = time.time() - start
    spaf_tokens = spaf_output.size(1) - input_ids.size(1)
    spaf_tps = spaf_tokens / spaf_time
    
    logger.info(f"Time: {spaf_time:.2f}s")
    logger.info(f"Tokens: {spaf_tokens}")
    logger.info(f"Speed: {spaf_tps:.2f} tokens/s")
    
    # Speedup
    speedup = baseline_time / spaf_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Efficiency: {(spaf_tps / baseline_tps):.2f}x")
    logger.info(f"{'='*60}")
    
    # Compare outputs
    tokenizer = model.get_tokenizer()
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    spaf_text = tokenizer.decode(spaf_output[0], skip_special_tokens=True)
    
    logger.info("\n[Baseline Output]:")
    logger.info(baseline_text)
    logger.info("\n[SpAF Output]:")
    logger.info(spaf_text)


def main():
    # Configuration
    base_model_path = 'meta-llama/Llama-2-7b-hf'  # Change this
    spaf_checkpoint_path = './spaf_checkpoints/best_model'  # Change this
    
    # SpAF configuration
    spaf_config = {
        'enable_spaf': True,
        'cutoff_layer': 16,  # For 7B model
        'adapter_dim_ratio': 0.25,
        'alignment_head': {
            'num_layers': 2,
            'hidden_dim_ratio': 2.0,
        }
    }
    
    # Generation settings
    generation_config = {
        'max_new_tokens': 100,
        'num_draft_tokens': 5,
        'serial_parallel_transition': 2,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
    }
    
    # Load model
    logger.info("Loading SpAF model...")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"SpAF checkpoint: {spaf_checkpoint_path}")
    
    model = SpAFModel.from_pretrained(
        base_model_path=base_model_path,
        spaf_config=spaf_config,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    
    # Load trained SpAF components
    if os.path.exists(spaf_checkpoint_path):
        logger.info("Loading trained SpAF components...")
        model.load_spaf_components(spaf_checkpoint_path)
    else:
        logger.warning(f"SpAF checkpoint not found at {spaf_checkpoint_path}")
        logger.warning("Using randomly initialized adapters and alignment head")
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    # Example prompts
    prompts = [
        "Once upon a time in a land far away,",
        "The key to building successful AI systems is",
        "In the year 2050, technology will have",
    ]
    
    # Generate for each prompt
    for i, prompt in enumerate(prompts):
        logger.info("\n" + "="*60)
        logger.info(f"Example {i+1}: {prompt}")
        logger.info("="*60)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        logger.info(f"Input length: {input_ids.size(1)} tokens")
        
        # Generate with SpAF
        logger.info("\nGenerating with SpAF...")
        start_time = time.time()
        
        output_ids = spaf_generate(
            model=model,
            input_ids=input_ids,
            **generation_config
        )
        
        generation_time = time.time() - start_time
        num_new_tokens = output_ids.size(1) - input_ids.size(1)
        
        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Print results
        logger.info(f"\nGeneration time: {generation_time:.2f}s")
        logger.info(f"Tokens generated: {num_new_tokens}")
        logger.info(f"Speed: {num_new_tokens/generation_time:.2f} tokens/s")
        logger.info(f"\nGenerated text:")
        logger.info("-" * 60)
        logger.info(generated_text)
        logger.info("-" * 60)
    
    # Benchmark performance
    logger.info("\n" + "="*60)
    logger.info("Performance Benchmark")
    logger.info("="*60)
    
    benchmark_prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(benchmark_prompt, return_tensors='pt').to(device)
    
    benchmark_generation(
        model=model,
        input_ids=input_ids,
        num_runs=3,
        max_new_tokens=50,
        num_draft_tokens=5,
        temperature=0.0,
    )
    
    # Compare with baseline
    if input("Compare with baseline? (y/n): ").lower() == 'y':
        compare_with_baseline(
            model=model,
            input_ids=input_ids,
            max_new_tokens=50,
        )
    
    logger.info("\n" + "="*60)
    logger.info("Inference completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
