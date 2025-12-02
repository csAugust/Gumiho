"""
Test script for TiDAR generation
Compares TiDAR generation with standard autoregressive generation
"""

import torch
import time
from transformers import AutoTokenizer
from tidar.model.modeling_tidar_qwen2 import TiDARQwen2ForCausalLM


def test_tidar_generation():
    """Test TiDAR generation and compare with standard generation."""
    
    # Configuration
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
    draft_len = 5
    max_new_tokens = 100
    prompt = "Once upon a time, in a land far away,"
    
    print("=" * 80)
    print("TiDAR Generation Test")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Draft Length: {draft_len}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load TiDAR model
    model = TiDARQwen2ForCausalLM.from_pretrained(
        model_path,
        block_size=draft_len,
        clean_ratio=0.5,
        use_tidar=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input length: {input_ids.shape[1]} tokens")
    
    # Test 1: Standard Generation
    print("\n" + "=" * 80)
    print("Test 1: Standard Autoregressive Generation")
    print("=" * 80)
    
    torch.cuda.empty_cache() if device == "cuda" else None
    
    start_time = time.time()
    with torch.no_grad():
        standard_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    standard_time = end_time - start_time
    standard_tokens = standard_output.shape[1] - input_ids.shape[1]
    standard_speed = standard_tokens / standard_time
    
    standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=True)
    
    print(f"Generated {standard_tokens} tokens in {standard_time:.2f}s")
    print(f"Speed: {standard_speed:.2f} tokens/s")
    print(f"\nGenerated text:\n{standard_text[len(prompt):]}")
    
    # Test 2: TiDAR Generation
    print("\n" + "=" * 80)
    print("Test 2: TiDAR Speculative Generation")
    print("=" * 80)
    
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Track statistics
    total_draft_tokens = 0
    total_accepted_tokens = 0
    num_iterations = 0
    
    # Monkey-patch to track statistics
    original_validate = model._validate_draft_tokens
    
    def validate_with_stats(draft_tokens, validation_logits, logits_processor, device):
        nonlocal total_draft_tokens, total_accepted_tokens, num_iterations
        accept_length, resampled_token = original_validate(
            draft_tokens, validation_logits, logits_processor, device
        )
        total_draft_tokens += draft_tokens.shape[1]
        total_accepted_tokens += accept_length
        if resampled_token is not None:
            total_accepted_tokens += 1
        num_iterations += 1
        return accept_length, resampled_token
    
    model._validate_draft_tokens = validate_with_stats
    
    start_time = time.time()
    with torch.no_grad():
        tidar_output = model.tidar_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Greedy decoding
            draft_len=draft_len
        )
    end_time = time.time()
    
    # Restore original method
    model._validate_draft_tokens = original_validate
    
    tidar_time = end_time - start_time
    tidar_tokens = tidar_output.shape[1] - input_ids.shape[1]
    tidar_speed = tidar_tokens / tidar_time
    
    tidar_text = tokenizer.decode(tidar_output[0], skip_special_tokens=True)
    
    # Calculate statistics
    acceptance_rate = (total_accepted_tokens / total_draft_tokens * 100) if total_draft_tokens > 0 else 0
    avg_accepted_per_iter = total_accepted_tokens / num_iterations if num_iterations > 0 else 0
    
    print(f"Generated {tidar_tokens} tokens in {tidar_time:.2f}s")
    print(f"Speed: {tidar_speed:.2f} tokens/s")
    print(f"\nStatistics:")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Total draft tokens: {total_draft_tokens}")
    print(f"  Total accepted tokens: {total_accepted_tokens}")
    print(f"  Acceptance rate: {acceptance_rate:.2f}%")
    print(f"  Avg accepted per iteration: {avg_accepted_per_iter:.2f}")
    print(f"\nGenerated text:\n{tidar_text[len(prompt):]}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"Standard Generation:")
    print(f"  Time: {standard_time:.2f}s")
    print(f"  Speed: {standard_speed:.2f} tokens/s")
    print(f"\nTiDAR Generation:")
    print(f"  Time: {tidar_time:.2f}s")
    print(f"  Speed: {tidar_speed:.2f} tokens/s")
    print(f"  Acceptance Rate: {acceptance_rate:.2f}%")
    print(f"\nSpeedup: {tidar_speed / standard_speed:.2f}x")
    
    # Check if outputs match (should be identical with greedy decoding)
    if torch.equal(standard_output, tidar_output):
        print("\n✓ Outputs match! (Greedy decoding produces identical results)")
    else:
        print("\n✗ Outputs differ")
        print(f"  Standard tokens: {standard_tokens}")
        print(f"  TiDAR tokens: {tidar_tokens}")
        # Show first difference
        for i in range(min(len(standard_output[0]), len(tidar_output[0]))):
            if standard_output[0][i] != tidar_output[0][i]:
                print(f"  First difference at position {i}")
                print(f"    Standard: {tokenizer.decode([standard_output[0][i]])}")
                print(f"    TiDAR: {tokenizer.decode([tidar_output[0][i]])}")
                break
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_tidar_generation()
