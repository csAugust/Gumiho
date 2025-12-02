"""
Test script for TiDAR generation
Compares TiDAR generation with standard autoregressive generation
"""

import torch
import time
from transformers import AutoTokenizer
from model.modeling_tidar_qwen2 import TiDARQwen2ForCausalLM
import os
import sys
from fastchat.model import get_conversation_template

from model.modeling_qwen2 import Qwen2ForCausalLM

# sys.path.append(os.path.dirname(__file__))


def test_tidar_generation():
    """Test TiDAR generation and compare with standard generation."""
    
    # Configuration
    qwen_model_path = "/mnt/bos-text/models/hf_models/Qwen2.5-1.5B-Instruct"  # Small model for testing
    tidar_model_path = "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/tidar/train/tidar_checkpoints/tidar_init"  # Small model for testing
    draft_len = 3
    max_new_tokens = 50
    prompt = "Once upon a time, "

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

    # messages = [
    #     {"role": "system", "content": 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.'},
    #     {"role": "user", "content": "Introduce Large Languag Models."}
    # ]
    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=False,
    # )
    # print(prompt)
    
    # conv = get_conversation_template("qwen2")
    # conv.append_message(conv.roles[0], "Introduce Large Languag Models.")
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()


    print("=" * 80)
    print("TiDAR Generation Test")
    print("=" * 80)
    print(f"Model: {tidar_model_path}")
    print(f"Draft Length: {draft_len}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
        
    # Load TiDAR model
    model = TiDARQwen2ForCausalLM.from_pretrained(
        tidar_model_path,
        block_size=draft_len,
        clean_ratio=0.5,
        use_tidar=True,
        is_training=False,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input length: {input_ids.shape[1]} tokens")
    

    # Test 0: Qwen Generation
    # print("\n" + "=" * 80)
    # print("Test 0: Standard Autoregressive Generation")
    # print("=" * 80)
    
    # qwen_model = Qwen2ForCausalLM.from_pretrained(
    #     qwen_model_path,
    #     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    #     device_map=device
    # )

    # torch.cuda.empty_cache() if device == "cuda" else None
    
    # start_time = time.time()
    # with torch.no_grad():
    #     standard_output = qwen_model.generate(
    #         input_ids,
    #         max_new_tokens=max_new_tokens,
    #         do_sample=False,  # Greedy decoding for reproducibility
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    # end_time = time.time()
    
    # standard_time = end_time - start_time
    # standard_tokens = standard_output.shape[1] - input_ids.shape[1]
    # standard_speed = standard_tokens / standard_time
    
    # standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=True)
    
    # print(f"Generated {standard_tokens} tokens in {standard_time:.2f}s")
    # print(f"Speed: {standard_speed:.2f} tokens/s")
    # print(f"\nGenerated text:\n{standard_text[len(prompt):]}")
    
    # Test 1: Standard Generation
    print("\n" + "=" * 80)
    print("Test 1: Standard Autoregressive Generation")
    print("=" * 80)
    
    existing_model_path = "/mnt/user-ssd/chenzhiyang1/workspace/Train/Gumiho/tidar/train/tidar_checkpoints/epoch_199/pytorch_model.bin"
    checkpoint = torch.load(existing_model_path)
    model.load_state_dict(checkpoint, strict=True)

    torch.cuda.empty_cache() if device == "cuda" else None
    
    start_time = time.time()
    with torch.no_grad():
        standard_output = model.naivegenerate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id,
            tokenizer = tokenizer
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
            draft_len=draft_len,
            tokenizer=tokenizer
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
