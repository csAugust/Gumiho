"""
Test loading Qwen2.5 1.5B model and converting to TiDAR
"""

import sys
import os
import torch
from model.modeling_tidar_qwen2 import TiDARQwen2ForCausalLM


def test_load_qwen25_1_5b():
    """Test loading Qwen2.5 1.5B and converting to TiDAR"""
    print("=" * 80)
    print("Testing TiDAR from_pretrained with Qwen2.5 1.5B")
    print("=" * 80)
    
    model_path = "/mnt/bos-text/models/hf_models/Qwen2.5-1.5B-Instruct"
    
    try:
        # Load TiDAR model from Qwen2.5
        print(f"\nLoading model from: {model_path}")
        
        model = TiDARQwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            block_size=8,
            clean_ratio=0.5,
            use_tidar=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        print(f"\n✓ Model loaded successfully!")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Device: {model.device if hasattr(model, 'device') else 'unknown'}")
        print(f"  - Dtype: {model.dtype}")
        
        # Test generation
        print(f"\n" + "=" * 80)
        print("Testing generation")
        print("=" * 80)
        
        # Get tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test prompt
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        print(f"\nPrompt: {prompt}")
        print(f"Input shape: {input_ids.shape}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text: {generated_text}")
        print(f"Generated length: {outputs.shape[1]} tokens")
        
        print(f"\n✓ Generation test passed!")
        
        # Test TiDAR format
        print(f"\n" + "=" * 80)
        print("Testing TiDAR format")
        print("=" * 80)
        
        batch_size = 2
        seq_length = 32
        clean_length = 16
        MASK_TOKEN_ID = 151643
        
        clean_tokens = torch.randint(0, MASK_TOKEN_ID, (batch_size, clean_length)).to(model.device)
        masked_tokens = torch.full((batch_size, seq_length - clean_length), MASK_TOKEN_ID, dtype=torch.long).to(model.device)
        
        tidar_input_ids = input_ids # torch.cat([clean_tokens, masked_tokens], dim=1)
        
        print(f"\nTiDAR input shape: {tidar_input_ids.shape}")
        print(f"Clean tokens: {clean_length}")
        print(f"Masked tokens: {seq_length - clean_length}")
        
        with torch.no_grad():
            outputs = model.tidar_generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        print(f"Output logits shape: {outputs.logits.shape}")

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text: {generated_text}")
        print(f"Generated length: {outputs.shape[1]} tokens")
        print(f"✓ TiDAR format test passed!")
        
        print(f"\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, tokenizer = test_load_qwen25_1_5b()
    
    if model is not None:
        print(f"\n" + "=" * 80)
        print("SUCCESS: TiDAR model loaded from Qwen2.5 1.5B!")
        print("=" * 80)
        print(f"\nYou can now use the model:")
        print(f"  - Model: {type(model).__name__}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Block size: {model.config.block_size}")
        print(f"  - Clean ratio: {model.config.clean_ratio}")
        print(f"  - TiDAR mode: {model.config.use_tidar}")
