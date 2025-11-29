"""
Example: Load Qwen2.5 1.5B and convert to TiDAR
"""

import torch
from tidar.model.modeling_tidar_qwen2 import TiDARQwen2ForCausalLM
from transformers import AutoTokenizer


def main():
    """Example of loading Qwen2.5 1.5B and converting to TiDAR"""
    
    print("=" * 80)
    print("TiDAR-Qwen2.5 1.5B Example")
    print("=" * 80)
    
    # Model path
    model_path = "/mnt/bos-text/models/hf_models/Qwen2.5-1.5B-Instruct"
    
    # Load TiDAR model from Qwen2.5
    print(f"\nLoading TiDAR model from Qwen2.5...")
    model = TiDARQwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        block_size=8,
        clean_ratio=0.5,
        use_tidar=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Block size: {model.config.block_size}")
    print(f"  - Clean ratio: {model.config.clean_ratio}")
    print(f"  - TiDAR mode: {model.config.use_tidar}")
    
    # Example 1: Standard generation
    print(f"\n" + "=" * 80)
    print("Example 1: Standard Generation")
    print("=" * 80)
    
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    
    print(f"\nPrompt: {prompt}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")
    
    # Example 2: TiDAR format input
    print(f"\n" + "=" * 80)
    print("Example 2: TiDAR Format Input")
    print("=" * 80)
    
    batch_size = 1
    seq_length = 32
    clean_length = 16
    MASK_TOKEN_ID = 151643
    
    # Create TiDAR format input
    clean_tokens = torch.randint(0, MASK_TOKEN_ID, (batch_size, clean_length)).to(model.device)
    masked_tokens = torch.full((batch_size, seq_length - clean_length), MASK_TOKEN_ID, dtype=torch.long).to(model.device)
    
    tidar_input_ids = torch.cat([clean_tokens, masked_tokens], dim=1)
    
    print(f"\nTiDAR input:")
    print(f"  - Clean tokens: {clean_length}")
    print(f"  - Masked tokens: {seq_length - clean_length}")
    print(f"  - Total length: {seq_length}")
    
    with torch.no_grad():
        outputs = model(tidar_input_ids)
    
    print(f"\nOutput logits shape: {outputs.logits.shape}")
    print(f"✓ TiDAR format processing successful!")
    
    # Example 3: Different configurations
    print(f"\n" + "=" * 80)
    print("Example 3: Different Configurations")
    print("=" * 80)
    
    configs = [
        {"block_size": 4, "clean_ratio": 0.5, "name": "Small blocks"},
        {"block_size": 16, "clean_ratio": 0.5, "name": "Large blocks"},
        {"block_size": 8, "clean_ratio": 0.7, "name": "More clean tokens"},
        {"block_size": 8, "clean_ratio": 0.3, "name": "More masked tokens"},
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  - Block size: {config['block_size']}")
        print(f"  - Clean ratio: {config['clean_ratio']}")
    
    print(f"\n" + "=" * 80)
    print("✓ All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
