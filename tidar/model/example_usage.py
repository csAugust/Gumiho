"""
TiDAR-Qwen2 Usage Examples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tidar import TiDARQwen2Config, TiDARQwen2ForCausalLM


def example_basic_usage():
    """Example 1: Basic usage with standard input"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a small model for demonstration
    config = TiDARQwen2Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        use_tidar=True,
        block_size=4,
        clean_ratio=0.5
    )
    
    model = TiDARQwen2ForCausalLM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Standard input
    input_ids = torch.randint(0, 1000, (2, 16))
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print("✓ Basic usage example completed!\n")


def example_tidar_format():
    """Example 2: Using TiDAR input format"""
    print("=" * 60)
    print("Example 2: TiDAR Input Format")
    print("=" * 60)
    
    config = TiDARQwen2Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        use_tidar=True,
        block_size=4,
        clean_ratio=0.5
    )
    
    model = TiDARQwen2ForCausalLM(config)
    
    # Create TiDAR format input
    batch_size = 2
    seq_length = 16
    clean_length = int(seq_length * config.clean_ratio)
    MASK_TOKEN_ID = 500
    
    print(f"Sequence length: {seq_length}")
    print(f"Clean tokens: {clean_length}")
    print(f"Masked tokens: {seq_length - clean_length}")
    
    # Create input with clean and masked tokens
    clean_tokens = torch.randint(0, MASK_TOKEN_ID, (batch_size, clean_length))
    masked_tokens = torch.full((batch_size, seq_length - clean_length), MASK_TOKEN_ID)
    
    input_ids = torch.cat([clean_tokens, masked_tokens], dim=1)
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print("✓ TiDAR format example completed!\n")


def example_attention_mask():
    """Example 3: Using custom attention mask"""
    print("=" * 60)
    print("Example 3: Custom Attention Mask")
    print("=" * 60)
    
    config = TiDARQwen2Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        use_tidar=True,
        block_size=4,
        clean_ratio=0.5
    )
    
    model = TiDARQwen2ForCausalLM(config)
    
    # Create input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Create custom attention mask (mask last 4 tokens)
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[:, -4:] = 0
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Masked positions: {(attention_mask == 0).sum().item()}")
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print("✓ Custom attention mask example completed!\n")


def example_model_modes():
    """Example 4: Comparing TiDAR mode on/off"""
    print("=" * 60)
    print("Example 4: Model Modes Comparison")
    print("=" * 60)
    
    # Create two models: one with TiDAR, one without
    config_tidar = TiDARQwen2Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        max_position_embeddings=256,
        use_tidar=True,
        block_size=4,
        clean_ratio=0.5
    )
    
    config_normal = TiDARQwen2Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        max_position_embeddings=256,
        use_tidar=False,  # Disable TiDAR
        block_size=4,
        clean_ratio=0.5
    )
    
    model_tidar = TiDARQwen2ForCausalLM(config_tidar)
    model_normal = TiDARQwen2ForCausalLM(config_normal)
    
    print(f"TiDAR model: {model_tidar.config.use_tidar}")
    print(f"Normal model: {model_normal.config.use_tidar}")
    
    # Test with same input
    input_ids = torch.randint(0, 1000, (1, 16))
    
    with torch.no_grad():
        output_tidar = model_tidar(input_ids)
        output_normal = model_normal(input_ids)
    
    print(f"TiDAR output shape: {output_tidar.logits.shape}")
    print(f"Normal output shape: {output_normal.logits.shape}")
    print("✓ Model modes comparison completed!\n")


def example_generation():
    """Example 5: Text generation"""
    print("=" * 60)
    print("Example 5: Text Generation")
    print("=" * 60)
    
    config = TiDARQwen2Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        max_position_embeddings=256,
        use_tidar=True,
        block_size=4,
        clean_ratio=0.5
    )
    
    model = TiDARQwen2ForCausalLM(config)
    
    # Generate text
    input_ids = torch.randint(0, 1000, (1, 16))
    print(f"Input shape: {input_ids.shape}")
    
    # Standard generation (TiDAR generation coming soon)
    generated = model.generate(input_ids, max_new_tokens=32)
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated sequence length: {generated.shape[1]}")
    print("✓ Generation example completed!\n")


def example_custom_config():
    """Example 6: Custom configuration"""
    print("=" * 60)
    print("Example 6: Custom Configuration")
    print("=" * 60)
    
    # Experiment with different configurations
    configs = [
        {
            'name': 'Small model',
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'block_size': 4,
            'clean_ratio': 0.5
        },
        {
            'name': 'Large block size',
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'block_size': 8,
            'clean_ratio': 0.5
        },
        {
            'name': 'Different clean ratio',
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'block_size': 4,
            'clean_ratio': 0.7
        }
    ]
    
    for config_dict in configs:
        config = TiDARQwen2Config(
            vocab_size=1000,
            hidden_size=config_dict['hidden_size'],
            num_hidden_layers=config_dict['num_hidden_layers'],
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=256,
            max_position_embeddings=256,
            use_tidar=True,
            block_size=config_dict['block_size'],
            clean_ratio=config_dict['clean_ratio']
        )
        
        model = TiDARQwen2ForCausalLM(config)
        params = sum(p.numel() for p in model.parameters())
        
        print(f"{config_dict['name']}:")
        print(f"  - Parameters: {params:,}")
        print(f"  - Block size: {config.block_size}")
        print(f"  - Clean ratio: {config.clean_ratio}")
    
    print("✓ Custom configuration example completed!\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("TiDAR-Qwen2 Usage Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_tidar_format()
        example_attention_mask()
        example_model_modes()
        example_generation()
        example_custom_config()
        
        print("=" * 60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. TiDAR models can handle both standard and TiDAR format inputs")
        print("2. Hybrid attention masks are created automatically")
        print("3. Models can be configured with different parameters")
        print("4. TiDAR mode can be toggled on/off")
        print("5. Standard generation is supported (TiDAR generation coming soon)")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
