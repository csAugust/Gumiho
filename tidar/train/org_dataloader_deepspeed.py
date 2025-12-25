import json
import math
import os
import random
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm.auto import tqdm


def list_files(paths):
    """Recursively list all files in a directory"""
    datapath = []
    for path in paths:
        for root, directories, files in os.walk(path, followlinks=True):
            for file in files:
                file_path = os.path.join(root, file)
                datapath.append(file_path)
    return datapath


class TidarCustomDataset(Dataset):
    """
    Dataset for TiDAR training.
    
    Generates TiDAR-style data from input_ids:
    - input_ids: [clean_tokens, MASK_tokens] (2S length)
    - labels: [shifted_clean_tokens + ignore, original_tokens] (2S length)
    - attention_mask: special 2S x 2S structure
    - position_ids: [0..S-1, 0..S-1] (restarting for masked region)
    """
    
    def __init__(self, datapath, tokenizer, max_len=512):
        self.data = datapath
        self.tokenizer = tokenizer
        # Since we'll double the sequence, actual limit is max_len // 2
        self.max_len = max_len // 2
        self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else 151643

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        
        # Load original sequence (limit to S tokens)
        original_input_ids = data['input_ids'][:self.max_len]
        original_loss_mask = data['loss_mask'][:self.max_len]  # Load loss_mask
        seq_length = len(original_input_ids)  # S
        
        if isinstance(original_input_ids, list):
            original_input_ids = torch.tensor(original_input_ids, dtype=torch.long)
        if isinstance(original_loss_mask, list):
            original_loss_mask = torch.tensor(original_loss_mask, dtype=torch.long)

        # 1. Construct input_ids: [clean_tokens, MASK_tokens] (2S length)
        clean_tokens = original_input_ids  # [t1, t2, ..., tS]
        masked_tokens = torch.full((seq_length,), self.mask_token_id, dtype=torch.long)  # [MASK, MASK, ..., MASK]
        input_ids = torch.cat([clean_tokens, masked_tokens])  # Length 2S
        
        # 2. Construct labels: [AR_labels, Diffusion_labels] (2S length)
        # AR labels: shifted by 1 (for next-token prediction), last position is ignored
        ar_labels = torch.cat([
            original_input_ids[1:],  # [t2, t3, ..., tS]
            torch.tensor([0], dtype=torch.long)  # ignore last position
        ])
        # Diffusion labels: original tokens (for denoising)
        diffusion_labels = original_input_ids  # [t1, t2, ..., tS]
        labels = torch.cat([ar_labels, diffusion_labels])  # Length 2S
        
        # 3. Construct loss_mask: [AR_loss_mask, Diffusion_loss_mask] (2S length)
        # AR loss_mask: shifted by 1 to match AR labels, last position is 0
        ar_loss_mask = torch.cat([
            original_loss_mask[:-1],
            torch.tensor([0], dtype=torch.long)  # No loss for last position
        ])
        loss_mask = torch.cat([ar_loss_mask, ar_loss_mask])  # Length 2S
        
        # 4. Construct position_ids: [0..S-1, 0..S-1] (restarting for masked region)
        position_ids = torch.cat([
            torch.arange(seq_length, dtype=torch.long),
            torch.arange(seq_length, dtype=torch.long)
        ])  # Length 2S
        
        # 5. Attention mask will be created in collator (needs to be 2S x 2S)
        # For now, just mark valid positions
        attention_mask = torch.ones(2 * seq_length, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "seq_length": seq_length,  # Store S for creating attention mask later
        }


class TidarDataCollatorWithPadding:
    """Collate function for batching TiDAR data with special 2S x 2S attention mask"""
    
    def __init__(self, pad_token_id=0, block_size=8):
        self.pad_token_id = pad_token_id
        self.block_size = block_size
    
    def create_tidar_attention_mask(self, seq_length):
        """
        Create TiDAR attention mask with structure:
        - Clean -> Clean (top-left): Causal
        - Masked -> Clean (bottom-left): Full
        - Masked -> Masked (bottom-right): Block-wise Bidirectional
        - Clean -> Masked (top-right): Zero
        
        Args:
            seq_length: S (half of the total 2S sequence)
        
        Returns:
            attention_mask: [2S, 2S] tensor
        """
        total_length = 2 * seq_length
        mask = torch.zeros(total_length, total_length, dtype=torch.long)
        
        # Top-left: Clean -> Clean (Causal mask)
        # Position i can attend to position j if j <= i
        for i in range(seq_length):
            mask[i, :i+1] = 1

        for i in range(seq_length, total_length, self.block_size):
            block_end = min(i + self.block_size, total_length)
            # Bottom-right:
            # Within each block, tokens can attend to each other (bidirectional)
            mask[i:block_end, i:block_end] = 1
            # Bottom-left:
            if i > seq_length: # start from second block
                mask[i:block_end, :i-seq_length] = 1
        
        return mask
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Find max sequence length (note: this is 2S)
        max_length = max(len(item['input_ids']) for item in features)
        max_seq_length = max_length // 2  # This is S
        
        batch_input_ids = []
        batch_labels = []
        batch_loss_mask = []
        batch_position_ids = []
        batch_attention_mask = []
        batch_seq_lengths = []
        
        for item in features:
            length = len(item['input_ids'])  # Current 2S
            seq_length = item['seq_length']  # Current S
            pad_length = max_length - length
            
            # Pad input_ids
            input_ids = torch.cat([
                item['input_ids'],
                torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            ])
            batch_input_ids.append(input_ids)
            
            # Pad labels
            labels = torch.cat([
                item['labels'],
                torch.full((pad_length,), -100, dtype=torch.long)  # -100 is ignore index
            ])
            batch_labels.append(labels)
            
            # Pad loss_mask
            loss_mask = torch.cat([
                item['loss_mask'],
                torch.zeros(pad_length, dtype=torch.long)  # Padded positions don't contribute to loss
            ])
            batch_loss_mask.append(loss_mask)
            
            # Pad position_ids
            position_ids = torch.cat([
                item['position_ids'],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch_position_ids.append(position_ids)
            
            # Create 2D attention mask for this sequence
            # Start with the special TiDAR structure
            attn_mask = self.create_tidar_attention_mask(seq_length)
            
            # Pad to max_length x max_length
            if length < max_length:
                padded_mask = torch.zeros(max_length, max_length, dtype=torch.long)
                padded_mask[:length, :length] = attn_mask
                attn_mask = padded_mask
            
            batch_attention_mask.append(attn_mask)
            batch_seq_lengths.append(seq_length)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "loss_mask": torch.stack(batch_loss_mask),
            "position_ids": torch.stack(batch_position_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "seq_lengths": torch.tensor(batch_seq_lengths),
        }
    

# datapath = list_files(args.data_dir)
# traindatapath = datapath[:int(len(datapath) * 1)]
# # testdatapath = datapath[int(len(datapath) * 0.95):]

# train_ds = CustomDataset(
#     traindatapath, 
#     tokenizer, 
#     max_len=args.max_len,
# )
# print("Training data size: ", len(train_ds))

# model_engine, optimizer, train_loader, _ = deepspeed.initialize(
#     args=args,
#     model=model,
#     model_parameters=model.parameters(),
#     training_data=train_ds,
#     collate_fn=DataCollatorWithPadding(
#         pad_token_id=tokenizer.pad_token_id,
#         block_size=args.tidar_block_size
#     )
# )
