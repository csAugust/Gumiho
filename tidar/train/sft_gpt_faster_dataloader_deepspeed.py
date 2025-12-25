import json
import math
import os
import random
from typing import List, Dict, Any

from datasets import load_dataset
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def convert_token_ids(old_token_ids, old_tokenizer, new_tokenizer, skip_special_tokens=False):
    """
    将token ids从旧词表转换为新词表。
    
    参数:
    old_token_ids (list or np.ndarray): 原始token ids
    old_tokenizer: 原始tokenizer
    new_tokenizer: 新的tokenizer
    skip_special_tokens (bool): 是否跳过特殊token
    
    返回:
    list: 转换后的token ids
    """
    # 将旧token ids解码为文本
    if isinstance(old_token_ids, np.ndarray):
        old_token_ids = old_token_ids.tolist()
    
    # 解码为文本
    text = old_tokenizer.decode(old_token_ids, skip_special_tokens=skip_special_tokens)
    
    # 用新tokenizer编码
    new_token_ids = new_tokenizer.encode(text, add_special_tokens=False)
    if (len(new_token_ids) != len(old_token_ids)):
        print(f"Warning: new length {len(new_token_ids)} != old length {len(old_token_ids)}")
    
    return new_token_ids


def load_all_datasets_recursively(main_folder_path):
    """
    递归加载指定主文件夹及其所有子文件夹中通过 save_to_disk 保存的数据集。

    参数:
    main_folder_path (str): 包含多个数据集子文件夹（可能在不同层级）的主文件夹路径。

    返回:
    datasets.Dataset or datasets.DatasetDict: 如果可以合并，则返回合并后的 Dataset 对象；
                                              否则返回一个包含所有加载的数据集的字典，
                                              键是数据集的相对路径，值是 Dataset 对象。
                                              如果未找到任何数据集，则返回 None。
    """
    loaded_datasets_list = []
    loaded_datasets_dict = {}

    print(f"正在从以下主文件夹及其子文件夹递归加载数据集: {main_folder_path}")

    if not os.path.isdir(main_folder_path):
        print(f"错误: 文件夹 {main_folder_path} 不存在。")
        return None

    for dirpath, dirnames, filenames in os.walk(main_folder_path):
        # 检查当前目录 dirpath 是否为一个 datasets 保存的目录
        # (通过检查是否存在 dataset_info.json 文件)
        dataset_info_path = os.path.join(dirpath, "dataset_info.json")

        if os.path.exists(dataset_info_path):
            # 如果找到了 dataset_info.json，那么 dirpath 就是一个数据集目录
            relative_path = os.path.relpath(dirpath, main_folder_path)
            print(f"  发现潜在的数据集目录: {relative_path} (位于: {dirpath})")
            try:
                dataset = load_from_disk(dirpath)
                loaded_datasets_list.append(dataset)
                # 使用相对路径作为键，以避免同名子文件夹在不同路径下引发的键冲突
                loaded_datasets_dict[relative_path] = dataset
                print(f"    成功加载: {relative_path}, 条目数: {len(dataset)}")

                # 重要: 一旦我们识别并加载了一个数据集目录 (dirpath)，
                # 我们通常不希望 os.walk 再继续深入这个已识别的数据集目录内部去寻找其他数据集。
                # 因为一个 save_to_disk 的目录本身就是数据集的边界。
                # 清空 dirnames 可以阻止 os.walk 进入当前 dirpath 的子目录。
                dirnames[:] = []  # 清空列表，原地修改

                break

            except Exception as e:
                print(f"    加载 {relative_path} (位于: {dirpath}) 失败: {e}")

    if not loaded_datasets_list:
        print("在指定目录及其子目录中未能加载任何数据集。")
        return None

    # 尝试合并所有加载的数据集
    try:
        processed_datasets_for_concatenation = []
        for i, ds_obj in enumerate(loaded_datasets_list):
            dataset_key_name = list(loaded_datasets_dict.keys())[i]  # 获取对应的字典键名
            if isinstance(ds_obj, Dataset):
                processed_datasets_for_concatenation.append(ds_obj)
            elif isinstance(ds_obj, DatasetDict):
                if 'train' in ds_obj:
                    print(f"  从 {dataset_key_name} 中提取 'train' split 进行合并。")
                    processed_datasets_for_concatenation.append(ds_obj['train'])
                else:
                    first_split_name = next(iter(ds_obj.keys()), None)
                    if first_split_name:
                        print(f"  警告: {dataset_key_name} 是一个 DatasetDict。将使用第一个 split '{first_split_name}' 进行合并。")
                        processed_datasets_for_concatenation.append(ds_obj[first_split_name])
                    else:
                        print(f"  警告: DatasetDict {dataset_key_name} 为空，跳过合并。")
            else:
                print(f"  警告: {dataset_key_name} 是未知类型 ({type(ds_obj)})，跳过合并。")

        if not processed_datasets_for_concatenation:
            print("没有可用于合并的 Dataset 对象。将返回已加载数据集的字典。")
            return loaded_datasets_dict

        print("\n正在尝试合并所有已加载的数据集...")
        combined_dataset = concatenate_datasets(processed_datasets_for_concatenation)
        print(f"所有数据集已成功合并。总条目数: {len(combined_dataset)}")
        return combined_dataset
    except Exception as e:
        print(f"合并数据集失败: {e}")
        print("这可能是因为数据集的特征不完全一致。将返回已加载数据集的字典。")
        return loaded_datasets_dict


def data_concatenation(seq_lengths, max_seq_length, block_size=500):
    """
    将句子长度列表按块处理，每块内拼接句子使其长度尽可能接近 max_seq，但不超过 max_seq。

    参数:
    seq_lengths (list[int]): 每个句子的长度列表
    max_seq_length (int): 允许的最大拼接长度
    block_size (int): 每个块的大小，即每次处理的句子数量

    返回:
    result (list[list[int]]): 每个子列表是拼接后的句子长度
    result_indices (list[list[int]]): 每个子列表是对应拼接句子的原始索引
    """

    def process_block(block, max_seq_length):
        """
        处理一个块中的句子长度，使用贪心算法选择尽量接近 max_seq_length 的组合。

        参数:
        block (np.ndarray): 当前块中的句子长度
        max_seq_length (int): 允许的最大拼接长度

        返回:
        block_result (list[list[int]]): 当前块中的拼接结果
        block_indices_result (list[list[int]]): 当前块中的拼接结果对应的原始索引
        """

        block_result = []
        block_indices_result = []

        # 将块中的句子长度按从大到小排序
        sorted_indices = np.argsort(block)[::-1]
        sorted_lengths = block[sorted_indices]

        # discard overflow data
        overflow_count = (sorted_lengths > max_seq_length).sum()
        sorted_lengths = sorted_lengths[overflow_count:]
        sorted_indices = sorted_indices[overflow_count:]
        if overflow_count > 0:
            print(f"Removed {overflow_count} elements exceeding max_length {max_seq_length}")

        current_sum = 0
        current_combination = []
        current_indices = []

        # 贪心选择尽量接近 max_seq_length 的组合
        for i in range(sorted_lengths.size):
            length = sorted_lengths[i]

            if current_sum + length <= max_seq_length:
                current_combination.append(length)
                current_indices.append(sorted_indices[i])
                current_sum += length
            else:
                # add current block into result
                block_result.append(current_combination)
                block_indices_result.append(current_indices)
                # add current data into new block
                current_sum = length
                current_combination = [length]
                current_indices = [sorted_indices[i]]

        # 将最后的组合加入结果
        if current_combination:
            block_result.append(current_combination)
            block_indices_result.append(current_indices)

        return block_result, block_indices_result

    seq_lengths = np.array(seq_lengths)
    result = []
    result_indices = []

    # 分块处理句子长度列表
    for start in range(0, len(seq_lengths), block_size):
        end = min(start + block_size, len(seq_lengths))
        block = seq_lengths[start:end]
        block_result, block_indices_result = process_block(block, max_seq_length)

        # 将每块处理后的结果加入最终结果
        for combination, indices in zip(block_result, block_indices_result):
            result.append(combination)
            result_indices.append([start + idx for idx in indices])

    return result, result_indices


class TiDARDeepspeedDataset(Dataset):
    """
    TiDAR数据集，适配Deepspeed训练。
    
    支持两种模式：
    1. packing模式：将多个短序列拼接成一个长序列
    2. 非packing模式：直接处理单个序列
    
    数据格式适配TiDAR模型的2S结构：
    - input_ids: [clean_tokens, MASK_tokens] (2S length)
    - labels: [AR_labels (shifted), Diffusion_labels (unshifted)] (2S length)
    - loss_mask: [AR_loss_mask, Diffusion_loss_mask] (2S length)
    - position_ids: [0..S-1, 0..S-1] (2S length)
    - seq_length: S (half of the total sequence)
    """
    
    def __init__(self, seed, args, jsonl_path, tokenizer, mask_token_id, old_tokenizer_path=None):
        self.args = args
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.pad_id = getattr(tokenizer, "eos_token_id", getattr(tokenizer, "pad_token_id", None))
        
        # 加载原始tokenizer（如果提供了路径）
        self.old_tokenizer = None
        self.enable_token_conversion = False
        if old_tokenizer_path is not None:
            print(f"正在加载原始tokenizer: {old_tokenizer_path}")
            self.old_tokenizer = AutoTokenizer.from_pretrained(old_tokenizer_path, trust_remote_code=True)
            self.enable_token_conversion = True
            print(f"原始tokenizer词表大小: {len(self.old_tokenizer)}")
            print(f"新tokenizer词表大小: {len(self.tokenizer)}")
            print("启用token id转换功能")
        
        # 加载数据集
        self.raw_datasets = load_all_datasets_recursively(jsonl_path)
        
        num_docs = len(self.raw_datasets)
        print(f"数据集总条目数: {num_docs}")

        # 提取序列长度
        temp_ds = self.raw_datasets.map(
            lambda batch: {
                "seq_length": [
                    data_item["seq_lengths"]
                    for data_item in batch["data_info"]
                ]
            },
            batched=True,
            batch_size=10000,
            num_proc=128,
            remove_columns=['text']
        )
        self.seq_lengths = temp_ds["seq_length"]
        
        # 如果启用packing，进行数据打包
        if self.args.sft_packing:
            _, self.doc_concate_indexes = data_concatenation(
                self.seq_lengths,
                max_seq_length=args.seq_length
            )
            num_docs = len(self.doc_concate_indexes)
            print(f"启用packing，打包后的数据条目数: {num_docs}")
        
        self.num_docs = num_docs

        # 随机打乱文档顺序
        np_rng = np.random.RandomState(seed)
        self.doc_index = np_rng.permutation(num_docs)
        
        print(f"数据集大小: {len(self.doc_index)}")

    def __len__(self):
        return len(self.doc_index)

    def __getitems__(self, indices):
        """处理批量索引，返回批量样本"""
        if isinstance(indices, int):
            indices = [indices]
        return [self.__getitem__(idx) for idx in indices]

    def __getitem__(self, idx):
        doc_idx = self.doc_index[idx]
        
        if self.args.sft_packing:
            # Packing模式：拼接多个序列
            concate_indexes = self.doc_concate_indexes[int(doc_idx)]

            concate_token_id = []
            concate_loss_mask = []
            concate_label = []
            concate_seq_lengths = [0]

            for concate_index in concate_indexes:
                doc = self.raw_datasets[int(concate_index)]["data_info"]
                sample, loss_mask, seq_lengths = doc["docs"], doc["training_loss_mask"], doc["seq_lengths"]
                tokens = sample[:-1]
                labels = sample[1:]
                
                # 如果启用token id转换，将tokens和labels转换为新词表
                if self.enable_token_conversion:
                    tokens = convert_token_ids(tokens, self.old_tokenizer, self.tokenizer)
                    labels = convert_token_ids(labels, self.old_tokenizer, self.tokenizer)
                
                concate_token_id = concate_token_id + tokens
                concate_label = concate_label + labels
                concate_loss_mask = concate_loss_mask + loss_mask
                concate_seq_lengths.append(concate_seq_lengths[-1] + seq_lengths)

            # 确保长度不超过seq_length
            assert len(concate_token_id) <= self.args.seq_length, \
                f"len(concate_token_id) = {len(concate_token_id)} which should be <= {self.args.seq_length}"
            
            pad_length = self.args.seq_length - len(concate_token_id)
            if pad_length > 0:
                pad_ids = np.array([self.pad_id] * pad_length)
                concate_token_id = np.concatenate([concate_token_id, pad_ids])
                concate_label = np.concatenate([concate_label, pad_ids])
                concate_loss_mask = np.concatenate([concate_loss_mask, np.array([0] * pad_length)])
                concate_seq_lengths.append(concate_seq_lengths[-1] + pad_length)
            
            # 转换为TiDAR格式
            tokens = concate_token_id
            seq_length = len(tokens)
            
        else:
            # 非packing模式：处理单个序列
            doc = self.raw_datasets[int(doc_idx)]["data_info"]
            sample, loss_mask, seq_lengths = doc["docs"], doc["training_loss_mask"], doc["seq_lengths"]
            tokens = sample[:-1]
            labels = sample[1:]
            
            # 如果启用token id转换，将tokens和labels转换为新词表
            if self.enable_token_conversion:
                tokens = convert_token_ids(tokens, self.old_tokenizer, self.tokenizer)
                labels = convert_token_ids(labels, self.old_tokenizer, self.tokenizer)
                # 转换后需要调整loss_mask长度
                if len(tokens) != len(loss_mask):
                    # 如果转换后的长度不同，需要调整loss_mask
                    # 假设转换是连续的，保留原始loss_mask的模式
                    if len(tokens) < len(loss_mask):
                        loss_mask = loss_mask[:len(tokens)]
                    else:
                        # 扩展loss_mask，使用1填充（新增的tokens不计算loss）
                        loss_mask = np.concatenate([loss_mask, np.zeros(len(tokens) - len(loss_mask))])
            
            # 填充到max_len
            if len(tokens) < self.args.max_len:
                pad_length = self.args.max_len - len(tokens)
                pad_ids = np.array([self.pad_id] * pad_length)
                tokens = np.concatenate([tokens, pad_ids])
                labels = np.concatenate([labels, pad_ids])
                loss_mask = np.concatenate([loss_mask, np.array([0] * pad_length)])
            
            seq_length = len(tokens)
        
        # 转换为TiDAR格式
        original_input_ids = torch.tensor(tokens, dtype=torch.long)
        original_loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        original_labels = torch.tensor(labels, dtype=torch.long)
        
        # 1. 构造input_ids: [clean_tokens, MASK_tokens] (2S长度)
        clean_tokens = original_input_ids  # [t1, t2, ..., tS]
        masked_tokens = torch.full((seq_length,), self.mask_token_id, dtype=torch.long)  # [MASK, MASK, ..., MASK]
        input_ids = torch.cat([clean_tokens, masked_tokens])  # 长度 2S
        
        # 2. 构造labels: [AR_labels, Diffusion_labels] (2S长度)
        # AR labels: 向后移动1位（用于下一个token预测），最后一个位置被忽略
        ar_labels = torch.cat([
            original_input_ids[1:],  # [t2, t3, ..., tS]
            torch.tensor([0], dtype=torch.long)  # 忽略最后一个位置
        ])
        # Diffusion labels: 原始tokens（用于去噪）
        diffusion_labels = original_input_ids  # [t1, t2, ..., tS]
        labels = torch.cat([ar_labels, diffusion_labels])  # 长度 2S
        
        # 3. 构造loss_mask: [AR_loss_mask, Diffusion_loss_mask] (2S长度)
        # AR loss_mask: 向后移动1位以匹配AR labels，最后一个位置为0
        ar_loss_mask = torch.cat([
            original_loss_mask[:-1],
            torch.tensor([0], dtype=torch.long)  # 最后一个位置没有loss
        ])
        loss_mask = torch.cat([ar_loss_mask, ar_loss_mask])  # 长度 2S
        
        # 4. 构造position_ids: [0..S-1, 0..S-1] (2S长度)
        position_ids = torch.cat([
            torch.arange(seq_length, dtype=torch.long),
            torch.arange(seq_length, dtype=torch.long)
        ])  # 长度 2S
        
        # 5. Attention mask将在collator中创建（需要2S x 2S）
        # 这里只标记有效位置
        attention_mask = torch.ones(2 * seq_length, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
        }


class TiDARDataCollator:
    """用于批处理TiDAR数据的collate函数，带有特殊的2S x 2S attention mask"""
    
    def __init__(self, pad_token_id=0, block_size=8):
        self.pad_token_id = pad_token_id
        self.block_size = block_size
    
    def create_tidar_attention_mask(self, seq_length):
        """
        创建TiDAR attention mask，结构如下：
        - Clean -> Clean (左上): Causal
        - Masked -> Clean (左下): Full
        - Masked -> Masked (右下): Block-wise Bidirectional
        - Clean -> Masked (右上): Zero
        
        Args:
            seq_length: S (总长度2S的一半)
        
        Returns:
            attention_mask: [2S, 2S] tensor
        """
        total_length = 2 * seq_length
        mask = torch.zeros(total_length, total_length, dtype=torch.long)
        
        # 左上: Clean -> Clean (Causal mask)
        # 位置i可以attend到位置j如果j <= i
        for i in range(seq_length):
            mask[i, :i+1] = 1

        # 右下和左下: Masked区域
        for i in range(seq_length, total_length, self.block_size):
            block_end = min(i + self.block_size, total_length)
            # 右下:
            # 在每个block内，tokens可以相互attend（双向）
            mask[i:block_end, i:block_end] = 1
            # 左下:
            if i > seq_length:  # 从第二个block开始
                mask[i:block_end, :i-seq_length] = 1
        
        return mask
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找到最大序列长度（注意：这是2S）
        max_length = max(len(item['input_ids']) for item in features)
        max_seq_length = max_length // 2  # 这是S
        
        batch_input_ids = []
        batch_labels = []
        batch_loss_mask = []
        batch_position_ids = []
        batch_attention_mask = []
        batch_seq_lengths = []
        
        for item in features:
            length = len(item['input_ids'])  # 当前2S
            seq_length = item['seq_length'].item() if isinstance(item['seq_length'], torch.Tensor) else item['seq_length']  # 当前S，转换tensor为int
            pad_length = max_length - length
            
            # 填充input_ids
            input_ids = torch.cat([
                item['input_ids'],
                torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            ])
            batch_input_ids.append(input_ids)
            
            # 填充labels
            labels = torch.cat([
                item['labels'],
                torch.full((pad_length,), -100, dtype=torch.long)  # -100是忽略索引
            ])
            batch_labels.append(labels)
            
            # 填充loss_mask
            loss_mask = torch.cat([
                item['loss_mask'],
                torch.zeros(pad_length, dtype=torch.long)  # 填充位置不贡献loss
            ])
            batch_loss_mask.append(loss_mask)
            
            # 填充position_ids
            position_ids = torch.cat([
                item['position_ids'],
                torch.zeros(pad_length, dtype=torch.long)
            ])
            batch_position_ids.append(position_ids)
            
            # 为这个序列创建2D attention mask
            # 从特殊的TiDAR结构开始
            attn_mask = self.create_tidar_attention_mask(seq_length)
            
            # 填充到max_length x max_length
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


def build_train_valid_test_datasets(args, tokenizer, train_data_prefix, mask_token_id, valid_data_prefix=None, old_tokenizer_path=None):
    """
    构建训练、验证和测试数据集
    
    参数:
    args: 训练参数
    tokenizer: 新的tokenizer
    train_data_prefix: 训练数据路径
    mask_token_id: MASK token的ID
    valid_data_prefix: 验证数据路径（可选）
    old_tokenizer_path: 原始tokenizer路径（可选）。如果提供，将启用token id转换功能
    """
    assert train_data_prefix is not None

    train_dataset = TiDARDeepspeedDataset(
        seed=42,
        args=args,
        jsonl_path=train_data_prefix,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        old_tokenizer_path=old_tokenizer_path
    )
    return train_dataset, None, None


def train_valid_test_datasets_provider(args, tokenizer, train_jsonl_path, mask_token_id, valid_jsonl_path=None, old_tokenizer_path=None):
    """
    提供训练、验证和测试数据集
    
    参数:
    args: 训练参数
    tokenizer: 新的tokenizer
    train_jsonl_path: 训练数据路径
    mask_token_id: MASK token的ID
    valid_jsonl_path: 验证数据路径（可选）
    old_tokenizer_path: 原始tokenizer路径（可选）。如果提供，将启用token id转换功能。
                       默认为 '/mnt/user-ssd/chenzhiyang1/ckpts/sft_v2_edge_3b_1201_5heads_th_logitloss_reverse/hf/iter_0020000'
    """
    # 如果没有传入old_tokenizer_path，使用默认路径
    if old_tokenizer_path is None:
        old_tokenizer_path = "/mnt/user-ssd/chenzhiyang1/ckpts/sft_v2_edge_3b_1201_5heads_th_logitloss_reverse/hf/iter_0020000"
        print(f"使用默认的原始tokenizer路径: {old_tokenizer_path}")
    
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        args=args,
        tokenizer=tokenizer,
        train_data_prefix=train_jsonl_path,
        mask_token_id=mask_token_id,
        valid_data_prefix=valid_jsonl_path,
        old_tokenizer_path=old_tokenizer_path,
    )
    return train_ds, valid_ds, test_ds
