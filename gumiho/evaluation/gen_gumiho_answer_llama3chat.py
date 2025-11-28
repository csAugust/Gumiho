# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --config_path scripts/eval_config.json
"""
import argparse
import json
import os
import sys
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from gumiho.evaluation.eval_config_loader import EvalConfigLoader
from accelerate.utils import set_seed
set_seed(0)

import time
from datetime import datetime, timedelta
import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from ..model.gumiho_model import GumihoModel
from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *

from loguru import logger
logger.remove()

# Initialize distributed training if needed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

def run_eval(args):
    """Run evaluation with configuration from args"""
    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)
    
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    # shuffled_ids = [q["question_id"] for q in questions]

    # Split the question file into `num_gpus` files
    assert args.num_gpus_total % args.num_gpus_per_model == 0
    use_ray = args.num_gpus_total // args.num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=args.num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (args.num_gpus_total // args.num_gpus_per_model)  # // 2
    ans_handles = []


    model = GumihoModel.from_pretrained(
        base_model_path=args.base_model_path,
        gumiho_model_path=args.gumiho_model_path,
        total_token=args.total_tokens,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        args=args
    )
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                questions[i: i + chunk_size],
                args.max_new_token,
                args.num_choices,
                args.num_gpus_per_model,
                args.max_gpu_memory,
                args.temperature,
                args
            )
        )

    if use_ray:
        results = ray.get(ans_handles)
        logger.info(f"{results=}")
        return sum(results) / len(results)


@torch.inference_mode()
def get_model_answers(
        model,
        questions,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, _ = model.gumihoGenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # stop_str = "</s>"
            # if stop_str and output.find(stop_str) > 0:
            #     output = output[: output.find(stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
    print('Warmup done')

    # questions=questions[6:]
    accept_length_list = []
    speed_list = []
    draft_time_list = []
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                # try:
                torch.cuda.synchronize()
                start_time = time.time()

                output_ids, new_token, idx, _ = model.gumihoGenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                
                idx = max(1, idx)
                accept_length_list.append(new_token/(idx))
                logger.info(f"accept_length (Current/Mean): {new_token/(idx):.3f}/{sum(accept_length_list)/len(accept_length_list):.3f}")
                # total_token = sum(accept_length)
                # logger.debug(f"{type(new_token)= }, {type(total_time)= }, {type(new_token/total_time)= }, {(new_token/total_time).dtype= }")
                speed_list.append(new_token/total_time)
                draft_time_list.append(total_time/(idx))
                logger.info(f"speed: {sum(speed_list)/len(speed_list):.3f}")
                logger.info(f"total_token:{new_token}, total_time:{total_time}")
                logger.info(f"draft_time:{sum(draft_time_list)/len(draft_time_list)*1000:.3f}ms")
                logger.warning(f" ")


                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        
    
    return sum(accept_length_list)/len(accept_length_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gumiho Evaluation')
    parser.add_argument('--config_path', type=str, default='scripts/eval_config.json',
                        help='Path to the evaluation configuration file')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    
    args = parser.parse_args()

    # Load configuration from the specified config file
    config_loader = EvalConfigLoader(args.config_path)
    evaluation_config = config_loader.get_evaluation_config()
    gumiho_params = config_loader.get_gumiho_params()
    distributed_config = config_loader.get_distributed_config()
    model_paths = config_loader.get_models_config()
    logging_config = config_loader.get_logging_config()

    # Dynamically add all configuration parameters to args
    for category, config_dict in [
        ('evaluation', evaluation_config),
        ('gumiho_params', gumiho_params),
        ('distributed', distributed_config),
        ('logging', logging_config)
    ]:
        for key, value in config_dict.items():
            setattr(args, key, value)
    
    # Handle model-specific paths
    model_name = getattr(args, 'model_name', 'l3_8b')
    if model_name in model_paths:
        model_config = model_paths[model_name]
        if 'gumiho_path' in model_config:
            setattr(args, 'gumiho_model_path', model_config['gumiho_path'])
        if 'base_model_path' in model_config:
            setattr(args, 'base_model_path', model_config['base_model_path'])
    
    logger.info(f"{args=}")
    logger.remove()

    # Initialize distributed training if needed
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"Initialized distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    log_path = os.path.join(f"{args.log_dir}", f"{args.model_name}", f"{args.bench_name}")
    os.makedirs(log_path, exist_ok=True)
    
    # Add rank to logger file for distributed training
    # Get current time in GMT+8 (UTC+8)
    gmt8_time = datetime.now() + timedelta(hours=8)
    time_str = gmt8_time.strftime("%Y%m%d_%H%M%S")
    
    logger_file = f"{time_str}"
    if world_size > 1:
        logger_file = f"{logger_file}_rank{local_rank}"
    
    logger.add(f"{log_path}/{logger_file}.log", level="DEBUG", mode="w", format="{message}")
    print(f"---------> Output to {log_path}/{logger_file}.log")

    args_dict = vars(args)
    logger.info("="*30)
    logger.info(f"Distributed: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("="*30)

    # Initialize Ray for multi-GPU if needed
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    # For distributed training, each rank processes a subset of questions
    if world_size > 1:
        question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
        questions = load_questions(question_file, args.question_begin, args.question_end)
        chunk_size = len(questions) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < world_size - 1 else len(questions)
        questions_subset = questions[start_idx:end_idx]
        
        logger.info(f"Rank {rank} processing {len(questions_subset)} questions ({start_idx} to {end_idx})")
        
        # Create a temporary question file for this rank
        temp_question_file = f"{question_file}.rank{rank}"
        with open(temp_question_file, 'w') as f:
            for q in questions_subset:
                f.write(json.dumps(q) + '\n')
        
        # Override question file and range for this rank
        args.question_file = temp_question_file
        args.question_begin = None
        args.question_end = None

    run_eval(args)
    
    # Clean up temporary question file
    if world_size > 1 and 'temp_question_file' in locals():
        try:
            os.remove(temp_question_file)
        except:
            pass
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()
