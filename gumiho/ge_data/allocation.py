import argparse
import copy
import os
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description='Data allocation for training data generation')
parser.add_argument('--outdir', type=str, default='./train_data', help='Output directory for training data')
parser.add_argument('--start', type=int, default=0, help='Start index for data range')
parser.add_argument('--end', type=int, default=67999, help='End index for data range')
parser.add_argument('--num_processes', type=int, default=2, help='Number of parallel processes')
parser.add_argument('--gpus', type=str, default='0,1,2,3|4,5,6,7', help='GPU assignments for processes (comma-separated, pipe-separated for multiple processes)')
parser.add_argument('--model_path', type=str, default='/mnt/bos-text/models/hf_models/Llama-3.1-8B-Instruct', help='Path to the model')
parser.add_argument('--dataset_path', type=str, default='/mnt/user-ssd/chenzhiyang1/workspace/Datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json', help='Path to the dataset')
parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
parser.add_argument('--system_prompt', type=str, default='', help='System prompt for the model (if empty, uses default)')
args = parser.parse_args()

s = args.start
e = args.end
num_p = args.num_processes

# Parse GPU assignments
gpu_assignments = args.gpus.split('|')
gpus = []
for gpu_str in gpu_assignments:
    gpus.append([int(gpu.strip()) for gpu in gpu_str.split(',')])

outdir = '{}/{}/{}/{}_{}_mufp16'.format(args.outdir, args.model_path.split('/')[-1], args.dataset_path.split('/')[-1], s, e)


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    cuda_visible_str = ','.join(map(str, gpu_index))
    command = "CUDA_VISIBLE_DEVICES={} python3 gumiho/ge_data/ge_data_all_llama3.py --start={} --end={} --index={} --gpu_index {} --outdir {} --model_path {} --dataset_path {} --max_length {}".format(
        cuda_visible_str, start, end, index, gpu_index_str, outdir, args.model_path, args.dataset_path, args.max_length)
    
    # Add system_prompt if provided
    if args.system_prompt:
        command += " --system_prompt '{}'".format(args.system_prompt.replace("'", "'\"'\"'"))
    commands.append(command)

with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
