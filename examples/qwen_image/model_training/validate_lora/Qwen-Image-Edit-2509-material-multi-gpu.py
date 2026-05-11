import torch
import torch.multiprocessing as mp
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import json
import os
import argparse

def load_pipeline(gpu_id):
    """在指定 GPU 上加载 pipeline"""
    device = f"cuda:{gpu_id}"
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )
    pipe.load_lora(pipe.dit, "/upfs/user/lanchen/MaterialEdit/outputs/material_lora/epoch-9.safetensors")
    return pipe


def worker(gpu_id, num_gpus, data, data_root, output_dir):
    """每个 GPU 的工作函数"""
    # 分配数据
    chunk_size = (len(data) + num_gpus - 1) // num_gpus
    start_idx = gpu_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(data))
    my_data = data[start_idx:end_idx]
    
    if len(my_data) == 0:
        print(f"GPU {gpu_id}: No data to process")
        return
    
    print(f"GPU {gpu_id}: Processing {len(my_data)} samples (index {start_idx}-{end_idx})")
    
    # 加载 pipeline
    pipe = load_pipeline(gpu_id)
    
    # 处理数据
    for i, item in enumerate(my_data):
        prompt = item['prompt']
        image_path = item['image']
        
        image_info = [Image.open(os.path.join(data_root, image_path)).convert("RGB").resize((1024, 1024))]
        image = pipe(prompt, edit_image=image_info, seed=123, num_inference_steps=40, height=1024, width=1024)
        
        output_path = os.path.join(output_dir, image_path.replace("/", "_"))
        image.save(output_path)
        print(f"GPU {gpu_id} [{i+1}/{len(my_data)}]: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--test_json", type=str, default="/upfs/user/lanchen/MaterialEdit/data/val.json")
    parser.add_argument("--data_root", type=str, default="/upfs/user/lanchen/render/data/cg_wh_300")
    parser.add_argument("--output_dir", type=str, default="/upfs/user/lanchen/MaterialEdit/outputs/material_lora/val")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    args = parser.parse_args()
    
    # 检测可用 GPU 数量
    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    # 加载数据
    with open(args.test_json, 'r') as f:
        data = json.load(f)
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"Total samples: {len(data)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 启动多进程
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, num_gpus, data, args.data_root, args.output_dir))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("All done!")


if __name__ == "__main__":
    main()

