import sys
sys.path.append('core')

import time
import argparse
import torch
from raft_stereo import RAFTStereo

def benchmark_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default="models/raftstereo-middlebury.pth")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--use_refinement', action='store_true', help="是否启用 refinement head")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=device), strict=False)
    model = model.module
    model.to(device)
    
    # If not using mixed precision for benchmarking baseline, we do not convert anything
    # And we turn off explicit mixed precision arg override if the user didn't ask for it
    # We will adjust args.mixed_precision dynamically based on the command line
    # (Removed args.mixed_precision = True from above)

    model.eval()

    # Input Tensors
    image1 = torch.rand(1, 3, 384, 1000).to(device)
    image2 = torch.rand(1, 3, 384, 1000).to(device)

    is_fp16 = args.mixed_precision
    dtype_to_use = torch.float16 if is_fp16 else torch.float32

    # Warmup
    print(f"Warming up for 10 iterations (FP16/Mixed Precision: {is_fp16})...")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype_to_use, enabled=is_fp16):
            for _ in range(10):
                _ = model(image1, image2, iters=args.valid_iters, test_mode=True)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Benchmark
    num_runs = 50
    print(f"Running inference for {num_runs} iterations...")
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype_to_use, enabled=is_fp16):
            for _ in range(num_runs):
                _ = model(image1, image2, iters=args.valid_iters, test_mode=True)
                
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_inference_time_ms = total_time_ms / num_runs
    fps = 1000.0 / avg_inference_time_ms
    max_memory_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # In MB

    print("=== Inference Benchmark Results ===")
    print(f"Average Inference Time : {avg_inference_time_ms:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print(f"Max CUDA Memory Alloc  : {max_memory_alloc:.2f} MB")

if __name__ == '__main__':
    benchmark_inference()