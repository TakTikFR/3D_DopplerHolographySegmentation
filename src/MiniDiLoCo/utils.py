import torch
import torch.distributed as dist

def setup(BACKEND, RANK, WORLD_SIZE):
    dist.init_process_group(backend=BACKEND, rank=RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(RANK)

def cleanup():
    dist.destroy_process_group()