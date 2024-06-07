"""
Helpers for distributed training.
"""

import io
import multiprocessing
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set the CUDA visible devices

    # Initialize the process group
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = hostname
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"  # Set the world size to 1 for single process
    os.environ["MASTER_PORT"] = "29501"  # Set a fixed master port

    th.distributed.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across processes.
    """
    chunk_size = 2**30  # Set the chunk size
    if os.getpid() == 0:  # The main process
        with open(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        for i in range(0, len(data), chunk_size):
            for proc in range(1, multiprocessing.cpu_count()):  # Start from process 1
                multiprocessing.Process(target=send_data, args=(proc, data[i : i + chunk_size])).start()
    else:  # Child processes
        num_chunks = multiprocessing.Queue().get()
        data = b''
        for _ in range(num_chunks):
            data += multiprocessing.Queue().get()

    return th.load(io.BytesIO(data), **kwargs)

def send_data(proc, data_chunk):
    multiprocessing.Queue().put(data_chunk)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
