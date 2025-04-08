import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from IPython import get_ipython
from queue import Queue
import time
import uuid
import sys
import logging
from io import StringIO
from IPython.core.magic import register_cell_magic

_PROCESS_POOL = []
_COMMAND_QUEUES = []
_RESULT_QUEUES = []
_INITIALIZED = False
_WORLD_SIZE = 0

def _worker_process(rank, world_size, command_queue, result_queue, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    result_queue.put({"status": "init", "message": f"Worker {rank}/{world_size} initialized on GPU {rank} on {master_addr}:{master_port}"})
    
    local_namespace = {"rank": rank, "world_size": world_size}
    exec("import torch", local_namespace)
    exec("import torch.nn as nn", local_namespace)
    exec("import torch.distributed as dist", local_namespace)
    exec("import torch.multiprocessing as mp", local_namespace)
    exec("from torch.nn.parallel import DistributedDataParallel as DDP", local_namespace)
    
    while True:
        try:
            cmd, cmd_id = command_queue.get()
            
            if cmd == "EXIT":
                break

            output_buffer = StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            
            try:
                exec(cmd, local_namespace)
                stdout_content = output_buffer.getvalue()
                result = {
                    "status": "success", 
                    "output": stdout_content, 
                    "cmd_id": cmd_id,
                    "rank": rank
                }
            except Exception as e:
                stdout_content = output_buffer.getvalue()
                result = {
                    "status": "error", 
                    "error": str(e), 
                    "output": stdout_content,
                    "cmd_id": cmd_id,
                    "rank": rank
                }
            finally:
                sys.stdout = original_stdout
                
            result_queue.put(result)
            
        except Exception as e:
            result_queue.put({
                "status": "error", 
                "error": f"Worker error: {str(e)}", 
                "cmd_id": cmd_id,
                "rank": rank
            })
    
    dist.destroy_process_group()
    result_queue.put({"status": "exit", "message": f"Worker {rank} exiting"})

def init_multigpus_repl(num_gpus=None, master_addr = "localhost", master_port = "12355"):
    global _PROCESS_POOL, _COMMAND_QUEUES, _RESULT_QUEUES, _INITIALIZED, _WORLD_SIZE
    
    if _INITIALIZED:
        print("Multi-GPU REPL already initialized")
        return
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 0:
        raise ValueError("No GPUs available")
    
    logging.info(f"Initializing multi-GPU REPL with {num_gpus} GPUs...")
    _WORLD_SIZE = num_gpus
    
    for _ in range(num_gpus):
        _COMMAND_QUEUES.append(mp.Queue())
        _RESULT_QUEUES.append(mp.Queue())
    
    for rank in range(num_gpus):
        p = mp.Process(
            target=_worker_process,
            args=(
                rank, 
                num_gpus, 
                _COMMAND_QUEUES[rank], 
                _RESULT_QUEUES[rank],
                master_addr,
                master_port,
            )
        )
        p.start()
        _PROCESS_POOL.append(p)
    
    for _ in range(num_gpus):
        result = _RESULT_QUEUES[_].get()
        if result["status"] == "init":
            print(result["message"])
    
    _INITIALIZED = True
    logging.info("Multi-GPU REPL environment ready")

def execute_on_gpus(code):
    global _INITIALIZED, _WORLD_SIZE
    
    if not _INITIALIZED:
        raise RuntimeError("Multi-GPU REPL not initialized. Call init_multigpus_repl() first.")
    
    cmd_id = str(uuid.uuid4())
    
    for rank in range(_WORLD_SIZE):
        _COMMAND_QUEUES[rank].put((code, cmd_id))
    
    results = []
    for rank in range(_WORLD_SIZE):
        result = _RESULT_QUEUES[rank].get()
        results.append(result)
    
    results.sort(key=lambda r: r.get("rank", 0))
    
    had_errors = False
    for result in results:
        if result["status"] == "error":
            logging.error(f"[GPU {result['rank']}] Error: {result['error']}")
        else:
            if result.get("output") and result["output"].strip():
                print(f"[GPU {result['rank']}] {result['output'].strip()}")

@register_cell_magic
def multigpus(line, cell):
    if not _INITIALIZED:
        num_gpus = int(line.strip()) if line.strip() else None
        init_multigpus_repl(num_gpus)
    
    return execute_on_gpus(cell)