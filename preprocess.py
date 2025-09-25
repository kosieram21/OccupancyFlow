import os
import argparse
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from datasets import WaymoDataset, waymo_collate_fn, create_idx, cache_data

def worker(tfrecord_dir, idx_dir, cache_dir, worker_id, num_workers):
    dataset = WaymoDataset(tfrecord_dir, idx_dir, worker_id, num_workers)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=waymo_collate_fn)
    worker_cache_dir = os.path.join(cache_dir, f"worker_{worker_id}")
    cache_data(dataloader, worker_cache_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the waymo dataset and cache the scene tensors for fast acess.")

    parser.add_argument('--tfrecord_dir', type=str, required=True, help='Path to the tfrecord directory')
    parser.add_argument('--idx_dir', type=str, required=True, help='Path to the tfrecord index directory')
    parser.add_argument('--cache_dir', type=str, required=True, help='Path to the tensor cache directory')

    args = parser.parse_args()

    if not os.path.exists(args.tfrecord_dir):
        raise FileNotFoundError(f"tfrecord directory does not exist: {args.tfrecord_dir}")

    if not os.path.exists(args.idx_dir):
        create_idx(args.tfrecord_dir, args.idx_dir)

    num_workers = 1#torch.get_num_threads()

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(args.tfrecord_dir, args.idx_dir, args.cache_dir, worker_id, num_workers))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
