import os
import argparse
from datasets import WaymoDataset, waymo_collate_fn, create_idx, cache_data
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the waymo dataset and cache the scene tensors for fast acess.")

    parser.add_argument('--tfrecord_dir', type=str, required=True, help='Path to the tfrecord directory')
    parser.add_argument('--idx_dir', type=str, required=True, help='Path to the tfrecord index directory')
    parser.add_argument('--cache_dir', type=str, required=True, help='Path to the tensor cache directory')

    args = parser.parse_args()

    if not os.path.exists(args.tfrecord_dir):
        raise FileNotFoundError(f"tfrecord directory does not exist: {args.tfrecord_dir}")

    os.makedirs(args.idx_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    create_idx(args.tfrecord_dir, args.idx_dir)

    dataset = WaymoDataset(args.tfrecord_path, args.idx_path)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: waymo_collate_fn(x))

    cache_data(dataloader, args.cache_dir)