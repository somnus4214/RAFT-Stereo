#!/usr/bin/env bash
set -euo pipefail

cd ~/newRAFT

mkdir -p logs

echo "[$(date '+%F %T')] Start: raftstereo_refine_1000"
python train_stereo.py \
  --name raftstereo_refine_1000 \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --batch_size 1 \
  --train_iters 16 \
  --valid_iters 16 \
  --num_steps 1000 \
  --lr 1e-4 \
  --mixed_precision \
  --use_refinement \
  --refine_only \
  --train_dataset middlebury 2>&1 | tee logs/raftstereo_refine_1000.log

echo "[$(date '+%F %T')] Finished: raftstereo_refine_1000"

echo "[$(date '+%F %T')] Start: raftstereo_refine_edge_1000"
python train_stereo.py \
  --name raftstereo_refine_edge_1000 \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --batch_size 1 \
  --train_iters 16 \
  --valid_iters 16 \
  --num_steps 1000 \
  --lr 1e-4 \
  --mixed_precision \
  --use_refinement \
  --use_edge_loss \
  --refine_only \
  --lambda_ref 1.0 \
  --lambda_edge 0.2 \
  --train_dataset middlebury 2>&1 | tee logs/raftstereo_refine_edge_1000.log

echo "[$(date '+%F %T')] All jobs finished successfully."