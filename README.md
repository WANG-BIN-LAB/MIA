# MIA
Experimental Settings
GBT
Hyperparameters: Batch size = 16 (64 in original paper, official released code uses 16), 200 epochs, 2-layer multi-head attention, lr=1e-4, weight decay=1e-4; train/val/test split = 7:1:2, 5 repeated runs.
Hardware: NVIDIA GeForce RTX 4090
Code Adjustment: Only revise data loading for our custom dataset.
ALTER
Hyperparameters: Batch size = 16, 200 epochs, L=2 nonlinear layers, M=4 attention heads, lr=1e-4, weight decay=1e-4; train/val/test split = 7:1:2, 10 repeated runs; adaptive random walk step K=16.
Hardware: Tesla V100
Code Adjustment: Complete missing source files and adapt data loading to our dataset.
CAGT
Hyperparameters: Batch size = 64, 70 epochs, Transformer with 2 layers & 8 heads, lr=1e-4, weight decay=1e-6, 10-fold cross validation; top-30 connections preserved as edges per node.
Hardware: GeForce RTX 3080 Ti
Code Adjustment: Move read_data.py & comm_utils.py to root folder; implement missing is_float() in utils.py; add required raw_name argument for ABIDEDataset initialization in train.py.
