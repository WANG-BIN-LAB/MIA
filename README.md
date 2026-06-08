# Dependencies
- python==3.11.10
- torch==2.1.0+cu118
- numpy==1.26.4
- tqdm==4.67.1
- pandas==2.2.3
- networkx==3.4.2
- argparse==1.4.0
- scipy==1.31.1
- scikit-learn==1.5.2
##  🛠 Installation
Run the following command to create and configure the environment:

```bash
# Create environment
conda create --name PyEnv python=3.11.10

# Activate environment
conda activate PyEnv

# Install dependency packages
pip install -r requirements.txt
```
## 🚀 Usage
```bash
GBT:

python -m source --multirun datasz=100p model=gbt dataset=ABIDE preprocess=non_mixup

ALTER:

python -m alter model=lrbgt dataset=ABIDE

CAGT:

python train.py --root_dir ./dataset/ABIDE_I --epochs 70 --batch-size 64 --dropout 0.2

