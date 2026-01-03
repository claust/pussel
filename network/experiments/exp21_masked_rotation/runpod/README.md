# RunPod Deployment for Exp21

## Quick Start

### 1. Prepare Package (local machine)
```bash
cd network/experiments/exp21_masked_rotation
./runpod/prepare_package.sh
```

### 2. Upload to RunPod
```bash
scp -P 30611 -i ~/.ssh/id_ed25519 \
    ../../../runpod_package/runpod_training.tar.gz \
    root@38.80.152.77:/workspace/
```

### 3. Run Training (on RunPod)
```bash
ssh -p 30611 -i ~/.ssh/id_ed25519 root@38.80.152.77
cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh
```

### 4. Download Results
```bash
scp -P 30611 -i ~/.ssh/id_ed25519 \
    'root@38.80.152.77:/workspace/outputs/*' \
    ./outputs/
```

## Expected Training Time

- RTX 4090: ~3 hours (50 epochs, 10800 training puzzles)
- Same as exp20 since architecture is similar

## Key Differences from Exp20

1. **Dataset returns masks**: Each piece has a mask derived from black background
2. **Masked rotation correlation**: Only compares puzzle content, ignores background
3. **Same dataset**: Reuses exp20's realistic_4x4 dataset (no regeneration)

## Success Criteria

| Metric | Exp20 | Target |
|--------|-------|--------|
| Cell accuracy | 73% | >= 70% |
| Rotation accuracy | 25% | > 50% |
| Rotation train-test gap | 70% | < 20% |
