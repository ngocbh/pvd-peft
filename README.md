# Pretraining and Fine-Tuning for Equivariant Graph Neural Network on Molecular Datasets

## How to use this code

### Install dependencies

Clone the repository:
```
git clone https://github.com/ngocbh/pvd-peft
cd pvd-peft
```

Create a virtual environment containing the dependencies and activate it:
```
conda env create -f environment.yml -n pvd
conda activate pvd
```

Install the package into the environment:
```
pip install -e .
```
### Pre-training on PCQM4Mv2

The model is pre-trained on the [PCQM4Mv2]() dataset, which contains over 3 million molecular structures at equilibrium. Run the following command to pre-train the architecture first. Note that this will download and pre-process the PCQM4Mv2 dataset when run for the first time, which can take a couple of hours depending on the machine.

```
python scripts/train.py --conf examples/ET-PCQM4MV2.yaml --layernorm-on-vec whitened --job-id pretraining
```

The option `--layernorm-on-vec whitened` includes an optional equivariant whitening-based layer norm, which stabilizes denoising. The pre-trained model checkpoint will be in `./experiments/pretraining`. A pre-trained checkpoint is included in this repo at `checkpoints/denoised-pcqm4mv2.ckpt`.

### Parameter-efficient Fine-tuning on QM9

To parameter-efficient fine-tune (LoRA or IA3) the model for HOMO/LUMO prediction on QM9, run the following command:

```bash
python scripts/train.sh qm9_lora
python scripts/train.sh qm9_ia3
```

### Data Parallelism 

By default, the code will use all available GPUs to train the model. We used single GPU (NVIDIA RTX 2080Ti) for fine-tuning.

## Acknowledgement

This implementation relies on the following source code:
- https://github.com/shehzaidi/pre-training-via-denoising
