
######## full fine-tuning ET QM9
if [[ $1 == 'pretrain_et_pcqm4mv' ]]; then
  python scripts/train.py --conf examples/ET-PCQM4MV2.yaml --layernorm-on-vec whitened --job-id pretraining 
fi


######## full fine-tuning ET QM9
if [[ $1 == 'qm9_ft' ]]; then
  python scripts/train.py --conf examples/ET-QM9-FT.yaml --layernorm-on-vec whitened --job-id qm9_ft$2 --dataset-arg homo --pretrained-model checkpoints/denoised-pcqm4mv2.ckpt
fi


######## full fine-tuning ET QM9
if [[ $1 == 'qm9_tiny_ft' ]]; then
  python scripts/train.py --conf examples/ET-QM9-TINY-FT.yaml --layernorm-on-vec whitened --job-id qm9_tiny_ft$2 --dataset-arg homo --pretrained-model checkpoints/denoised-pcqm4mv2.ckpt
fi

######## full fine-tuning ET QM9
if [[ $1 == 'qm9_peft' ]]; then
  python scripts/train.py --conf examples/ET-QM9-PEFT.yaml --layernorm-on-vec whitened --job-id qm9_peft$2 --dataset-arg homo --pretrained-model checkpoints/denoised-pcqm4mv2.ckpt --peft ia3
fi

######## full fine-tuning ET QM9
if [[ $1 == 'qm9_tiny_peft' ]]; then
  python scripts/train.py --conf examples/ET-QM9-TINY-PEFT.yaml --layernorm-on-vec whitened --job-id qm9_tiny_peft$2 --dataset-arg homo --pretrained-model checkpoints/denoised-pcqm4mv2.ckpt --peft ia3
fi

######## full fine-tuning ET QM9
if [[ $1 == 'qm9_peft_debug' ]]; then
  python scripts/train.py --conf examples/ET-QM9-PEFT-debug.yaml --layernorm-on-vec whitened --job-id qm9_peft_debug$2 --dataset-arg homo --pretrained-model checkpoints/denoised-pcqm4mv2.ckpt --peft ia3
fi


######## full fine-tuning ET MT17
if [[ $1 == 'md17' ]]; then
  python scripts/train.py --conf examples/ET-MD17.yaml --layernorm-on-vec whitened --job-id md17_peft --dataset-arg aspirin --pretrained-model checkpoints/denoised-pcqm4mv2.ckpt
fi
