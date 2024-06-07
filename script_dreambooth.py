

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="val/MIMOLETTE"
export OUTPUT_DIR="mimolette"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

export HF_TOKEN="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"


accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of MIMOLETTE cheese" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=600 \
  --validation_prompt="A piece of MIMOLETTE cheese sits on a wooden board in warm sunlight." \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub