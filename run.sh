#!/bin/bash
#PBS -A PBML
#PBS -l select=1:ncpus=36:ngpus=8
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -q by-gpu
#PBS -j oe
#PBS -o output.log

export NUMEXPR_MAX_THREADS=32
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMBA_NUM_THREADS=4
export HUGGINGFACE_TOKEN=$(cat huggingface_credential.txt)

cd $PBS_O_WORKDIR


echo "Verifying with nvidia-smi:"
nvidia-smi

module load conda/2024-08-08

source ~/.bashrc

ENV_NAME="radioactive_watermark"

if ! conda env list | grep -q "^$ENV_NAME\s"; then
	echo "Conda env '$ENV_NAME' not found. Creating it..."
	conda create -n "radioactive_watermark" python=3.8
	conda activate radioactive_watermark

	#conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
	#pip install -r requirements.txt
else
	echo "Using existing Conda environment: $ENV_NAME"
	conda activate $ENV_NAME
fi

rm -rf $ENV_DIR/lib/python3.8/site-packages/bitsandbytes*

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121


# Install other compatible packages
pip install \
  transformers==4.31.0 \
  sentence-transformers==2.2.2 \
  huggingface_hub==0.17.3 \
  datasets==2.14.5 \
  peft==0.4.0 \
  accelerate==0.21.0 \
  scipy \
  tqdm \
  numexpr \
  bitsandbytes==0.46.0

pip install --no-cache-dir "numba<0.59"
pip install datasets

# === Clear unused cache to stay within quota ===
rm -rf ~/.cache/huggingface

echo "Testing bitsandbytes setup:"
python -m bitsandbytes

#MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
#python main_watermark.py \
#    --model_name $MODEL_NAME \
#    --prompt_path "data/used_maryland_ngram2_seed0.jsonl" \
#    --method none --method_detect maryland \
#    --ngram 2 --scoring_method v2 \
#    --nsamples 10000 --batch_size 16 \
#    --output_dir output_closed_supervised_0p05/ \
#    --filter_path "data/used_maryland_ngram2_seed0_filter.pkl"

#python lora_train.py
python data_prep.py



