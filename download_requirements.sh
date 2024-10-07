pip3 install -U "huggingface_hub[cli]"

echo "Downloading model checkpoints.. (about baselines via MiniLLM github repository)"
mkdir checkpoints/llama-moe/foundation
mkdir checkpoints/llama-moe/sft
mkdir checkpoints/sheared-llama/foundation
mkdir checkpoints/sheared-llama/sft
huggingface-cli download llama-moe/LLaMA-MoE-v1-3_0B-2_16 --local-dir checkpoints/llama-moe/foundation/3_0B-2_16
huggingface-cli download llama-moe/LLaMA-MoE-v1-3_0B-2_16-sft --local-dir checkpoints/llama-moe/sft/3_0B-2_16
huggingface-cli download llama-moe/LLaMA-MoE-v1-3_5B-4_16 --local-dir checkpoints/llama-moe/foundation/3_5B-4_16
huggingface-cli download llama-moe/LLaMA-MoE-v1-3_5B-4_16-sft --local-dir checkpoints/llama-moe/sft/3_5B-4_16
huggingface-cli download llama-moe/LLaMA-MoE-v1-3_5B-2_8 --local-dir checkpoints/llama-moe/foundation/3_5B-2_8
huggingface-cli download llama-moe/LLaMA-MoE-v1-3_5B-2_8-sft --local-dir checkpoints/llama-moe/sft/3_5B-2_8
huggingface-cli download princeton-nlp/Sheared-LLaMA-1.3B --local-dir checkpoints/sheared-llama/foundation/1_3B
huggingface-cli download princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT --local-dir checkpoints/sheared-llama/sft/1_3B
huggingface-cli download princeton-nlp/Sheared-LLaMA-2.7B --local-dir checkpoints/sheared-llama/foundation/2_7B
huggingface-cli download princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT --local-dir checkpoints/sheared-llama/sft/2_7B

echo "Downloading data for training and evaluation.."
wget -O data.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/data.tar
tar -xvf data.tar
wget -O processed_data.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/processed_data.tar
tar -xvf processed_data.tar

echo "Pre-processing data for training and evaluation.."
bash scripts/moe/tools/process_data_dolly.sh .

