export NCCL_DEBUG=""
# Image: vault.habana.ai/gaudi-docker/1.16.2/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest
# pip install optimum[habana]
pip3 install git+https://github.com/HabanaAI/DeepSpeed.git@1.17.0
pip3 install -e transformers/
# pip3 install torch==2.0.1
# pip3 install deepspeed==0.10.0
# pip3 install torchvision==0.15.2
pip3 install nltk
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install sentencepiece
pip3 install protobuf==3.20.3
pip3 install peft==0.13.2
pip3 install -U "huggingface_hub[cli]"
