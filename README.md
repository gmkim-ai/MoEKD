# MoEKD

## 설명

- 모든 script는 다 scripts 폴더에 정리되어 있어서, "bash scripts/moe/{method}/{file}.sh" 이런식으로 실행해서 실험 진행했습니다.
  
- scripts 폴더 내에서, moe폴더만 제가 수정해서 사용했고, 나머지 폴더는 무시하셔도 돼요.

- script 파일 뒤에 _loop이 붙은 경우엔, 한번에 10 epoch이 아니라 1 epoch씩 돌려서, 제대로 돌아가야만 그 다음 epoch을 수행하도록 끊어서 실행하는 방식이에요. Gaudi에서 가끔 Semantation Fault 하면서 그냥 학습 중에 멈춰버리는 이슈가 있어서, 이런식으로 우회했습니다.
  
- scripts/moe 내에서,

eval: 학습이 다 끝나고 evaluation할때 사용되는 스크립트에요. 5개의 dataset에 대해 response를 생성하고, ROUGE-L score를 계산합니다.

gkd: baseline 중 하나인 gkd 입니다.

kd: baseline 중 하나인 hinton_kd에요.

minillm, promptkd, seqkd: 무시하셔도 됩니다. (예전 코드라 ;;)

moekd: knowledge augmentation을 실험한 스크립트들이에요. kd_{teacher모델}_to_{student}_{gkd/temp}_nr{몇번 repeat할건지, 즉 knowledge augmentation 개수}_ns{몇개의 expert를 골라서 사용할것인지}_sp{sampling ratio로, 이 확률로 sampling 진행, 아니면 그냥 Top-ns개 사용}

**가령** scripts/moe/moekd/ kd_3_5B-4_16_to_1_3B_gkd_nr1_ns15_sp05_loop 라면, llama-moe/LLaMA-MoE-v1-3_5B-4_16 모델을 1.3B 모델로, gkd 방식으로, 15개 expert를 5%확률로 샘플링, 95%확률로는 top15 고정으로다가 가지고 와서 knowledge augmentation 1번 하는 실행코드임!!

여기서 temp는 초반 observation에 사용된 실험으로, gkd는 gkd처럼 on-policy로 student가 생성한 pseudo-target을 데이터로 써서 진행하는거고, temp는 그냥 hinton kd처럼 ground truth를 데이터로 써서 진행하는 차이만 있습니다.

sfrmoekd: student-aware router 방법이에요.

sft: 지금 사용하는 student 모델과 teacher 모델 모두 dolly dataset에 대해 어느정도 SFT를 진행한 모델을 사용해서 KD를 시작하는데, 이 SFT를 하는 코드에요.

## 참고용

- https://github.com/microsoft/LMOps/tree/main/minillm 에 있는 코드가 eval 시에 쓰이는 코드임.
- https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py 에 있는 코드는 trl에서 구현한 GKD 코드임.
