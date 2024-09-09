base_path=${1-"."}
ckpt=${2-"."}
port=${3-"15031"}
model_type=${4-"llama"}
gpu_num=${5-"1"}


for data in dolly self_inst vicuna sinst uinst 
do
    for seed in 10 20 30 40 50
    do
        bash ${base_path}/scripts/moe/eval/eval_main_${data}.sh ${base_path} ${port} ${gpu_num} ${ckpt} --model-type ${model_type} --seed $seed  --eval-batch-size 32
        sleep 10
    done
done

python3 compute_score.py --name ${ckpt}
