base_path=${1-"."}
ckpt=${2-"."}
port=${3-"15031"}
model_type=${4-"llama"}


for data in dolly self_inst vicuna sinst uinst 
do
    for seed in 10 20 30 40 50
    do
        bash ${base_path}/scripts/moe/eval/eval_main_${data}.sh ${base_path} ${port} 2 ${ckpt} --model-type ${model_type} --seed $seed  --eval-batch-size 32
    done
done
