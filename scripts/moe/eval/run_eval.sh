base_path=${1-"."}
ckpt=${2-"."}
port=${3-"15031"}
model_type=${4-"llama"}
gpu_num=${5-"1"}
num_selects=${6-"None"}

if [ ${model_type} == "moe" ] && [ ${num_selects} != "None" ]; then
    ckpt_path=${ckpt//"/"/"_"}_ns${num_selects}
else
    ckpt_path=${ckpt//"/"/"_"}
fi

for data in dolly self_inst vicuna
do
    for seed in 10 20 30 40 50
    do
        while ! test -f ./results/moe/eval_main/${data}-512/${ckpt_path}/${seed}/preds.txt
        do
            bash ${base_path}/scripts/moe/eval/eval_main_${data}.sh ${base_path} ${port} ${gpu_num} ${ckpt} ${num_selects} --model-type ${model_type} --seed $seed  --eval-batch-size 64
            sleep 20
        done
    done
done

for data in sinst uinst
do
    for seed in 10 20 30 40 50
    do
        while ! test -f ./results/moe/eval_main/${data}_11_-512/${ckpt_path}/${seed}/preds.txt
        do
            bash ${base_path}/scripts/moe/eval/eval_main_${data}.sh ${base_path} ${port} ${gpu_num} ${ckpt} ${num_selects} --model-type ${model_type} --seed $seed  --eval-batch-size 64
            sleep 20
        done
    done
done

python3 compute_score.py --name ${ckpt}
