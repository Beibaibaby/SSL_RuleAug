

TRAILS=(0 1 2 3 4)
NUM_SAMPLES=(3000)
SAVE_PATH_PREFIX=../autodl-tmp/Experiment-Results/DCNet_MTM/Sample

for num in ${NUM_SAMPLES[*]}
do
    for tr in ${TRAILS[*]}
    do
        SAVE_PATH=${SAVE_PATH_PREFIX}${num}/exp_${tr}/

        # python main_mtm.py --dataset-name I-RAVEN --dataset-dir ../autodl-tmp/ -j 16 --gpu 0,1,2,3 --fp16 \
        #                --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
        #                -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
        #                --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
        #                --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20

        python main_mtm.py --dataset-name RAVEN-FAIR --dataset-dir ../autodl-tmp/ -j 16 --gpu 0,1,2,3 --fp16 \
                       --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
                       -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
                       --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
                       --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20

        # python main_mtm.py --dataset-name RAVEN --dataset-dir ../autodl-tmp/ -j 16 --gpu 0,1,2,3 --fp16 \
        #                --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
        #                -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
        #                --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
        #                --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20
    done
done