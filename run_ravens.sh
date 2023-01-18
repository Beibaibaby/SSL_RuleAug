


TRAILS=(0)
NUM_SAMPLES=(5000 3000)
SAVE_PATH_PREFIX=ckpts/RelBase_MixMatch/Sample
DATA_PATH=/media/data4/yanglx/Datasets/AbstractReasoning

for num in ${NUM_SAMPLES[*]}
do
    for tr in ${TRAILS[*]}
    do
        SAVE_PATH=${SAVE_PATH_PREFIX}${num}/exp_${tr}/

        python main_mixmatch.py --dataset-name I-RAVEN --dataset-dir ${DATA_PATH} -j 16 --gpu 0,1,2,3 --fp16 \
                       --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
                       -a relbase_mtm --block-drop 0.1 --classifier-drop 0.5 \
                       --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
                       --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20 --max-iterations 500

        python main_mixmatch.py --dataset-name RAVEN-FAIR --dataset-dir ${DATA_PATH} -j 16 --gpu 0,1,2,3 --fp16 \
                       --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
                       -a relbase_mtm --block-drop 0.1 --classifier-drop 0.5 \
                       --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
                       --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20 --max-iterations 500

        # python main_mtm.py --dataset-name RAVEN --dataset-dir ../autodl-tmp/ -j 16 --gpu 0,1,2,3 --fp16 \
        #                --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
        #                -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
        #                --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
        #                --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20
    done
done

# TRAILS=(0)
# NUM_SAMPLES=(1000 500 100)
# SAVE_PATH_PREFIX=ckpts/DCNet_MixMatch/Sample
# DATA_PATH=/media/data4/yanglx/Datasets/AbstractReasoning

# for num in ${NUM_SAMPLES[*]}
# do
#     for tr in ${TRAILS[*]}
#     do
#         SAVE_PATH=${SAVE_PATH_PREFIX}${num}/exp_${tr}/

#         python main_mixmatch.py --dataset-name I-RAVEN --dataset-dir ${DATA_PATH} -j 16 --gpu 0,1,2,3 --fp16 \
#                        --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
#                        -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
#                        --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
#                        --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20 --max-iterations 100

#         python main_mixmatch.py --dataset-name RAVEN-FAIR --dataset-dir ${DATA_PATH} -j 16 --gpu 0,1,2,3 --fp16 \
#                        --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
#                        -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
#                        --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
#                        --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20 --max-iterations 100

#         # python main_mtm.py --dataset-name RAVEN --dataset-dir ../autodl-tmp/ -j 16 --gpu 0,1,2,3 --fp16 \
#         #                --image-size 80 --epochs 100 --evaluate-on-test -p 50 --seed ${tr} \
#         #                -a dcnet_mtm --block-drop 0.1 --classifier-drop 0.5 \
#         #                --batch-size 64 --lr 1e-3 --optim adam --wd 1e-5 --loss-type ct \
#         #                --ckpt ${SAVE_PATH} --num-train-samples ${num} --early-stopping 20
#     done
# done