


result_path=/localdisk2/SSL_Results/ckpts/MRNET_MixMatch/Sample5000


# python print_results.py --dataset-name RAVEN --image-size 80 --epochs 100 --optim adam --wd 1e-5 --loss-type ct \
#                         -a dcnet --block-drop 0.1 --classifier-drop 0.5 \
#                         --ckpt ${result_path} --exps 0,1,2,3,4

python print_results.py --dataset-name RAVEN-FAIR --image-size 80 --epochs 100 --optim adam --wd 1e-5 --loss-type ct \
                        -a relbase_mtm --block-drop 0.1 --classifier-drop 0.5 \
                        --ckpt ${result_path} --exps 0,1,2

python print_results.py --dataset-name I-RAVEN --image-size 80 --epochs 100 --optim adam --wd 1e-5 --loss-type ct \
                        -a relbase_mtm --block-drop 0.1 --classifier-drop 0.5 \
                        --ckpt ${result_path} --exps 0,1,2