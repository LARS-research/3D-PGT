# tox21
python main.py --cfg configs/GPS/tox21-GPS+RWSE.yaml wandb.use False seed 1
python main.py --cfg configs/GPS/tox21-GPS+RWSE.yaml wandb.use False seed 2
python main.py --cfg configs/GPS/tox21-GPS+RWSE.yaml wandb.use False seed 3

# toxcast
python main.py --cfg configs/GPS/toxcast-GPS+RWSE.yaml wandb.use False seed 1
python main.py --cfg configs/GPS/toxcast-GPS+RWSE.yaml wandb.use False seed 2
python main.py --cfg configs/GPS/toxcast-GPS+RWSE.yaml wandb.use False seed 3

