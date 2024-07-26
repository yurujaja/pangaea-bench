#baseline
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 
# full finetuning
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune full_finetuning
# norm tuning
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune norm_tuning
# bias tuning
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune bias_tuning
# patch embed
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune patch_embed
# lora
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune lora
# lora patch embed
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune lora_patch_embed
# slr 
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune low-rank-scaling
# slr patch embed
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune low-rank-scaling-patch-embed
# patch embed 6 chans
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae_6_bands.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune patch_embed
# lora patch embed
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae_6_bands.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune lora_patch_embed
# slr patch embed
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae_6_bands.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune low-rank-scaling-patch-embed
# full finetuning
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae_6_bands.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet.yaml --num_workers 4 --eval_interval 5 --finetune full_finetuning
# adapter
torchrun --nnodes=1 --nproc_per_node=1 run.py  --encoder_config configs/foundation_models/scalemae_adapt.yaml --dataset_config configs/datasets/hlsburnscars.yaml --segmentor_config configs/segmentors/upernet_adapt.yaml --num_workers 4 --eval_interval 5 