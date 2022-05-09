# Attacks

This section consists of 6 attacks: Add-CD, Add-HD, Knn, Drop(100, 200), and Perturb. The code in this part is based on [IF-Defense]
(https://github.com/Wuziyi616/IF-Defense) repository.
## Add
To run attack of add, please run the following command (note that you should specify the path of the pretrained model inside the file named `targeted_add_attack.py`)): 
```python
!NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 targeted_add_attack.py --model={MODEL NAME (pointnet/pointnet2/dgcnn)} --dist_func={DISTANCE FUNCTION (chamfer/hausdorff)} --num_points=1024 --dataset=mn40 --batch_size={BATCH SIZE} --binary_step=10 --num_class={NUMBER OF CLASS CATEGORIES} --data_root={PATH TO DATA TO BE ATTACKED}
```
## Perturb
To run attack of perturbation, please run the following command (note that you should specify the path of the pretrained model inside the file named `targeted_perturb_attack.py`)): 
```python
!NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 targeted_perturb_attack.py --model=pointnet --num_points=1024 --dataset=mn40 --batch_size={BATCH SIZE} --binary_step=10 --num_class={NUMBER OF CLASS CATEGORIES} --data_root={PATH TO DATA TO BE ATTACKED}
```
## Knn
To run attack of Knn, please run the following command (note that you should specify the path of the pretrained model inside the file named `targeted_knn_attack.py`)): 
```python
!NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 targeted_knn_attack.py --model=pointnet --num_points=1024 --dataset=mn40 --batch_size={BATCH SIZE} --num_class={NUMBER OF CLASS CATEGORIES} --data_root={PATH TO DATA TO BE ATTACKED}
```
## Drop
To run the drop attack, please follow the instructions in the [attack/drop.ipynb](https://github.com/kimianoorbakhsh/LPF-Defense/blob/main/attacks/attack/drop.ipynb) file.