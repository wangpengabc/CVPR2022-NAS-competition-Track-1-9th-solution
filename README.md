# CVPR NAS Track 2022
There are two versions of implementation, which are based on PaddlePaddle and Pytorch,respectively.

ps: This project is adapted from OFA.
## Table of content
- [Project Structure](#I)
- [Training Porcedure](#II)

### <a id="I">Part I. Project Structure </a>
> There are mainly two parts in this project, compofacvprsinglemachine and compofacvprpaddle.

The compofacvprpaddle is used for building the project based on paddlepaddle.The compofacvprsinglemachine is used for supporting pytorch. 
Moreover, the compofacvpr is an implementation which contains distributed training and inference 
based on horovod and pytorch.
```
-|compofacvpr
-|compofacvprpaddle
-|compofacvprsinglemachine
-|result
-|runs
train_ofa_resnet.py
train_ofa_resnet_single_machine.py
train_ofa_resnet_single_machine_paddlepaddle.py
get_acc_for_cvpr_subnets.py
get_acc_split_test_cases.sh
-README.MD
```
### <a id="II">Part II. Training Porcedure </a>
- Training

Teacher model
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ofa_resnet_single_machine.py --task teacher --phase 1 --fixed_kernel
or
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ofa_resnet_single_machine_paddlepaddle.py --task teacher --phase 1 --fixed_kernel 
```
depth - First, the model is training in phase 1, then it is trained in phase 2.
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ofa_resnet_single_machine.py --task depth --phase 1 --fixed_kernel
or 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ofa_resnet_single_machine_paddlepaddle.py --task depth --phase 1 --fixed_kernel
```
expand - First, the model is training in phase 1, then it is trained in phase 2.
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ofa_resnet_single_machine.py --task expand --phase 1 --fixed_kernel
or 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ofa_resnet_single_machine_paddlepaddle.py --task expand --phase 1 --fixed_kernel
```

- Evaluation
To evaluate the subnets faster, we split the overall candidate subnets into 8 parts which
are evaluated on different gpus in parallel, respectively. The evaluation results are stored 
in the directory of result. After evaluating all the parts, we merge the results into one unified
json file.
```python
./get_acc_split_test_cases.sh
```
And the "submit_step" in get_acc_split_test_cases.py should change to True or False depending on 
the stage which may be merge or evaluating.


The checkpoint download url: [checkpoint](https://drive.google.com/file/d/1aI6oT_daWlZLiJ4LlTXsoovHR6xTCO2Y/view?usp=sharing)

The checkpoint should be placed in ./runs/default/depth2depth_width/phase2/checkpoint/