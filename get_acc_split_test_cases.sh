#!/bin/bash
#condaacti
# cuda 6 and 7 is shutdown
for((i=1;i<=5;i++));
do
  for ((j=0;j<=3;j++))
  do
  CUDA_VISIBLE_DEVICES=$i python get_acc_for_cvpr_subnets.py --arch_idx_label $i --arch_idx_gpu_split $j &
  done
done

