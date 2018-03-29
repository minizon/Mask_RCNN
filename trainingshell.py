#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:53:08 2018

@author: pami4
"""
import os

run_GPU = 'CUDA_VISIBLE_DEVICES=0,1,2,3'
    

experiment_name = 'text_logs'
if os.path.exists(experiment_name):
    print("Dir already exist\n")
else:
    os.system('mkdir -p ' + experiment_name)

cmd = 'coco.py train --dataset=".." --model=coco'

file_nohup = './{:s}/kp_training.nohup'.format(experiment_name)

print('\nRun the training on GPU with nohup')
os.system(run_GPU + 'nohup python -u '+ cmd + ' > ' + file_nohup)