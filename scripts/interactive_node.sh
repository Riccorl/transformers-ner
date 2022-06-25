#!/bin/bash

srun --pty -A tra22_Nvaitc -p m100_sys_test -q qos_test -N 1 --gres=gpu:1 --cpus-per-task=10 bash
