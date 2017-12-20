 #!/bin/bash

 CUDA_VISIBLE_DEVICES=$(free-gpu) /home/ywang/anaconda3/bin/python nn_bs_torch.py --gpu=$(free-gpu) --load-pickle false
