#!/bin/bash
# Please download data from https://traffic-signal-control.github.io/
source ~/.bashrc
xparl start --cpu_num 33 --port 8018
cityflow_py3 ./main.py --cluster_ip "localhost" --aux_coef 0.01 --logdir "hz_reroute/vae_aux_coeff_0.01_exp1" --model_type "CModelVAE" --data_path "./examples/hz.json" &
cityflow_py3 ./main.py --cluster_ip "localhost" --aux_coef 0.01 --logdir "hz_reroute/vae_aux_coeff_0.01_exp2" --model_type "CModelVAE" --data_path "./examples/hz.json" &
cityflow_py3 ./main.py --cluster_ip "localhost" --aux_coef 0.01 --logdir "hz_reroute/vae_aux_coeff_0.01_exp3" --model_type "CModelVAE" --data_path "./examples/hz.json" &
wait
