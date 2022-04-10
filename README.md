# ATMLGraphAttentionNetworks

## Datasets
Make sure there is a "data" folder in the root folder of this repository before you run any of the experiments. The datasets
should be automatically downloaded to "./data" when you run the experiment.

## Experiments
To run one of the experiments from the report, just run the associated python script. The inductive dataset results are
from run_inductive.py. The CIFAR10 results are from run_gnn_benchmark.py. The extension experiments are from run_act_func_experiment.py,
run_heads_experiments.py and run_params_experiment.py. Each of those files has hyperparameters as all caps variables at the top of the
file, which can be adjusted to change the experiment. 
