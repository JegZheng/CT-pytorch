## Demo code for conditional transport (CT) on toy experiments

"1d_GMM_exp.ipynb": The 1d-GMM experiments
Other files: Experiments on 2d toy datasets

### Requirements
- pytorch >= 1.2.0
- seaborn == 0.9.0
- pandas
- sklearn
- Tensorboard (for visualization of toy data experiments, but optional)

### Example usage

`$ python main.py --dataset 8gaussians --method CT`

or run all methods on all toy datasets with

`$ python main.py --run_all`

Use --help for more options

`$ python main.py --help`