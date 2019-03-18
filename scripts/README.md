
# Descriptions

## Models

`gsd.py` - Graph-structured decomposition

`models.py` - Models for benchmarking. Functions should take in 2D array, `X`, as well as any other model parameters. Also wraps `gsd.py`. 

## Synthetic data

`samplers.py` - Generate source and loadings matrices using different sampling strategies

## Evaluation

`scoring.py` - Metrics for model performance. 

`evaluate.py` - Evaluate performance of models on different synthetic datasets. All parameters should be specified in `.json` files beforehand. Performs the following: 

 1. **Generate and save synthetic data** by calling classes in `samplers.py`. Synthetic data paramters are specified by `data_specs_file`. 
 2. **Run models** on each synthetic dataset by calling functions from `models.py`. Models and model parameters are specified by `model_specs_file`. 
 3. **Score performance** of each model on each dataset using metrics defined in `scoring.py`. 