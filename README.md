This is an anonymized repo.

## Requirement

Python **3**

PyTorch **0.3**

## Data

Due to copyright issues, we can't directly release the datasets used in our experiments.
Instead, we provide the links to the data sources:

- [RCV1](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)
- [NYT](https://catalog.ldc.upenn.edu/LDC2008T19)
- [Yelp](https://www.yelp.com/dataset/challenge)
- [FunCat & GO](https://dtai.cs.kuleuven.be/clus/hmcdatasets/)

Please check `readData_*.py` to see how to use our scripts to process and generate the datasets from the original data.

## Training
All the parameters in `conf.py` have default values. Change parameters `mode`, `base_model`, and `dataset` to train and test on different settings.

## Test
To test a model, set `load_model=model_file` & `is_Train=False` in `conf.py`.

## More

More details to be updated.
