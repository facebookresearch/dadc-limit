# DADC in the Limit

This repository contains the official data and code for the ACL 2022 findings paper "Analyzing Dynamic Adversarial Training Data in the Limit".

## Data

The `data` folders contains our final training datasets, broken down by non-adversarial data collection, static adversarial data collection, and dynamic adversarial collection. The datasets differ slightly from the ones used in the paper, namely a small amount of data was removed for quality reasons. We also include our manually-crafted test set.

## Analysis

See the `analysis` folder to recreate the analysis experiments from Section 5 of the paper. In particular:
+ `diversity.py` will recreate the diversity rows in Table 4.
+ `complexity.py` will recreate the complexity rows in Table 4.
+ `artifacts.py` will recreate the artifacts rows in Table 4.

## Training Models

The `training` folder contains code to reproduce the training experiments using RoBERTa large models. To run the `train.sh` script to recreate the experiments, you will need to run the following installation inside the `training` folder.

```
conda create -n dadc python=3.7
source activate dadc
pip install -r requirements.txt
```
Now you should be ready to go!

## Citation

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@inproceedings{Wallace2022Dynamic,  
    Title = {Analyzing Dynamic Adversarial Training Data in the Limit},
    Author = {Eric Wallace and Adina Williams and Robin Jia and Douwe Kiela}, 
    Booktitle={Findings of the Association for Computational Linguistics},
    Year = {2022}
}
```

## Contributions and Contact

This code was developed by Eric Wallace, contact available at ericwallace@berkeley.edu.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/facebookresearch/dadc-limit). If you find an issue with the code, please open an [issue](https://github.com/facebookresearch/dadc-limit).
