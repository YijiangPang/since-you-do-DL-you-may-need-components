# [Optimizers](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/Optimizers)
* minimal code version Adam, AdamW
* zeroth-order optimizer [MeZO](https://github.com/YijiangPang/since-you-do-DL-you-may-need-components/tree/main/Optimizers/ZerothOptimizer)[[1]](https://arxiv.org/abs/2305.17333), e.g., ```optimizer.zo_step(model, loss_func, x, y)```
* [tuning-free optimizer](https://github.com/YijiangPang/since-you-do-DL-you-may-need-components/tree/main/Optimizers/Tuning_free). [AdamG](https://github.com/YijiangPang/since-you-do-DL-you-may-need-components/blob/main/Optimizers/Tuning_free/AdamG.py)

# [Model](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/Model)
* Mixed of Expert [MoE](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/Model/MoE), ResNet18 version
* [Neural ODE model](https://github.com/YijiangPang/since-you-do-DL-you-may-need-components/blob/main/Model/NeuralODE_model.py)
* sklearn-like PyTorch-implimented [Logistic Regression](https://github.com/YijiangPang/since-you-do-DL-you-may-need-components/blob/main/Model/LR_pytorch.py). 

# [Loss funtion](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/Loss_function)
* DRO loss, Chi2, CVaR

# [Dataset](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/Dataset)
* [Chest radiography datasets](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/Dataset). CheXpert[[1]](https://stanfordmlgroup.github.io/competitions/chexpert/) and MIMIC-CXR Database[[2]](https://physionet.org/content/mimic-cxr/2.0.0/)


# [ML pipeline](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/ML_pipeline)
* [Outlier](https://github.com/YijiangPang/since-you-do-ML-you-may-need/tree/main/ML_pipeline/Outlier). Spectral Signature[[1]](https://arxiv.org/abs/1811.00636), Quantum Entropy Scoring[[2]](https://arxiv.org/abs/1906.11366)
* [BaggingClassifier](https://github.com/YijiangPang/since-you-do-DL-you-may-need-components/blob/main/ML_pipeline/BaggingClassifier.py)