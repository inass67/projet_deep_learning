# Deep Learning with PyTorch: MLP, CNN & RNN/Seq2Seq

> Academic project – **EMSI**, Data & Artificial Intelligence Engineering
>
> Author: **Inas Mourid**
> 
>A hands-on study of how deep learning architectures adapt to the structure of the data: **tabular** (MLP), **images** (CNN) and **sequences** (RNN / LSTM / GRU / Seq2Seq). Every model is built from scratch in **PyTorch** and tested on real datasets.

## Results

| Part | Data type | Dataset | Model | Result |
|---|---|---|---|---|
| I | Tabular | Breast Cancer Wisconsin (Diagnostic) | MLP | **95% accuracy** |
| II | Images | Fashion-MNIST | CNN (LeNet) vs MLP | **87.12%** vs 84.13% |
| III | Text sequences | Tatoeba (French → English) | RNN / LSTM / GRU, Seq2Seq | **BLEU-1: 71.65%** |

##  Contents

### Part I – MLP & PyTorch engineering
- Binary classification on real medical data
- Inspection of model parameters (`named_parameters`, `state_dict`)
- Comparison of 3 weight initialization strategies and their effect on convergence
- Saving and restoring the best model

### Part II – CNN & computer vision
- Manual computation of output dimensions (kernel, padding, stride)
- LeNet-style CNN on Fashion-MNIST, compared with an MLP
- Ablation study: Max vs Average pooling, 1×1 convolutions, number of filters (83.29% → 85.79%)

### Part III – RNN, LSTM, GRU & Seq2Seq
- Backpropagation through time (BPTT) and gradient clipping
- From simple RNN to LSTM/GRU (vanishing gradient problem)
- Encoder-decoder (Seq2Seq) for French → English machine translation and decoding strategies

## Tech stack

Python · PyTorch · Torchvision · NumPy · Matplotlib · Jupyter

## Run

    pip install torch torchvision numpy matplotlib jupyter
    jupyter notebook notebooks/

The full report (in French) is available in the PDF file of this repository.
