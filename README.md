# Discrete Diffusion
---
This project will attempt to implement the ideas from the *Simple and Effective Masked Diffusion Language Models* into the Diffusions.jl package to create a diffusion transformer model capable of generating new antibody amino acid-sequences. 

## Contents

`diffusion.jl` contains many of the functions important for the diffusion process; e.g. the forward masking and the reverse unmasking process functions. `loss.jl`, as expected, contains the loss function used for this model. Finally, `transformers.ipynb` is a jupyter notebook where a diffusion transformer has been created to train on some generated data and `Antibodies.ipynb` is a jupyter notebook in which we train the diffusion transformer for generating antibodies.

