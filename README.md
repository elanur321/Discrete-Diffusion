# Discrete Diffusion
---
This project will attempt to implement and integrate the ideas from the [*Simple and Effective Masked Diffusion Language Models*](https://arxiv.org/abs/2406.07524) article into the [`Diffusions.jl`](https://github.com/MurrellGroup/Diffusions.jl) package. In this article, a new approach to masked diffusion language is suggested, and according to the article-authors, the efficacy of the model is dependent on the well-engineered implmentation. Here, we try to implement these ideas into [`Diffusions.jl`](https://github.com/MurrellGroup/Diffusions.jl) to allow for easy usage and combination of this new diffusion model. 

Moreover, here we also try to apply this model to create a diffusion transformer model capable of generating new antibody amino acid-sequences. We use data from [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) (OAS) and train a transformer to generate new amino acid sequences by utilizing the created masked diffusion model.

## Contents

`diffusion.jl` contains many of the functions important for the diffusion process; e.g. the forward masking and the reverse unmasking process functions. `loss.jl`, as expected, contains the loss function used for this model. Finally, `transformers.ipynb` is a jupyter notebook where a diffusion transformer has been created to train on some generated data and `Antibodies.ipynb` is a jupyter notebook in which we train the diffusion transformer for generating antibodies.

## A Brief Technical Description




