# Learning Odor Quality Across Concentrations and Mixtures

This repository contains the LVS2 team’s model for the 2025 DREAM Olfactory Mixtures Prediction Challenge, developed by Vahid Satarifard and Laura Sisson. **Project page:** [Synapse (syn64743570)](https://www.synapse.org/Synapse:syn64743570/wiki/630800)

**Task 1:** Predict how odor quality changes with concentration for 151 monomolecular odorants using 51-term RATA profiles (two intensity levels per odorant).  
**Task 2:** Predict full 51-term odor profiles for >650 mixtures (sizes 2, 3, 5, 10) composed from a pool of >145 odorants.

## Summary
For Task 1, we train CatBoost via Optuna using Mordred-Morgan descriptors, dilutions, and flags to predict odor perception across concentrations. For Task 2, we fine-tune a graph neural network pre-trained on aroma-chemical pairs to predict odor perception for blends of aroma-chemicals.

## Background
Despite advances in modeling odor perception for single molecules [2,3], predicting how odor profiles shift across concentration remains unresolved.
For Task 1, we treated the change in concentration as a translation problem: an Optuna-tuned CatBoost regressor learned to map the 51-dimensional rate-all-that-apply (RATA) profile at one concentration to that at another, using Mordred descriptors, Morgan fingerprints, dilution scalars, and direction flags indicating High→Low or Low→High prediction.

For Task 2, we used a separate modeling strategy based on a fine-tuned pair-model architecture for predicting odor qualities of mixtures [4]. The model represents a mixture as a disconnected graph, with each aroma-chemical forming a component in a message-passing phase, followed by a readout attention phase that learns non-linear blending effects. It was first trained on a large corpus of molecule–molecule pairs, then fine-tuned on the challenge mixture dataset containing 2, 3, 5, and 10-component blends evaluated on 51 semantic descriptors. Both phases are invariant to the number of molecules.

## Methods
### Task 1
**Feature Selection:** We combined 1,827 Mordred descriptors and 2,049 hashed Morgan fingerprints, dropping constant columns. We appended task-specific features: (1) dilution scalars for both concentrations in an order encoding the prediction direction, and (2) one-hot direction flags. The known concentration’s rating vector was concatenated, yielding the feature layout: [descriptors, dil_1, dil_2, known ratings, flags].

**Model Training and Prediction:** For each molecule with both High (H) and Low (L) odour profiles, we created two supervised pairs (H→L, L→H), doubling the sample size. We used 5-fold stratified GroupKFold at the molecule level, binning molecules by “sigmoid-fraction” (fraction of terms with |log₂(H/L)| > sigmoid_cut). A multi-output CatBoostRegressor was tuned with Optuna (TPE sampler, 40 trials, GPU backend), optimizing 1− r̄ (mean Pearson correlation). The best trial achieved mean Pearson r = 0.635 and cosine distance = 0.244. Parameters were frozen and the model refit on all training data; final test predictions were the average over the 5 folds. Output: a 51-dimensional odour rating vector.



### Task 2
**Feature Selection:** We began with a pre-trained GNN (~500k parameters):

1- Message passing: 100k parameters over 3 graph isomorphism convolutions.

2- Readout phase: 400k parameters using set transformer aggregation.

Molecules were converted to PyTorch Geometric graphs (Open Graph Benchmark utilities) and concatenated into a single disconnected graph per mixture. Fine-tuning data included the training dataset, leaderboard dataset, and a subset of Task 1 data. Concentrations were not included as inputs.

**Model Training and Prediction:** We initialized a new descriptor prediction head and trained with the loss: gamma * MSE + (1.0 - gamma) * cosine_distance. Early stopping (min_delta = 1e-4) prevented reaching the 500-epoch cap. Hyperparameters (learning rate, weight decay, warmup, xi, gamma, target caps, MLP flag) were tuned via Optuna with 3-fold ShuffleSplit, maximizing (Pearson – cosine distance). The best trial achieved mean Pearson r = 0.700 and cosine distance = 0.187. Final test predictions averaged over the 3 folds.

## Conclusion
We approached the two tasks using different machine learning techniques.
For Task 1, we framed concentration change as a translation between paired RATA profiles, enabling CatBoost to exploit molecular structure and perceptual context. GroupKFold splitting avoided High/Low leakage. The best trial achieved Pearson r = 0.635 and cosine distance = 0.244; final predictions averaged over 5 folds. The approach captured concentration-dependent perceptual shifts by combining descriptors, dilution scalars, direction flags, and known odour profiles.

For Task 2, we fine-tuned a GNN pair-model to generalize across mixtures of varying size without retraining the message-passing backbone. Despite being concentration-agnostic, it achieved Pearson r = 0.700 and cosine distance = 0.187, demonstrating robust mixture representation transfer.

Task 1 performance may be limited by descriptor coverage and how well features capture non-linear structure–perception relationships. Task 2’s lack of concentration input may limit performance where intensity–quality interactions are strong.


## References
[1] DREAM Olfactory Mixtures Prediction Challenge 2025. Project SynID: syn64743570

[2] Keller, A., et al. Science, 355(6327):820–826, 2017.

[3] Lee, B.K., et al. Science, 381(6661):999–1006, 2023.

[4] Sisson, L., et al. ACS Omega, 10(9):8980–8992, 2025.
