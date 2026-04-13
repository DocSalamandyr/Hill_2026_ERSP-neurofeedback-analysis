"""ERSP reanalysis toolkit for the neurofeedback mechanism paper.

Modules
-------
config      Central analysis parameters (dataclasses).
io          BDF import + BioSemi event parsing.
study       Group-level metadata and batch management.
preprocess  Filtering, re-reference, ICA pipeline.
epochs      Reward-locked epoch extraction, early/late splits.
ersp        ERSP / ITC with single-trial normalization.
erp         ERP component extraction (P50, N1, P2).
resting     Welch PSD for resting-state EC/EO spectra.
stats       Cluster permutation, mixed models, Bayesian null tests.
viz         Publication figure generation (subpackage).
"""
