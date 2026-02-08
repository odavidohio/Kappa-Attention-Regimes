# Kappa-Attention-Regimes

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Empirical application of the Kappa Method to attention dynamics in large language models.

## Overview
This repository provides empirical evidence that hallucination in LLMs
corresponds to persistent structural effort within a shared critical regime,
rather than a regime transition.

We apply the Kappa Method to attention dynamics across multiple architectures
(Phi-3, Mistral-7B, LLaMA-3.1), showing invariant regime allocation and
architecture-dependent separability.

## Relationship to Kappa Method
This work instantiates the observables and regime taxonomy defined in:
https://github.com/odavidohio/Kappa-Method

## What this repository contains
- Observable computation from attention matrices
- Cross-architecture experiments
- Reproducible analysis scripts
- Figures and metrics reported in the paper

## What this repository does NOT claim
- It does not eliminate hallucinations
- It does not modify model weights or logits
- It is not a production safety system
