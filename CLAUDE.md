# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian place field estimation with temporal stability analysis for neuroscience research. Analyzes hippocampal place cell firing patterns by:
- Computing spatial firing rate maps using kernel density estimation
- Splitting sessions into temporal blocks to assess place field stability
- Using Bayesian inference (conjugate normal-normal updates on Fisher-z transformed correlations) to estimate posterior distribution over stability

## Running the Code

```bash
python model1_test.py
```

Runs a synthetic data demo: simulates a place cell with slight field drift and outputs `place_field_stability.png`.

## Dependencies

- numpy
- scipy (ndimage, stats)
- matplotlib

## Key Functions

- `run_analysis()` - Main pipeline entry point
- `compute_occupancy()` / `compute_spike_map()` / `estimate_rate_map()` - Core rate map computation
- `split_into_blocks()` / `compute_block_rate_maps()` - Temporal block analysis
- `pairwise_spatial_correlation()` / `bayesian_stability_estimate()` - Stability metrics

## Data Format

- **position**: (N, 3) array of `[timestamp, x, y]`
- **spike_times**: (S,) array of spike timestamps
