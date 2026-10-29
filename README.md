# MT-Denoising: Time-Frequency Denoising for Magnetotelluric Data

This repository contains an MT adaptation of a dual-branch AIA Transformer denoising network. It provides code for preparing MT time series, adding synthetic noise during training, learning in the time-frequency domain, denoising full records, and estimating impedance tensors from the resulting complex spectra.

For the existing Chinese project notes, see [README_MT.md](README_MT.md).

## Overview

The workflow treats denoising as a supervised time-frequency learning problem:

1. Split source .STS records into the ex, ey, hx, hy, and hz channels.
2. Prepare training and cross-validation pools from source records.
3. Add synthetic MT noise to time-domain samples and convert them to short-time Fourier transform (STFT) features.
4. Train the MT network using complex real/imaginary (RI) features and a combined complex and magnitude loss.
5. Apply full-record inference and optionally estimate MT impedance, apparent resistivity, and phase from the denoised complex spectra.

The configuration describes training on randomly selected segments of 4,096–16,384 samples, while inference processes full records without chunking. The default STFT uses a Hann window, n_fft=512, and hop length 256, giving 257 frequency bins. Sampling rate defaults to 15 Hz and should match the input data.

## Model and noise setup

The MT model is implemented in mt_aia_net.py as MTDualAiaTransMergeCRM. The model accepts two-component RI time-frequency features with 257 frequency bins. The data pipeline applies synthetic noise in the waveform domain before STFT. The available noise generator (mt_noise.py) includes colored noise, harmonics, spikes, drift, and a white sensor-noise floor.

Training uses online augmentation: samples are randomly cropped and re-noised during training. The default configured training SNR range is −5 to 10 dB, with cross-validation SNR fixed at 5 dB. The default loss weights are 1.0 for complex loss and 0.5 for magnitude loss.

## Repository map

| Path | Purpose |
| --- | --- |
| split_channels.py | Splits an .STS record into channel-specific arrays. |
| prepare_mt_data.py | Builds train/cross-validation pools and sample manifests. |
| data_mt.py, mt_noise.py | Loads samples, performs online noise mixing, and creates STFT features. |
| mt_stft.py | STFT and inverse STFT utilities using RI features. |
| mt_aia_net.py, aia_inter_new.py | MT denoising network and shared AIA Transformer components. |
| train_mt.py, solver_mt.py | Training entry point and training loop. |
| infer_mt.py | Full-record inference; saves denoised waveforms and complex RI spectra. |
| evaluate_mt.py | Calculates sample-level and mean denoising metrics. |
| mt_impedance.py | Estimates impedance-related quantities from complex spectra. |
| config_mt.py | Data, STFT, noise, loss, and training configuration. |
| MT_dataset/ | Prepared data pools, split manifests, and JSON sample lists. |
| MT_enhanced/evaluation_ex.csv | Evaluation metrics for the ex channel. |
| README_MT.md | Existing Chinese workflow notes. |

## Quick start

Run these commands from the repository root. The commands below follow the workflow documented in README_MT.md; check config_mt.py and your data paths before running.

~~~bash
# 1. Split an input source into channels
python split_channels.py dataset/ModEM0.STS1

# 2. Prepare train and cross-validation pools and sample manifests
python prepare_mt_data.py

# 3. Train the model
python train_mt.py --epochs 100 --batch_size 4

# 4. Denoise selected channels from a source record
python infer_mt.py --channels ex,ey,hx,hy --stem ModEM7 --samples 16384

# 5. Estimate impedance from denoised complex spectra
python mt_impedance.py \
  --ex MT_enhanced/ex/ModEM7_denoised_ri.npy \
  --ey MT_enhanced/ey/ModEM7_denoised_ri.npy \
  --hx MT_enhanced/hx/ModEM7_denoised_ri.npy \
  --hy MT_enhanced/hy/ModEM7_denoised_ri.npy \
  --out MT_enhanced/impedance_ModEM7
~~~

The inference command expects a trained checkpoint at the configured model path unless another checkpoint is supplied. To evaluate results and write a metrics CSV, use evaluate_mt.py; inspect its --help output for the available channel, reference-data, and output-path options.

## Evaluation excerpt

The table below reproduces selected rows and the summary row from [MT_enhanced/evaluation_ex.csv](MT_enhanced/evaluation_ex.csv). It reports sample-wise metrics for the ex channel. SNR is in dB; MSE is as stored in the CSV. Higher SNR and correlation, and lower MSE, indicate closer agreement with the clean reference under the metric definitions used by the evaluation script.

| Sample | Input SNR (dB) | Denoised SNR (dB) | Noisy MSE | Denoised MSE | Noisy correlation | Denoised correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ModEM8_0000 | 5.000 | 14.309 | 1.0620 | 0.1245 | 0.8243 | 0.9764 |
| ModEM8_0001 | 5.000 | 16.182 | 2.5917 | 0.1974 | 0.8726 | 0.9930 |
| ModEM8_0002 | 5.000 | 9.952 | 1.0996 | 0.3516 | 0.8456 | 0.9482 |
| **Mean (39 samples)** | **5.000** | **9.757** | **1.4746** | **0.5748** | **0.8596** | **0.9361** |

These are the values in the committed CSV, not a newly run experiment. The mean row summarizes the 39 listed samples. It should not be interpreted as performance across other channels, datasets, noise settings, or evaluation protocols.

## Configuration and reproducibility notes

Key settings are centralized in config_mt.py, including sampling rate, segment lengths, source split, target channels, STFT parameters, noise range, validation SNR, loss weights, and model/output paths. The repository does not currently provide a pinned dependency file or a versioned environment specification. Install compatible versions of Python, PyTorch, and the scientific Python packages, then record the exact environment used for any reproduced results.

The tracked evaluation CSV gives example metrics, but a reproducible end-to-end report should also record the checkpoint used, source/test split, random seeds, data provenance, and exact configuration. Not every result or training artifact should be assumed to be regenerated by the commands above without checking local data and checkpoint availability.

## Citation and license

No paper citation, DOI, or repository license information is specified in the available project README files. Please cite the associated publication if one becomes available. Until a license is added, reuse and redistribution permissions for the code and data are not explicitly granted.
