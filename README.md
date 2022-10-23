# About


# Directory structure

```bash
├── ...
├── data                    # Datasets
│   ├── mimic               # MIMIC dataset: 'V', 'PLETH' and 'ABP' signals data
│   └── ptbxl               # PTB-XL dataset: 12-leads ECG signals data and diagnostic classes assigned to each sample
└── src                     # Source files
    ├── ...
    ├── data                # data module: data loading etc.
    ├── models              # models module: DeepLearning models
    └── signals             # signals module: Signal classes made for signal exploration and feature extraction
```



# Datasets used

1. PTB-XL

    * [source URL](https://physionet.org/content/ptb-xl/1.0.1/)
    * papers:
        * [PTB-XL, a large publicly available electrocardiography dataset](https://www.nature.com/articles/s41597-020-0495-6)
        * [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://ieeexplore.ieee.org/document/9190034)
        *

    * interesting urls:
        * [The China Physiological Signal Challenge 2018: Automatic identification of the rhythm/morphology abnormalities in 12-lead ECGs](http://2018.icbeb.org/Challenge.html)
        * [repository of Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://github.com/helme/ecg_ptbxl_benchmarking)

    * description:
        * 21837 samples:
            * train: 17441
            * val: 2193
            * test: 2203
        * Many samples had several classes assigned, so there was a need to filter them out (separately for each task)

    * preprocessing:
        1. Find measurements with only one class (or subclass) assigned
        2. Filter out the rest of measurements

    * After preprocessing:
        * *diagnostic_class* classification (16272 samples with one class assigned, 5 classes):
            * train: 12978
            * val: 1642
            * test: 1652
        * *diagnostic_subclass* classification (15239 samples with one class assigned, 23 classes):
            * train: 12123
            * val: 1555
            * test: 1561

    * tasks:
        * **classification**: *diagnostic_class*
        * **classification**: *diagnostic_subclass*





2. MIMIC

    * [source URL](https://physionet.org/content/mimicdb/1.0.0/055/#files-panel)
    * papers:
        * [Hypertension Assessment via ECG and PPG Signals: An Evaluation Using MIMIC Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163274/)
        *

    * interesting urls:
        * [Subjects list](https://physionet.org/files/mimicdb/1.0.0/mimic-index.shtml)
        * [Cuff-Free Blood Pressure Estimation Using Pulse Transit Time and Heart Rate](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4512231/)

    * description:
        * 72 recordings (multiple recordings for some of the subjects)
        * Average recording time ~40hours
        * Only some part of recordings have all 'V', 'ABP' and 'PLETH' signals
        * Some of 'PLETH' signals contains long `nan`, `0` or `1.0235` sequences

    * preprocessing:
        1. Find measurements with 'V', 'ABP' and 'PLETH' signals ("V" - ECG, "ABP" - Arterial Blood Pressure, "PLETH" - PPG Signal)
        2. Filter out segments where signal is `nan`, `0` or `1.0235` (PLETH case) for atleast 50 samples (0.4 seconds)
        3. Filter out segments shorter than 10 seconds.
        4. Interpolate `nan` fragments with `pchip` method
        5. Save to csv with 'start-end.csv' format (start is the start index of a segment and end is the end index of the segment)

    * After preprocessing:
        * 50 recordings with 'V', 'ABP' and 'PLETH' signals
        * ~1400 hours of continuus recordings
        * 496 682 complete ten seconds recordings

    * tasks:
        * **regression**: continuus Arterial Blood Pressure (ABP) prediction from ECG and PPG ('PLETH') signals
            * sample by sample (e.g. [video](https://www.youtube.com/watch?v=4fLw79l3Dh4))
        * **regression**: aggregated Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) predictions from PPG ('PLETH') and/or ECG signals
            * aggregated for 10 seconds of signal





3. MIMIC-III Waveform

    * [source URL](https://physionet.org/content/mimic3wdb/1.0/)
    * preprocessing:
        1.
        2.
    * papers:
        *
        *
    * interesting urls:
        * [The MIMIC-III Waveform Database](https://archive.physionet.org/physiobank/database/mimic3wdb/)

4. Sleep-EDF




# Signal representations

1. Raw signal

    * whole sequence (10 sec)
    * sequence aggregated to a single PPG / PQRS episode

2. Signal embedded with neural networks

    * Temporal Neighborhood Coding
        * [article](https://arxiv.org/abs/2106.00750)
        * [github](https://github.com/sanatonek/TNC_representation_learning)

    * A Transformer-based Framework for Multivariate Time Series Representation Learning
        * [article](https://arxiv.org/abs/2010.02803)
        * [github](https://github.com/gzerveas/mvts_transformer)
        * [video](https://www.youtube.com/watch?v=aXKv7uFNZLY)

    * Time-Series Representation Learning via Temporal and Contextual Contrasting
        * [article](https://arxiv.org/abs/2106.14112)
        * [github](https://github.com/emadeldeen24/TS-TCC)

3. Features extracted from signal as tabular data

    * Features for each PPG / PQRS episode
    * Features for aggregated PPG / PQRS episode



# Signals

## All signals

1. Papers

2. Features
    * Statistical features
        * quantiles
        * mean
        * median
        * minimum
        * maximum
        * variance
        * skewness
        * kurtosis
    * Other
        * zero crossing
        *

## ECG (Electrocardiogram)

1. Papers
    * [A novel ECG signal classification method using DEA-ELM](https://www.sciencedirect.com/science/article/pii/S0306987719312381#b0085)
    * [An Overview of Heart Rate Variability Metrics and Norms](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/)
    * [ECG-based heartbeat classification for arrhythmia detection: A survey](https://www.sciencedirect.com/science/article/pii/S0169260715003314)
    * [A survey on ECG analysis](https://www.sciencedirect.com/science/article/pii/S1746809418300636#bib0175)
    * [Interpretable deep learning for automatic diagnosis of 12-lead electrocardiogram](https://www.sciencedirect.com/science/article/pii/S2589004221003412)
    * [Automatic diagnosis of the 12-lead ECG using a deep neural network](https://www.nature.com/articles/s41467-020-15432-4)

2. Features
    * Single episode:
        * for each peak (P, Q, R, S, T):
            * start time
            * end time
            * duration
            * peak time
            * peak val
            * onset slope
            * offset slope
            * surface
            * energy
        * PQ segment and interval
        * PR segment and interval
        * ST segment and interval
        * QT segment and interval
        * QRS interval
    * Many episodes:
        * PP interval
        * QQ interval
        * RR interval
        * SS interval
        * TT interval
        * P signal
        * Q signal
        * R signal
        * S signal
        * T signal
        * most dominant frequency
        * HRV features:
            * heart rate
            * breathing rate
            * SDRR
            * SDNN
            * RMSSD



## PPG (Photopletysmogram)

1. Papers
    * [Continuous PPG-Based Blood Pressure Monitoring Using Multi-Linear Regression](https://arxiv.org/abs/2011.02231)
    * [Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning](https://www.mdpi.com/1424-8220/21/18/6022)
    * [End-to-End Blood Pressure Prediction via Fully Convolutional Networks](https://ieeexplore.ieee.org/abstract/document/8936850)
    * [A Benchmark Study of Machine Learning for Analysis of Signal Feature Extraction Techniques for Blood Pressure Estimation Using Photoplethysmography (PPG)](https://ieeexplore.ieee.org/document/9558767)
    * [A Deep Learning Approach to Predict Blood Pressure from PPG Signals](https://arxiv.org/abs/2108.00099)
    * [A Comparison of Deep Learning Techniques for Arterial Blood Pressure Prediction ](https://link.springer.com/article/10.1007/s12559-021-09910-0)
    * [BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram](https://arxiv.org/abs/2111.14558)
    * [Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning](https://www.mdpi.com/1424-8220/21/18/6022)
    * [PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks](https://arxiv.org/abs/2005.01669)
    * [Beat-to-Beat Continuous Blood Pressure Estimation Using Bidirectional Long Short-Term Memory Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7795062/)
    * [Cuffless Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring](https://pubmed.ncbi.nlm.nih.gov/27323356/)
    * [Continuous Blood Pressure Estimation Using Exclusively Photopletysmography by LSTM-Based Signal-to-Signal Translation](https://www.mdpi.com/1424-8220/21/9/2952)
    * [Generalized Deep Neural Network Model for Cuffless Blood Pressure Estimation with Photoplethysmogram Signal Only](https://www.mdpi.com/1424-8220/20/19/5668)


2. Features
    *

## EEG (Encephalogram)

1. Papers
    * [Automatic Human Sleep Stage Scoring Using Deep Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00781/full)
    * [SleepEEGNet: Automated sleep stage scoring with sequence to sequence deep learning approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6504038/) (https://arxiv.org/abs/1903.02108)
    * [Analysis and visualization of sleep stages based on deep neural networks](https://www.sciencedirect.com/science/article/pii/S2451994421000055)
    * [https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e](https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e)
    * [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/abs/1703.04046)


2. Features
    *


# Extra libs

 * [BioSPPy](https://github.com/PIA-Group/BioSPPy) - nice lib, PPG and ECG preprocessing available


# Representations

* Features - tabular features extracted from signals
* Whole signal waveforms - samples taken from whole signals (may be preprocessed)
* Per beat/window waveforms - samples taken from beats/windows extracted from signals
* Per beat/window features - features extracted from beats/windows
* Single beat/window waveforms - samples taken from aggregated beats/windows


# Models

* Support Vector Machine
* LGBM
* RNN (+ attn)
* CNN + RNN (+ attn)
* CNN
* Transformer
* CNN + Transformer
