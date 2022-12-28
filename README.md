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



# Datasets

1. PTB-XL

    * [source URL](https://physionet.org/content/ptb-xl/1.0.1/)
    * papers:
        * [PTB-XL, a large publicly available electrocardiography dataset](https://www.nature.com/articles/s41597-020-0495-6)
        * [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://ieeexplore.ieee.org/document/9190034)

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


2. MIMIC-III Waveform

    * [source URL](https://physionet.org/content/mimic3wdb/1.0/)
    * papers:
        * [Hypertension Assessment via ECG and PPG Signals: An Evaluation Using MIMIC Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163274/)

    * interesting urls:
        * [Subjects list](https://physionet.org/files/mimicdb/1.0.0/mimic-index.shtml)
        * [Cuff-Free Blood Pressure Estimation Using Pulse Transit Time and Heart Rate](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4512231/)
        * [The MIMIC-III Waveform Database](https://archive.physionet.org/physiobank/database/mimic3wdb/)

    * description:
        * TODO
        * Only some part of recordings have all 'V', 'ABP' and 'PLETH' signals
        * Some of 'PLETH' signals contains long `nan`, `0` or `1.0235` sequences

    * preprocessing:
        1. Find measurements with 'V', 'ABP' and 'PLETH' signals ("V" - ECG, "ABP" - Arterial Blood Pressure, "PLETH" - PPG Signal)
        2. Filter out bad segments.
        3. Filter out segments shorter than X seconds.

    * After preprocessing:
        * train: TODO
        * val: TODO
        * test: TODO

    * tasks:
        * **regression**: aggregated Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) predictions from PPG and ECG signals


3. Sleep-EDF
    * [source URL](TODO)
    * papers:
        * TODO

    * interesting urls:
        * TODO

    * description:
        TODO

    * preprocessing:
        TODO

    * After preprocessing:
        TODO

    * tasks:
        * **classification**: *sleep_stage*




# Signals

1. ECG (Electrocardiogram)
    * [A novel ECG signal classification method using DEA-ELM](https://www.sciencedirect.com/science/article/pii/S0306987719312381#b0085)
    * [A survey on ECG analysis](https://www.sciencedirect.com/science/article/pii/S1746809418300636#bib0175)
    * [Interpretable deep learning for automatic diagnosis of 12-lead electrocardiogram](https://www.sciencedirect.com/science/article/pii/S2589004221003412)
    * [Automatic diagnosis of the 12-lead ECG using a deep neural network](https://www.nature.com/articles/s41467-020-15432-4)

2. PPG (Photopletysmogram)
    * [Continuous PPG-Based Blood Pressure Monitoring Using Multi-Linear Regression](https://arxiv.org/abs/2011.02231)
    * [Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning](https://www.mdpi.com/1424-8220/21/18/6022)
    * [End-to-End Blood Pressure Prediction via Fully Convolutional Networks](https://ieeexplore.ieee.org/abstract/document/8936850)
    * [A Benchmark Study of Machine Learning for Analysis of Signal Feature Extraction Techniques for Blood Pressure Estimation Using Photoplethysmography (PPG)](https://ieeexplore.ieee.org/document/9558767)
    * [A Deep Learning Approach to Predict Blood Pressure from PPG Signals](https://arxiv.org/abs/2108.00099)
    * [A Comparison of Deep Learning Techniques for Arterial Blood Pressure Prediction ](https://link.springer.com/article/10.1007/s12559-021-09910-0)
    * [BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram](https://arxiv.org/abs/2111.14558)
    * [PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks](https://arxiv.org/abs/2005.01669)
    * [Beat-to-Beat Continuous Blood Pressure Estimation Using Bidirectional Long Short-Term Memory Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7795062/)
    * [Cuffless Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring](https://pubmed.ncbi.nlm.nih.gov/27323356/)
    * [Continuous Blood Pressure Estimation Using Exclusively Photopletysmography by LSTM-Based Signal-to-Signal Translation](https://www.mdpi.com/1424-8220/21/9/2952)
    * [Generalized Deep Neural Network Model for Cuffless Blood Pressure Estimation with Photoplethysmogram Signal Only](https://www.mdpi.com/1424-8220/20/19/5668)

3. EEG (Encephalogram)
    * [Automatic Human Sleep Stage Scoring Using Deep Neural Networks](https://www.frontiersin.org/articles/10.3389/fnins.2018.00781/full)
    * [SleepEEGNet: Automated sleep stage scoring with sequence to sequence deep learning approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6504038/) (https://arxiv.org/abs/1903.02108)
    * [Analysis and visualization of sleep stages based on deep neural networks](https://www.sciencedirect.com/science/article/pii/S2451994421000055)
    * [https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e](https://towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e)
    * [DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG](https://arxiv.org/abs/1703.04046)

3. EOG (Electrooculogram)
    * [TODO](TODO)


# Representations

1. Raw signal

    * whole signal sequence
    * signal split into windows
    * signal split into beats (for periodic signals like ECG, PPG)
    * signal split into beats and aggregated into single episode (sPPG, sPQRS)

2. Features extracted from signal as tabular data

    * features extracted for whole signal
    * features extracted for each window
    * features extracted for each beat (for periodic signals, like PPG, ECG)
    * features extracted for aggregated episode (for periodic signals), sPPG, sPQRS

3. Signal embedded with neural networks

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


# Features

* **Base**
    * **`BaseSignal`**
        * `basic`: mean, std, kurtosis, skewness, median, quantiles, zerocrossings, meancrossings, entropy, energy

    * **`Signal(BaseSignal)`**
        * `peaks`: `basic` features for peaks signal
        * `troughs`: `basic` features for troughs signal
        * `amplitudes`: `basic` features for amplitudes signal
        * `DWT`: Discrete Wavelet Transform features (`basic` features for all detail coeffs and last approx coeff)

    * **`PeriodicSignal(Signal)`**
        * `agg_beat`: `basic` + aggregated beat features dependent on signal type

    * **`BeatSignal(BaseSignal)`**
        * `crit_points`: critical points location, time and value for each critical point (dependent on signal type)
        * `area`: area features determined by critical points locations
        * `slope`: slope features determined by critical points locations
        * `energy`: energy features determined by critical points locations


* **ECG**
    * **`ECGSignal(PeriodicSignal)`**
        * `hrv`: hear rate variability features
        * `basic` for P/Q/R/S/T time locations and values
        * TODO: `basic` for PP/QQ/RR/SS/TT intervals


    * **`ECGBeat(BaseSignal)`**
        * `crit_points`: [p_onset, p, p_offst, r_onset, r, r_offset, s, t_onset, t, t_offset]
        * `area`: [p_onset, p_offset], [r_onset, r_offset], [t_onset, t_offset]
        * `slope`: [p_onset, p], [p, p_offset], [r_onset, r], [r, r_offset], [t_onset, t], [t, t_offset]
        * `energy`: [p_onset, p_offset], [r_onset, r_offset], [t_onset, t_offset]
        * `segments` for PQ/ST (segment is defined from prev_offset to next_onset)
        * `intervals` for PR/QRS/QT  (interval is defined from prev_onset to next_offset)


* **PPG**
    * **`PPGSignal(PeriodicSignal)`**
        * `hrv`: hear rate variability features

    * **`PPGBeat(BaseSignal)`**
        * `pulse_width`: pulse width features at different beat heights percantages
        * `pulse_height`: pulse height features at different beat duration percantages
        * `crit_points`: [systolic_peak, ]
        * `area`: [0, systolic_peak], [systolic_peak, end]
        * `slope`: [0, systolic_peak], [systolic_peak, end]
        * `energy`: [0, systolic_peak], [systolic_peak, end]


* **EOG**
    * **`EOGSignal`**
        * TODO


* **EEG**
    * **`EEGSignal`**
        * `frequency`: frequency features, i.e. mean frequency power for different frequency bands (delta, theta, alpha, sigma, beta)

# Models

* Support Vector Machine
* Random Forest
* MLP
* CNN
* LSTM


# References

* Packages
    * [Wavelets](https://pywavelets.readthedocs.io/en/latest/)

* Blogs:
    * [Wavelet Transform](https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/)
    * [Fourier Transform](https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/)

* Datasets
    * TODO

* Features:
    * Common:
        * [HRV](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/)
    * PPG:
        * [pulse width/height features](https://ieeexplore.ieee.org/document/9558767)
        * [Wavelet entropy](https://www.scirp.org/journal/paperinformation.aspx?paperid=112523)
        * [wavelet features](https://ieeexplore.ieee.org/document/6100581)
        * [HR features](https://www.researchgate.net/publication/319217053_Essential_Feature_Extraction_of_Photoplethysmography_Signal_of_Men_and_Women_in_Their_20s)
    * ECG
        * [ECGBeat features](https://www.sciencedirect.com/science/article/pii/S0169260715003314)
        * [Wavelet features](https://www.frontiersin.org/articles/10.3389/fncom.2017.00103/full)
    * EEG
        * [Wavelet features](https://downloads.hindawi.com/archive/2014/730218.pdf)
        * [Wavelet features](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6721346/)
        * [Complicated features](https://www.sciencedirect.com/science/article/pii/S1110016821007055)
        * [All features](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8198610/pdf/sensors-21-03786.pdf)
    * EOG
        * [Statistic features](https://www.hindawi.com/journals/cmmm/2014/713818/)
        * [Statistic features](https://www.researchgate.net/publication/257724368_Evaluating_Feature_Extraction_Methods_of_Electrooculography_EOG_Signal_for_Human-Computer_Interface)
        * [Statistic features](https://www.rsisinternational.org/journals/ijrsi/digital-library/volume-5-issue-4/287-290.pdf)

* Models
    * TODO


# In total:

* 3 datasets
* 4 representation types for 2 datasets (mimic, ptbxl) and 2 representation types for 1 dataset (sleep_edf)
* 5 models
