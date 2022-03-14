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





3. MIMIR-III Waveform

    * [source URL](https://physionet.org/content/mimic3wdb/1.0/)
    * preprocessing:
        1.
        2.
    * papers:
        *
        *
    * interesting urls:
        * [The MIMIC-III Waveform Database](https://archive.physionet.org/physiobank/database/mimic3wdb/)


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



# Extra libs:

 * [BioSPPy](https://github.com/PIA-Group/BioSPPy) - nice lib, PPG and ECG preprocessing available
