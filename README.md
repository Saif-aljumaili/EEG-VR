# EEG-VR Dataset Code

This repository contains reproducible code for the EEG-VR Data Descriptor paper:

**A seven-channel EEG dataset acquired during immersive virtual-reality working-memory assessment in elite triathletes and sedentary adults**

The code reads raw EEG CSV files exported from the DSI-7 system, performs basic quality control, applies a 0.5–40 Hz band-pass filter, segments the data into 1-second epochs, computes FFT-based theta-band summaries, and generates paper-ready quality-control outputs.

> **Important:** Raw human EEG CSV files should normally be deposited in a data repository such as Zenodo/Figshare/OpenNeuro, not committed directly to GitHub. This GitHub repository should contain code, metadata templates, documentation, and example outputs.

## Repository structure

```text
eeg-vr-dataset-code/
  python/
    run_eeg_vr_pipeline.py
  matlab/
    run_eeg_vr_pipeline.m
  data/
    metadata/
      participants_template.csv
      channel_map.csv
  results/
    sample_run/
      tables/
      figures/
  docs/
    code_availability_statement.md
    github_upload_instructions.md
  README.md
  requirements.txt
  CITATION.cff
  LICENSE
```

## Expected raw data folder structure

The pipeline expects the raw EEG data to be organized like this:

```text
RAW EEGS/
  raw_eegs_control/
    baseline/
      *.csv
    vr/
      *.csv
  raw_eegs_triathletes/
    baseline/
      *.csv
    vr/
      *.csv
```

## Common EEG columns

The code automatically detects these EEG channels when present:

```text
PzLE, F4LE, C4LE, P4LE, P3LE, C3LE, F3LE
```

Timestamp and trigger columns are used when available:

```text
DeviceTimeStamp, DeviceTimeUnixTimeStamp, SystemTimeUnixTimeStamp, SystemTime, Trigger
```

Some VR CSV exports may not include timestamp or trigger columns. In such cases, the code uses a default sampling rate of 300 Hz, which should be confirmed from the device documentation or acquisition export settings.

## Python quick start

```bash
pip install -r requirements.txt
python python/run_eeg_vr_pipeline.py --raw-root "D:/EEG_VR_Prof.Adil/Saif/RAW EEGS" --out-root results/full_run --fs 300
```

The output will include:

```text
quality_control_summary.csv
channel_raw_statistics.csv
theta_power_summary.csv
representative_eeg_segment.png
representative_psd_theta_band.png
sample_theta_summary.png
```

## MATLAB quick start

Open MATLAB, set this repository as the working folder, then run:

```matlab
rawRoot = 'D:\EEG_VR_Prof.Adil\Saif\RAW EEGS';
outRoot = fullfile(pwd, 'results', 'matlab_full_run');
run('matlab/run_eeg_vr_pipeline.m');
```

## Recommended use in the paper

Use the outputs as part of the Data Descriptor's **Technical Validation** and **Usage Notes** sections. Do not overinterpret the sample output as inferential results unless the full dataset and statistical model are used.

## Citation

Please cite the dataset DOI and the associated Data Descriptor when available.
