# GitHub Upload Instructions

1. Create a new GitHub repository, for example:

   `eeg-vr-dataset-code`

2. Keep the repository public when the paper is submitted.

3. Do **not** upload raw identifiable or sensitive EEG files to GitHub. Upload raw and processed data to a data repository such as Zenodo/Figshare/OpenNeuro and keep GitHub for code and documentation.

4. From this folder, run:

```bash
git init
git add .
git commit -m "Initial EEG-VR dataset processing code"
git branch -M main
git remote add origin https://github.com/Saif-aljumaili/EEG-VR.git
git push -u origin main
```

5. On GitHub, add topics such as:

```text
eeg, virtual-reality, working-memory, sports-science, matlab, scientific-data, data-descriptor
```

6. After the repository is public, archive a version in Zenodo to obtain a software DOI.
