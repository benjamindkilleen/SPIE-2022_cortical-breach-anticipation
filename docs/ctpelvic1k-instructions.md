# CTPelvic1K Annotation Instructions

These are instructions to download/annotate K-Wire trajectories in the CT volumes of the CTPelvic1K dataset. If you just want to download the existing annotations, use the `download()` function provided in the `CTPelvic1K` class.

First, download the data to `~/datasets/CTPelvic1K`, either by following [this link](https://zenodo.org/record/4588403#.YEyLq_0zaCo) or running

```bash
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset6_Anonymized_mask.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset6_data.tar.gz
tar -xvf *.tar.gz
```

Which are the essential files. The remaining files, if you wish, can be downloaded with

```bash
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset1_mask_mappingback.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset2_mask_mappingback.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset3_mask_mappingback.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset4_mask_mappingback.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset5_mask_mappingback.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset7_data.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_dataset7_mask.tar.gz
wget https://zenodo.org/record/4588403/files/CTPelvic1K_Models.tar.gz
```

but it is only necessary to obtain the files for `dataset6` for our purposes.

Now, download existing annotations using

```bash
wget https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ETZqDv6bzy5MkNzX3Opg1y0BBvBZrPpz72812IaKAHHwWQ?e=37vFz8\&download\=1 -O CTPelvic1K_dataset6_trajectories.zip
unzip CTPelvic1K_dataset6_trajectories.zip
```

After unzipping both files, verify the directory structure (e.g. with `tree`):

```txt
.
├── CTPelvic1K_dataset6_data
│   ├── dataset6_CLINIC_0001_data.nii.gz
│   ├── dataset6_CLINIC_0002_data.nii.gz
|   ├── dataset6_CLINIC_0003_data.nii.gz
│   ├── ...
│   └── dataset6_CLINIC_0103_data.nii.gz
├── CTPelvic1K_dataset6_data.tar.gz
└── CTPelvic1K_dataset6_trajectories
    ├── dataset6_CLINIC_0001_left_kwire_trajectory.mrk.json
    ├── dataset6_CLINIC_0001_right_kwire_trajectory_fractured.mrk.json
    └── ...
```

# Annotating a Trajectory

Annotating trajectories requires a recent version of 3D Slicer. Download [here](https://download.slicer.org/)

Upon opening Slicer, load the data:

1. ![open data](_static/open_data.png)
