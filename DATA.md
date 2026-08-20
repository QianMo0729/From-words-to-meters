# Data

This document describes the datasets used in the **From Words to Meters**
project, including their sources, directory structure, preprocessing steps,
and storage conventions.

> Large datasets are not tracked by Git.
> Follow the instructions below to download and prepare the required data.

---

## 1. Overview

The project uses RGB-D indoor-scene datasets to study the relationship between visual observations, language descriptions, and metric 3D spatial information.

Current datasets:

| Dataset | Status | Purpose |
| --- | --- | --- |
| SUN RGB-D | Primary | RGB-D indoor scene data |

---

## 2. SUN RGB-D

SUN RGB-D is an indoor RGB-D scene understanding dataset containing aligned:

- RGB images
- depth maps
- camera calibration information
- 2D annotations
- 3D annotations
- semantic labels

It is used as the primary dataset for the current experiments.

Official dataset download:

<https://rgbd.cs.princeton.edu/data/SUNRGBD.zip>

Original page:

<https://rgbd.cs.princeton.edu/data/>

---

## 3. Access and License

Before downloading or using the dataset, check the official dataset website for the latest license and terms of use.

The dataset itself is **not redistributed in this repository**.

Only project-specific metadata, scripts, and small examples may be committed.

---

## 4. Download

Download the original dataset from the official source.

Place the downloaded data under:

```text
data/raw/SUNRGBD/
```