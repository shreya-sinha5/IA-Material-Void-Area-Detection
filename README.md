# IA-Material-Void-Area-Detection
Computer-vision based pixel analysis of a crushed Impact Attenuator (IA) to estimate material and void area ratios using grayscale segmentation and Otsu's automatic thresholding.

# IA Pixel Area Analysis using Otsu Thresholding

A computer-vision based image-processing pipeline for estimating the material-to-void area ratio of a crushed Impact Attenuator (IA) from a top-view photograph.

The project uses grayscale conversion and Otsu's automatic thresholding to classify image pixels into two regions:

- Material / Silver region
- Void / Black region

The number of pixels belonging to each class is then used to calculate the corresponding area ratios.

---

## Overview

After an Impact Attenuator is crushed, its internal mesh/structure can contain a large number of openings and deformed regions.

Manually calculating the remaining material area from a photograph is difficult and subjective.

This project provides a reproducible image-processing method:

```text
Original Image
      ↓
Grayscale Conversion
      ↓
Otsu Automatic Thresholding
      ↓
Binary Segmentation
      ↓
Pixel Classification
      ↓
Material / Void Pixel Count
      ↓
Area Ratio Calculation
