# VISYN

### Deep Visual Intelligence for Quality Inspection

VISYN is a production-oriented computer vision system for automated industrial visual quality inspection.

It detects visual deviations in industrial objects by learning the visual characteristics of normal samples and comparing incoming images against those learned representations.

The project combines deep visual representations, spatial patch-level features, reference-bank anomaly detection, calibrated decision-making, anomaly localization, CPU optimization, REST API serving, Docker containerization, and deployment preparation.

---

## Overview

Industrial inspection systems often need to identify subtle visual deviations from an expected normal appearance.

Instead of training a conventional classifier that requires labeled examples of every possible defect, VISYN models the visual characteristics of normal objects and detects deviations from that normal reference distribution.

The core idea is:

Normal Images  
↓  
Deep Visual Representation  
↓  
Reference Bank  
↓  
New Image  
↓  
Visual Similarity Comparison  
↓  
Anomaly Score  
↓  
PASS / REVIEW / FAIL

This makes VISYN suitable for inspection scenarios where normal examples are easier to obtain than exhaustive examples of every possible defect.

---

## High-Level Production Pipeline

```text
┌─────────────────────┐
│     Input Image     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Preprocessing    │
│     224 × 224       │
│  ImageNet Normalize │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│      MobileNetV3-Small      │
└──────────────┬──────────────┘
               │
          ┌────┴────┐
          ▼         ▼
       L4 Features  L8 Features
          │         │
          ▼         ▼
      196 Patches  196 Patches
          │         │
          ▼         ▼
     L2 Normalize L2 Normalize
          │         │
          └────┬────┘
               ▼
        Reference Bank
         Normal Samples
               │
               ▼
       Patch Distance Matching
               │
               ▼
          TOP-5 Mean
               │
               ▼
          50/50 Fusion
           L4 + L8
               │
               ▼
     Category-Specific P99
          Calibration
               │
        ┌──────┼──────┐
        ▼      ▼      ▼
      PASS   REVIEW   FAIL
               │
               ▼
      Spatial Anomaly Evidence
               │
               ▼
       Heatmap / Mask / BBox