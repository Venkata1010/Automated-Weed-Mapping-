# Automated Weed Mapping for Spot Spraying (U-Net / U-Net++)

Semantic segmentation pipeline to detect weeds in field RGB imagery for **precision spot-spraying**. Built on the **DeepWeeds** dataset and implemented in **TensorFlow/Keras** with **OpenCV** preprocessing and **CVAT** annotation. The project targets UAV-ready inference and integration with targeted spraying systems.

---

## Overview

Weed infestation reduces yield and drives up herbicide costs when fields are broadcast-sprayed. This project develops a **pixel-wise weed segmentation** approach so only infested regions are treated. The core model is a lightweight **U-Net** tuned for small data; a deeper **U-Net++** variant was also explored to assess benefits vs. overfitting under limited data. Results show U-Net can generalize well even with a compact training set, making it practical for edge deployment on agricultural robots/UAVs.

**Highlights**
- Dataset: **DeepWeeds** (RGB images with 8 weed species across Australia; field conditions)  
- Model: **U-Net** (binary: weed vs. background), with **BCE + Dice loss**  
- Metrics: **Mean IoU ≈ 0.5757**, **Pixel Accuracy ≈ 0.8378** on validation  
- Generalization: Good visual performance on an external **2016 image set**  
- Exploration: **U-Net++** multiclass prototype (nested skip connections) for future scale-up.

---

## Dataset

- **DeepWeeds**: 17,509 RGB images across 8 invasive species with real-world variability (lighting, occlusion).  
- This work used a curated subset with **45 manually annotated images** (after removing negatives) for training/validation; an external **2016 image set** was used for additional testing.  
- Masks are grayscale binary (weed=1, background=0), resized to **512×512** for U-Net. :contentReference[oaicite:5]{index=5}

> Citation (dataset): Olsen et al., *DeepWeeds: A multiclass weed species image dataset for deep learning*, **Scientific Reports** (2019). DOI in sources. :contentReference[oaicite:6]{index=6}

---

## Method

**Preprocessing & Annotation**
- Images resized to **512×512** (U-Net) or **256×256** (U-Net++ prototypes).  
- Masks thresholded and converted to float32; image/mask pairing verified.  
- Annotation via **CVAT**; three empty images excluded → **45** images finalized. :contentReference[oaicite:7]{index=7}

**Modeling**
- **U-Net (binary)**: encoder–decoder with skip connections; ReLU convs and sigmoid output.  
- **Loss**: Binary Cross-Entropy + Dice (mitigates foreground sparsity).  
- **Training**: 80/20 split (train/val), batch size 2, up to 50 epochs with **EarlyStopping** (patience 10).  
- **U-Net++ (multiclass)**: nested skip connections with focal loss + custom mIoU/pixel-accuracy; overfit under small data and higher compute needs, so treated as exploratory. :contentReference[oaicite:8]{index=8}

---

## Results

- **Validation**: **Pixel Accuracy ≈ 0.8378**, **Mean IoU ≈ 0.5757**.  
- **External test (2016 set)**: qualitatively consistent masks indicating effective generalization despite limited training data.  
- **U-Net++**: training time ~3× U-Net and early overfitting (~epoch 10) under current data/compute; retained for future scale-up. :contentReference[oaicite:9]{index=9}

---

## Repository Structure

