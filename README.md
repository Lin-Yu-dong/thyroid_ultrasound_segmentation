# Improving Segmentation of Thyroid Ultrasound Images Using Hybrid Deep Learning Models
The dataset used in model training and the best checkpoint for each of the six architectures are available in the [Releases](https://github.com/Lin-Yu-dong/thyroid_ultrasound_segmentation/releases).

## Dataset
We used the TN3K dataset,  originally released with [this repository](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation/tree/main/picture)   
**Reference:**  
[Gong H, Chen J, Chen G, Li H, Li G, Chen F. *Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules.* Computers in Biology and Medicine. 2023 Mar;155:106389.](https://www.sciencedirect.com/science/article/abs/pii/S0010482522010976?via%3Dihub).

---

## 6 segmentation architectures
This repository contains implementations of four baseline architectures and two hybrid deep learning models for thyroid ultrasound image segmentation.  
By combining Attention U-Net and TransUNet, we developed **HybridAttentionUNet**, which achieved better quantitative metrics than all baseline architectures.  

Furthermore, by incorporating MobileViT into HybridAttentionUNet, we proposed **HybridAttenTransMob**, which attained the best quantitative results among all six architectures and demonstrated superior robustness in handling complex segmentation tasks.  

Due to time and resource constraints, each architecture was trained once. Results may vary slightly across runs, and Test IoU / Test Dice can fluctuate by approximately ±3–4%. Nevertheless, the overall trend (**hybrid > baseline**) remains consistent.  

---

## Recommended Training Epochs
For each architecture, the maximum number of training epochs was chosen empirically based on training/validation loss and IoU curves. Training was stopped at the point where the training and validation curves began to diverge. In particular, this refers to the phenomenon where the training loss (or training IoU) continues to improve while the validation loss plateaus or the validation IoU fails to increase. Such divergence indicates overfitting, and the corresponding epoch was chosen as the recommended training limit.

- MobileViT: **400 epochs**  
- U-Net: **400 epochs**  
- Attention U-Net: **350 epochs**  
- TransUNet: **300 epochs**  
- HybridAttentionUNet: **100 epochs**  
- HybridAttenTransMob: **200 epochs**  

These recommended values are also included in the code comments.  

In practice, the number of epochs could also be determined by **early stopping**. Here, we manually set architecture-specific limits based on observed convergence.  

---

## Project Structure
The repository is organized as follows: 

thyroid_ultrasound_segmentation/   
├── ozan_oktay_Attention_UNet/ # Reference implementation of Attention U-Net (Oktay et al.)   
├── Beckschen_TransUNet/ # Reference implementation of TransUNet (Beckschen et al.)   
├── src/ # Core source code (Python scripts: dataset, losses, metrics, utils, mobileTrans, etc.)   
├── README.md # Project documentation   
├── 6_segmentation_architecture_1.ipynb # Reproducible Jupyter Notebook (part 1)   
├── 6_segmentation_architecture_2.ipynb # Reproducible Jupyter Notebook (part 2)   
├── dataset_TN3K/ # Dataset folder   
│ ├── test-image/   
│ ├── test-mask/   
│ ├── training_image/   
│ ├── training_mask/   
│ ├── validation_image/   
│ └── validation_mask/   
└── models/ # Best checkpoint for each of the six architectures (config.yml, model.pth)


**Notes:**
- The folders `ozan_oktay_Attention_UNet` and `Beckschen_TransUNet` are third-party code obtained from the authors’ official GitHub releases:  
  - [Attention U-Net](https://github.com/ozan-oktay/Attention-Gated-Networks)  
  - [TransUNet](https://github.com/Beckschen/TransUNet)  
- The `src` folder contains the main Python modules used for training and evaluation.  
- The Jupyter notebooks provide step-by-step reproducible experiments.  
- Within `src`, the default dataset loader is `dataset.py`. An alternative implementation `dataset_alt.py` is also provided and may occasionally be used.  

---

## Environment

This code was tested with:
- Python 3.11
- PyTorch 2.7.1
- CUDA 11.8
- Torchvision 0.22.1

---

## Quantitative Results of 6 Architectures

### Baseline Architectures

| Network        | Publish        | Architecture Category | Input size | Best model epoch / All epoch | Test IoU | Test Dice | Improvement of Dice | Trainable parameters |
|----------------|----------------|------------------------|-----------|------------------------------|----------|-----------|---------------------|----------------------|
| U-Net          | MICCAI (2015)  | CNN                    | 224×224   | 47/50                        | 0.6904   | 0.7836    | 0.0325              | 2,465,329            |
|                |                |                        | 224×224   | 490/500                      | 0.7291   | 0.8161    |                     |                      |
| Attention U-Net| MIDL (2018)    | CNN                    | 224×224   | 49/50                        | 0.6219   | 0.7271    | **0.0988 (highest)**| 2,466,789            |
|                |                |                        | 224×224   | 459/500                      | 0.7387   | **0.8259**|                     |                      |
| MobileViT      | ICLR (2022)    | Hybrid CNN–Transformer | 224×224   | 45/50                        | 0.6253   | 0.7363    | 0.0805              | 8,171,227            |
|                |                |                        | 224×224   | 495/500                      | 0.7268   | 0.8168    |                     |                      |
| TransUNet      | MIA (2021)     | Hybrid CNN–Transformer | 224×224   | 49/50                        | 0.6967   | 0.7938    | 0.0232 (lowest)     | 8,544,321            |
|                |                |                        | 256×256   | 442/500                      | 0.7277   | 0.8170    |                     |                      |

---

### Hybrid Architectures

| Hybrid Network                        | Input size | Best model epoch / All epoch | Test IoU | Test Dice | Trainable parameters |
|---------------------------------------|------------|------------------------------|----------|-----------|-----------------------|
| Hybrid_Attention_TransUNet            | 224×224    | 47/50                        | 0.7109   | 0.8026    | 49,882,161            |
|                                       | 224×224    | 479/500                      | 0.7552   | 0.8390    |                       |
| Hybrid_Attention_TransUNet_MobileViT  | 224×224    | 48/50                        | 0.7008   | 0.7936    | 51,896,017            |
|                                       | 224×224    | 300/500                      | 0.7640   | **0.8460**|                       |



