#  Parameter-Efficient Adaptation of TruFor for Image Forgery Detection Using LoRA

##  Project Overview
This project focuses on **image forgery detection and localization** using **TruFor**, a transformer-based forensic model for detecting manipulated image regions.  
The main goal of this work is to make TruFor more efficient and better adapted for **modern AI-generated and diffusion-based image forgeries** by using **LoRA (Low-Rank Adaptation)**.

Instead of fully retraining the entire TruFor network, this project applies **parameter-efficient fine-tuning** so that only a small portion of the model is updated. This reduces training cost while still improving localization performance.

---

##  Problem Statement
Traditional image forgery detection models are often computationally expensive to fine-tune and may not be specifically adapted to **diffusion-based manipulations** such as:

- object insertion
- inpainting
- AI-edited regions
- softly blended synthetic content

The original **TruFor** model is strong for generic forgery localization, but it was not explicitly adapted for these modern generative manipulations.  
At the same time, full fine-tuning is costly and not practical for limited-resource environments like **Kaggle GPUs**.

---

##  Solution Approach
To address this, this project proposes a **LoRA-based parameter-efficient adaptation of TruFor**.

### What was done
- Loaded the **original pretrained TruFor** model as the baseline
- Identified **transformer attention modules** suitable for LoRA injection
- Added **LoRA adapters** to selected attention projections
- Froze most of the original TruFor model
- Fine-tuned only:
  - **LoRA adapters**
  - **decoder heads**
- Trained the adapted model on a diffusion-style forgery dataset

This approach preserves the strong forensic prior of TruFor while making it more specialized for modern image manipulations.

---

##  Technologies Used
- **Python**
- **PyTorch**
- **PEFT (Parameter-Efficient Fine-Tuning)**
- **LoRA (Low-Rank Adaptation)**
- **Transformers / SegFormer-style backbone**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Pandas**
- **Kaggle Notebooks**
- **Mixed Precision Training (AMP)**

---

## 📂 Dataset
This project used the **CocoGlide** dataset in the Kaggle environment.

### Dataset structure
- `fake/` → forged images
- `mask/` → ground-truth masks
- `real/` → authentic images
- `table.csv` → file mapping information

### Kaggle usage
The dataset was loaded and processed directly in **Kaggle notebooks**, where training and evaluation were performed on GPU.

> https://www.kaggle.com/datasets/longnguyen892001/cocoglide

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Read file paths using `table.csv`
- Built separate samples for:
  - forged images with mask
  - real images with zero mask
- Resized all images and masks to **512 × 512**
- Created custom PyTorch datasets and dataloaders

### 2. Baseline Model
- Loaded the original pretrained **TruFor** checkpoint
- Verified image-level heatmap generation and forged-region localization

### 3. LoRA Adaptation
LoRA was applied to selected transformer attention modules such as:
- `q`
- `kv`
- `proj`

The model was adapted using:
- `r = 8`
- `lora_alpha = 32`

### 4. Trainable Components
Most of the original model was frozen.  
Only the following components were trainable:
- LoRA adapters
- `decode_head`
- `decode_head_conf`

### 5. Training
Training was performed using:
- **batch size = 1**
- **mixed precision (AMP)**
- **gradient accumulation**
- **AdamW optimizer**

### 6. Evaluation
The model was evaluated using:
- **Dice Score**
- **IoU**
- **Precision**
- **Recall**
- **Accuracy**
- **False positive behavior on real images**
- **Threshold sweep analysis**

---

##  Model Architecture

### Baseline TruFor
```text
Input Image
   ↓
Dual Forensic Feature Extraction
   ↓
Transformer / SegFormer-style Backbone
   ↓
Decoder Heads
   ↓
Forgery Heatmap + Localization Mask
```

### Proposed LoRA-Adapted TruFor
```text
Input Image
   ↓
Original TruFor Backbone
   + LoRA adapters in selected attention layers
   ↓
Fine-tuned Decoder Heads
   ↓
Diffusion-aware Forgery Heatmap + Localization Mask
```

### Architecture Diagram
![Architecture Diagram](assets/images/architecture%20of%20our%20model.png)


---

## 📊 Results
![Accuracy Matrix](assets/images/evaluation%20matrix.png)
![Threshhold](assets/images/Screenshot%202026-04-16%20013243.png)

### Training Observation
Training and validation loss decreased steadily, and the best validation checkpoint was obtained around **epoch 4** in the larger experiment.

### Quantitative Results on Fake Images
| Metric | Baseline | LoRA-Adapted | Improvement |
|--------|---------:|-------------:|------------:|
| Dice | 0.395963 | 0.588028 | +0.192065 |
| IoU | 0.282662 | 0.465477 | +0.182815 |
| Precision | 0.282689 | 0.550688 | +0.267999 |
| Recall | 0.999870 | 0.774199 | -0.225671 |
| Accuracy | 0.282690 | 0.783954 | +0.501264 |

### Real Image Behavior
| Metric | Baseline | LoRA-Adapted |
|--------|---------:|-------------:|
| Predicted Forged Ratio | 0.999973 | 0.245164 |
| Predicted Forged Pixels | 262136.84 | 64268.17 |




### Interpretation
- The **baseline** tends to over-segment and mark almost everything as forged
- The **LoRA-adapted model** becomes more selective
- Dice, IoU, Precision, and Accuracy improve significantly
- False positives on real images are drastically reduced

---

## 🖼 Sample Outputs

The following visualizations compare the **Baseline TruFor** vs **LoRA-Adapted TruFor** on 5 sample images from the CocoGlide dataset.  
Each row shows: **Original Image | Ground Truth Mask | Baseline Heatmap | LoRA Heatmap | Overlay**

### Sample 1
![Sample Output 1](assets/images/Screenshot%202026-04-16%20014852.png)

### Sample 2
![Sample Output 2](assets/images/Screenshot%202026-04-16%20014933.png)

### Sample 3
![Sample Output 3](assets/images/Screenshot%202026-04-16%20014948.png)

### Sample 4
![Sample Output 4](assets/images/Screenshot%202026-04-16%20015004.png)

### Sample 5
![Sample Output 5](assets/images/Screenshot%202026-04-16%20015101.png)
> **Key observations:**
> - Baseline often produces broad or noisy suspicious regions
> - LoRA-adapted model gives cleaner and more object-focused localization
> - Strong improvement is especially visible in difficult diffusion-style manipulations

---

## 📁 Project Structure
```text
project-root/
│
├── notebooks/
│   ├── training_and_evaluation.ipynb
│   ├── qualitative_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── model_adaptation.py
│   ├── train.py
│   ├── evaluate.py
│
├── figures/
│   ├── architecture.png
│   ├── accuracy_matrix.png
│   ├── graphs/
│   └── qualitative_examples/
│       ├── sample_1.png
│       ├── sample_2.png
│       ├── sample_3.png
│       ├── sample_4.png
│       └── sample_5.png
│
├── README.md
└── requirements.txt
```

---

##  How to Run the Project

### 1. Clone the repository
```bash
git clone <your-github-repo-link>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open notebook / script
You can run the project either:
- from a **Kaggle notebook**
- or from local Jupyter / Python environment

### 4. Prepare dataset
- Attach or download the **CocoGlide** dataset
- Ensure `fake/`, `real/`, `mask/`, and `table.csv` are available

### 5. Load baseline TruFor
- Load the original checkpoint
- Verify baseline heatmap generation

### 6. Apply LoRA adaptation
- Insert LoRA into selected attention layers
- Freeze base model
- Fine-tune LoRA + decoder heads

### 7. Train the model
Run the training notebook or training script with the required paths and GPU setup.

### 8. Evaluate
- Run metric evaluation on fake images
- Run false positive analysis on real images
- Generate threshold sweep graphs and qualitative comparisons

---

##  GitHub Notebook Link
Add your GitHub notebook link here:

```text
https://github.com/Pawankumar1188/Parameter-Efficient-Adaptation-of-TruFor-for-Image-Forgery-Detection-Using-LoRA/blob/main/notebooks/training_and_evaluation%20final.ipynb
```

---


## 🚀 Future Improvements
Possible extensions of this project include:

- testing on larger and more diverse diffusion-forgery datasets
- comparing LoRA with other parameter-efficient fine-tuning methods
- adding forgery-type classification along with localization
- improving threshold calibration
- reducing false positives further on real images
- performing stronger ablation studies
- evaluating cross-dataset generalization

---

##  Conclusion
This project demonstrates that **LoRA-based parameter-efficient fine-tuning** can successfully adapt TruFor for modern image forgery localization tasks.  
Without retraining the full model, the adapted version improves **Dice, IoU, Precision, and Accuracy**, while significantly reducing false positives on real images.

In short, this work shows that **TruFor + LoRA** is a practical and efficient approach for diffusion-aware image forgery detection and localization.

---

##  Acknowledgment
This project builds upon the original **TruFor** architecture and extends it through **LoRA-based lightweight adaptation** for efficient fine-tuning and improved performance.

---

## 📧 Contact

```text
Group - 236
Name: Pawan Kumar 2023UCS1619
Name: Shubham Chakma 2023UCS1615
```
