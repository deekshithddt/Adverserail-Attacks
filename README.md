# Adversarial Attacks on Pix2Pix Image-to-Image Translation

## Overview
This repository implements a Pix2Pix Conditional Generative Adversarial Network (cGAN) for image-to-image translation, along with pretrained model weights and prediction code.

The project focuses on:

- Training Pix2Pix model  
- Generating translated images (e.g., face → sketch)  
- Loading pretrained model for inference  
- Studying adversarial attacks on generative models  

This repository can be used for research in:

- Image-to-image translation  
- Generative Adversarial Networks (GANs)  
- Adversarial machine learning  
- Computer vision  

---

## Repository Structure

```
Adverserail-Attacks/
│
├── pixtopix (1).ipynb        # Training and prediction notebook
├── pix2pix_sketch_G.pth     # Pretrained Pix2Pix generator model
├── README.md                # Project documentation
```

---

## Pix2Pix Architecture

Pix2Pix uses a Conditional GAN consisting of:

### Generator (U-Net)
- Encoder–decoder architecture  
- Skip connections preserve image details  
- Converts input image → output image  

### Discriminator (PatchGAN)
- Classifies image patches as real or fake  
- Improves image quality and sharpness  

---

## Loss Function

Pix2Pix uses combined loss:

```
Total Loss = GAN Loss + λ × L1 Loss
```

Where:

- GAN Loss → improves realism  
- L1 Loss → preserves structural similarity  

---

## Requirements

Install required libraries:

```
pip install torch torchvision numpy matplotlib pillow opencv-python tqdm scikit-image jupyter
```

---

## How to Run the Project

### Step 1: Clone repository

```
git clone https://github.com/deekshithddt/Adverserail-Attacks.git
cd Adverserail-Attacks
```

---

### Step 2: Open Notebook

```
jupyter notebook
```

Open:

```
pixtopix (1).ipynb
```

---

### Step 3: Train the Model

Run all cells to:

- Load dataset  
- Train Pix2Pix model  
- Save trained model  

Output model:

```
pix2pix_sketch_G.pth
```

---

### Step 4: Run Prediction

Load pretrained model:

```python
model.load_state_dict(torch.load("pix2pix_sketch_G.pth"))
model.eval()
```

Generate output:

```python
output = model(input_image)
```

---

## Model File

```
pix2pix_sketch_G.pth
```

Contains trained weights of Pix2Pix generator.

---

## Applications

- Face to sketch generation  
- Image translation  
- Autonomous driving image processing  
- Medical image translation  
- Adversarial attack research  

---

## Future Work

- Implement FGSM attack  
- Implement PGD attack  
- Evaluate robustness of Pix2Pix  
- Improve image quality  

---

## Author

Deekshith D  

---

## License

This project is for academic and research purposes.

---

## Acknowledgment

Based on Pix2Pix paper:

Image-to-Image Translation with Conditional Adversarial Networks  
Phillip Isola et al.

---

⭐ If you find this project useful, please star the repository.
