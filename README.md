# DL-project-Neural-Style-Transfer
🎨 Neural Style Transfer using VGG19 (Fine-Tuned)
This project implements **Neural Style Transfer (NST)** using a **fine-tuned VGG19 model**. The NST blends the **content of one image** with the **style of another**, generating visually aesthetic outputs.

# 📌 Project Overview
Neural Style Transfer uses a deep Convolutional Neural Network (CNN) to extract **content features** from one image and **style features** from another. In this project, we **partially fine-tuned a pre-trained VGG19 model** on our dataset to improve stylization, then saved and reloaded the model for inference.

## 🖼️ Example
![lion](https://github.com/user-attachments/assets/23e20e6b-77fa-4564-8139-3c309c619525)+
![style](https://github.com/user-attachments/assets/749f02cf-2a4f-4872-90f1-e45a3781b558)=
![stylized_output](https://github.com/user-attachments/assets/a49f9423-73d9-455b-8172-782094a46254)

## 📁 Directory Structure
```
├── app.py                          # Streamlit GUI for interactive NST
├── trained_model.ipynb             # Core logic and training notebook
├── vgg19_finetuned_nst_model.h5    # Saved fine-tuned VGG19 model
├──datasets                         # Datasets for training
    ├── ContentImages
    ├── StyleImages
```
## ⚙️ Features

- Load custom content and style images.
- **Fine-tune VGG19** for improved stylization.
- Save and **load the trained model**.
- Stylize using a GUI built with **Streamlit**.
- Visualize intermediate results and losses.

## 🚀 How to Run
### 1. Clone the Repository
   ```bash
   git clone https://github.com/yourusername/neural-style-transfer
   cd neural-style-transfer
   ```
### 2. Run Jupyter Notebook
   You can also explore training and fine-tuning in the Jupyter Notebook.
### 3. Loading the Fine-Tuned Model in app.py file
   The fine-tuned model is saved as fine_tuned_model.h5. It is automatically loaded in the app/notebook before stylizing the image.
   ```bash
   from tensorflow.keras.models import load_model
   model = load_model("fine_tuned_model.h5", compile=False)
   ```
### 4. Run the Streamlit App
   ```bash
   streamlit run app.py
   ```
## 📚 Technologies Used

-Python
-TensorFlow / Keras
-NumPy
-Matplotlib
-Streamlit (GUI)
-VGG19 (with partial fine-tuning)


## 📄 References

- [A Neural Algorithm of Artistic Style – Gatys et al.](https://arxiv.org/abs/1508.06576)
- TensorFlow Style Transfer Tutorials
- [NST Reasearch paper]-(https://www.researchgate.net/publication/333702745_Neural_Style_Transfer_A_Review).

## ✨ Acknowledgements
Thanks to the open-source community and foundational deep learning research for making such projects possible.

