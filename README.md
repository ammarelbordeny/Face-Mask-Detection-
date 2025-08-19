# 😷 Face Mask Detection using CNN

![Project Status](https://img.shields.io/badge/Status-Complete-green)
![Python Version](https://img.shields.io/badge/Python-3.9+-blue)
![Dependencies](https://img.shields.io/badge/Dependencies-TensorFlow%2C%20OpenCV%2C%20Streamlit-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

## 📌 Project Overview

This project is a real-time **Face Mask Detection** system built with a **Convolutional Neural Network (CNN)**. The system processes live video feeds from a webcam to automatically detect faces and classify whether a person is wearing a face mask or not. The project provides a user-friendly web interface using **Streamlit** to demonstrate its functionality.

## 🚀 Key Features

* **Real-time Detection:** The system processes live video streams from a webcam for instant detection.
* **Custom CNN Model:** The core of the project is a powerful CNN model, trained from scratch using **TensorFlow** and **Keras** to achieve high accuracy.
* **User-friendly Interface:** A simple and intuitive web application, built with **Streamlit**, allows users to test the system directly through their browser.
* **Pre-trained Model:** The repository includes the pre-trained model file (`mask_model.h5`), so you can run the application immediately without needing to train the model yourself.


## 📂 Project Files Breakdown

* `Face_Mask_Detection.ipynb`: This Jupyter Notebook contains all the steps of the machine learning pipeline, including **data preprocessing**, **model architecture**, **training**, and **evaluation**. It serves as the main documentation for how the model was created.
* `app.py`: This Python script is the heart of the web application. It uses the trained model (`mask_model.h5`) to perform real-time detection and presents the output via **Streamlit**.
* `mask_model.h5`: This is the **trained CNN model file**. It's a key component that the `app.py` script relies on to perform predictions.
* `requirements.txt`: This file lists all the necessary Python libraries and their specific versions required to run the project.

## 🛠️ How to Run the Application

To run the application, follow these simple steps. Make sure you have **Python** installed on your system.

1.  **Install Required Libraries:** Open your terminal and run this command to install all the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the Application:** After installation, run the Streamlit application with the following command:
    ```bash
    streamlit run app.py
    ```
    This will launch a new tab in your web browser with the real-time detection interface.

## 📸 Demo

Here are some examples showing the model in action.

****
**Example 1: Person with a mask**
<img src="https://github.com/ammarelbordeny/Face-Mask-Detection-/blob/main/images/WhatsApp%20Image%202025-08-19%20at%2015.02.45_04d94ec7.jpg" alt="Person with mask" width="500">

**Example 2: Person without a mask**
<img src="https://github.com/ammarelbordeny/Face-Mask-Detection-/blob/main/images/WhatsApp%20Image%202025-08-19%20at%2015.02.45_7ddebb7a.jpg" alt="Person without mask" width="500">
****

## 💾 Dataset

The model was trained on the **[Face Mask Detection dataset from Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)**, which is a high-quality dataset containing thousands of images.

---
*Developed by Ammar Ahmed*
