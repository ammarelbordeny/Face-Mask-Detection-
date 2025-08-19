# Face Mask Detection  

This project is a **Face Mask Detection System** built using **Deep Learning (CNN)**.  
It detects whether a person is wearing a mask or not in real time using the webcam or images.  

---

## 📂 Project Structure  

- `app.py` → Streamlit app to run the model.  
- `notebook.ipynb` → Jupyter/Colab notebook with model training code.  
- `requirements.txt` → List of dependencies.  
- `model.h5` → Saved trained model (⚠️ not uploaded here because of file size limit).  
- `data/` → Dataset (not uploaded here, see Kaggle link below).  

---

## 📊 Dataset  

The dataset is taken from **Kaggle**:  
👉 [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  

You can download it using the Kaggle API inside Google Colab:  

```bash
!kaggle datasets download -d omkargurav/face-mask-dataset
!unzip face-mask-dataset.zip -d data/
