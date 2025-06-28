# 🛡️ Network Intrusion Detection System (IDS) Using Machine Learning

A machine learning-based Intrusion Detection System built on the CIC-IDS2017 dataset to classify network traffic as benign or malicious. This project demonstrates end-to-end ML implementation: preprocessing, feature engineering, training, evaluation, and model inference.

---

## 📌 Project Highlights

* ✅ Utilizes the **CIC-IDS2017** dataset
* ✅ Preprocessing and cleaning of flow-based CSVs
* ✅ Binary classification: **Benign vs Malicious**
* ✅ Built using **Python**, **Scikit-learn**, **XGBoost**, and **LightGBM**
* ✅ Modular Jupyter Notebooks: Preprocessing, Training, Evaluation, Inference
* ✅ Ready to extend to multi-class or real-time detection

---

## 📂 Project Structure

```
Network-IDS-Project/
├── data/                         # 🚫 Not committed — contains raw dataset
│   └── GeneratedLabelledFlows/
├── models/                       # Saved trained models (.pkl)
├── src/
│   ├── preprocess.ipynb          # Data loading & cleaning
│   ├── train.ipynb               # Model training & selection
│   ├── evaluate.ipynb            # Evaluation and metrics reporting
│   └── inference.ipynb         # Run inference using saved model
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset Overview

* **Name:** CIC-IDS2017
* **Source:** [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
* **Format:** Flow-based `.csv` files
* **Labels:** Multiple attack types grouped into:

  * 🔵 Benign
  * 🔴 Malicious

### 📅 Getting the Data

Download `GeneratedLabelledFlows.zip` from the link above and extract into:

```
data/GeneratedLabelledFlows/TrafficLabelling/
```

Ensure that the path contains 8 CSV files, one per day.

---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
```

---

## 💡 Preprocessing Steps (preprocess.ipynb)

1. **Load and Merge** all CSVs into one DataFrame
2. **Drop Non-informative Features** like IP addresses, ports, etc.
3. **Handle Non-Numeric Data:**

   * Drop string-type columns
   * Replace inf values, then drop NaNs
4. **Feature Selection:**

   * Remove features with correlation to label between -0.1 and 0.1
5. **Export Processed Data** for training

---

## 🎓 Model Training (train.ipynb)

### Splitting

* Split processed dataset into **X, y**, and then into **Train/Test** sets.

### Trained Models

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* MLP Classifier
* Linear SVM

### Saving

* All models are saved using `joblib` in `/models`
* Test set also saved for reuse

---

## 🤮 Evaluation (evaluate.ipynb)

Models were evaluated using **Accuracy**, **Precision**, **Recall**, and **F1 Score**.

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| 🥇 Random Forest    | 0.9985   | 0.9962    | 0.9959 | 0.9961   |
| 🥈 XGBoost          | 0.9984   | 0.9960    | 0.9959 | 0.9960   |
| 🥉 LightGBM         | 0.9977   | 0.9950    | 0.9932 | 0.9941   |
| MLP Classifier      | 0.9776   | 0.9134    | 0.9790 | 0.9451   |
| Logistic Regression | 0.9254   | 0.8213    | 0.7935 | 0.8072   |
| Linear SVM          | 0.9199   | 0.8391    | 0.7337 | 0.7829   |

> ✅ **Random Forest** and **XGBoost** demonstrated the best generalization.

---

## 🔍 Inference (inference.ipynb)

* Load best-performing model from `/models`
* Load or input unseen test data
* Predict and display outcome (benign/malicious)

---

## 📊 Visualizations

* Correlation heatmap
* Feature distribution
* Accuracy comparison bar chart
* Confusion matrix for selected models

---

## 🔀 Future Work

* ⏳ Real-time streaming inference
* 🧠 Multi-class classification (attack categories)
* 🔄 Feature selection via PCA / mutual info
* ⚙️ Hyperparameter tuning with GridSearchCV
* ☁️ Integration with dashboards (Grafana, Kibana, etc.)

---

## 🛠️ Tech Stack

* Python 3.x
* Jupyter Notebooks
* Scikit-learn, XGBoost, LightGBM
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 📜 License

This project is for **educational and research** use only.
Dataset by the **Canadian Institute for Cybersecurity**.

---

## 🤝 Author

**Ahmed Elghazouly**
[GitHub Profile](https://github.com/Ahmed-Elghazouly)

---

## 🌟 Contributions

Pull requests are welcome! Fork this repo and feel free to suggest improvements or submit fixes.
