# Machine Learning Intrusion Detection System (IDS)

## Project Overview
This project documents the end-to-end process of building a robust Network Intrusion Detection System using machine learning. The primary goal is to accurately classify network traffic as either **benign** or **malicious**.

More than just a classification task, this project tells a story about a critical real-world challenge: **data drift**. It follows a two-phase approach to first uncover this problem and then build a resilient model to solve it, demonstrating a practical understanding of deploying ML systems.

---

## The Story: A Tale of Two Models

To simulate a realistic deployment lifecycle, the project was conducted in two phases.

### Phase 1: The Initial Model & Uncovering Data Drift
An initial set of models was trained on network data from Monday to Thursday and evaluated on a test set also from those days. The results were excellent, with the top model achieving over **98% accuracy**.

However, when this model was used for inference on new, unseen data from Friday, its performance crashed:
* **Accuracy dropped to ~70%**
* **F1-Score for attacks fell to a mere 44%**
* **Recall for attacks was only 28%**, meaning it missed almost 3 out of every 4 real attacks.

This is a classic demonstration of **data drift**, where the patterns in the "future" data were fundamentally different from the historical data the model was trained on. This phase successfully highlighted that a high test score doesn't always guarantee real-world performance.

### Phase 2: Building a Robust, Generalized Model
To address the data drift, a new, more resilient model was built.

* **Strategy:** All data from all five days (Monday - Friday) was merged and shuffled to create a comprehensive dataset. A standard 80/20 train-test split was performed on this complete dataset.
* **Result:** By training on a much wider variety of traffic patterns, the new models learned to generalize far more effectively.

---
## Final Model Performance

The best-performing model from the robust training phase was **Random Forest**, achieving the following outstanding results on the final, unseen test set:

| Model | Accuracy | Precision (Attack) | Recall (Attack) | F1-Score (Attack) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **98.7%** | **95.4%** | **98.1%** | **96.7%** |
| XGBoost | 98.6% | 95.4% | 97.8% | 96.6% |
| LightGBM | 98.5% | 95.1% | 97.6% | 96.3% |

---

## Project Structure

* `/data/`: Contains the raw CIC-IDS-2017 dataset and the final processed datasets.
* `/models/`: Stores the final, trained model objects (`.pkl` files).
* `/src/`: Contains the Jupyter Notebooks for each stage:
    * `preprocess.ipynb`: Handles all data loading, cleaning, feature selection, and splitting.
    * `train.ipynb`: Trains multiple classification models using `Pipelines` to bundle preprocessing where needed.
    * `evaluate.ipynb`: Evaluates all trained models on the test set and generates performance metrics and visualizations.
    * `inference.ipynb`: A simple script demonstrating how to use the final, best model to make a prediction on a new piece of data.

---

## How to Use

1.  Clone this repository.
2.  Install the required packages: `pip install -r requirements.txt`
3.  Run the Jupyter Notebooks in the `/src/` folder in the following order: `preprocess`, `train`, `evaluate`, and finally `inference`.