# 📊 Regression Model Comparison Dashboard

An interactive **Streamlit web application** for exploring, training, and comparing multiple regression models — designed for the **AI4Startups Datathon (Climate Smart Agriculture Track)**.

This project demonstrates regression modeling with real-world datasets and lets users upload their own CSVs, tune hyperparameters, and visualize model performance in real time.

---

## 🚀 Features

- 🧠 **Multiple Regression Models**

  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor

- ⚙️ **Hyperparameter Tuning**

  - Adjustable test size and model parameters (e.g., alpha, tree depth, estimators)

- 📈 **Performance Metrics**

  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

- 🔬 **Model Comparison Mode**

  - Compare all models at once and highlight the best performer

- 📉 **Visualizations**
  - Actual vs Predicted scatter plots
  - Feature importance charts (for Random Forest)
  - Model comparison bar charts

---

## 🏗️ Tech Stack

| Category        | Tools / Libraries           |
| --------------- | --------------------------- |
| Language        | Python 3.13                 |
| Framework       | Streamlit                   |
| ML Libraries    | Scikit-learn, Pandas, NumPy |
| Visualization   | Matplotlib                  |
| Version Control | Git + GitHub                |

---

## 🧩 Folder Structure

```
project_root/
│
├── app.py                # Streamlit entry point
├── regression.py         # Core regression logic and visualizations
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/makk-007/datathon-regression-analysis.git
cd <datathon-regression-analysis>
```

### 2️⃣ Create and Activate a Virtual Environment (Python 3.13)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at the link provided (usually `http://localhost:8501`).

---

## 📂 Uploading Your Dataset

1. Prepare your dataset in **CSV format**.
2. Make sure your target column (the value you want to predict) and feature columns are properly labeled.
3. Upload it using the **file uploader** in the app.
4. Choose your target and feature columns to begin training.

---

## 🧠 Example Use Case (Datathon Context)

In the **Climate Smart Agriculture** context, this app can:

- Predict crop yields based on soil, rainfall, and temperature data.
- Compare models to identify which regression algorithm best fits agricultural data.
- Experiment with hyperparameter tuning to minimize prediction error.

---

## 💡 Future Enhancements

- 🔄 Add cross-validation support
- 📊 Integrate SHAP explainability
- ☁️ Deploy on Streamlit Cloud or Hugging Face Spaces
- 🧱 Include classification mode for categorical predictions

---

## 👨🏽‍💻 Author

**Mawusenam Kwofie Kelvin Ackuaku**  
Computer Engineering Graduate | Cybersecurity & Artificial Intelligence Enthusiast  
📍 Ghana

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use and adapt it for educational and non-commercial purposes.

---

### 🌱 “From data to insight — empowering sustainable innovation.”
