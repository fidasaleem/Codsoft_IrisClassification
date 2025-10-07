# 🌸 Iris Flower Classification  

An **end-to-end machine learning project** that predicts the species of an Iris flower using its sepal and petal measurements.  
This project covers the full data science workflow — from **data exploration** to **model deployment** via a sleek **Streamlit web app**.

---

## 📘 Project Overview  

The **Iris dataset**, one of the most popular datasets in machine learning, is used to classify flowers into three species:  
🌼 *Iris-setosa*  
🌷 *Iris-versicolor*  
🌺 *Iris-virginica*  

This project includes:  
- 🔍 **Exploratory Data Analysis (EDA):** Statistical summaries, feature correlations, and visual insights  
- 📊 **Data Visualization:** Pairplots, boxplots, and PCA-based species clustering  
- ⚙️ **Model Building:** Training and evaluating multiple supervised learning algorithms  
- 🌐 **App Deployment:** A fully interactive Streamlit app for real-time predictions  

---

## 🧠 Key Insights  

- **Petal length and petal width** are the most powerful predictors (correlation > 0.9).  
- **Sepal dimensions** contribute less to classification accuracy.  
- **Support Vector Machine (SVM)** achieved the best accuracy of **96.7%**.  
- PCA visualization clearly separates the three Iris species into distinct clusters.  

---

## ⚙️ Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Libraries** | scikit-learn, pandas, numpy, matplotlib, seaborn, plotly |
| **Model** | Support Vector Machine (SVM) |
| **Deployment** | Streamlit |
| **Visualization** | Plotly, Matplotlib, Seaborn |

---

## 🚀 Try the App  

You can explore the model interactively using the live Streamlit app below:  
👉 [**Launch Iris Classifier App**](https://codsoftirisclassification-jxcibfqpfmizwuunj3sfyj.streamlit.app/)  

---

## 💻 Run Locally  

Follow these steps to run the project on your system:


# 1️⃣ Clone the repository
```bash
git clone https://github.com/fidasaleem/Codsoft_IrisClassification.git
cd Codsoft_IrisClassification
```

# 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
# 3️⃣ Run the Streamlit app
```bash
streamlit run Iris.py
```
