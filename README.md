# 🌸 Iris Flower Classification  

An **end-to-end machine learning project** that predicts the species of an Iris flower based on its sepal and petal measurements. The project covers **data exploration, model development, performance evaluation**, and an interactive **Streamlit web app** for real-time predictions.

---

## 📘 Project Overview  

This project demonstrates the complete data science workflow using the famous **Iris dataset**. The goal is to classify flowers into three species — *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica* — using supervised learning algorithms.  

It includes:
- Exploratory Data Analysis (EDA) with statistical summaries and correlation insights  
- Visualization of feature relationships and PCA clustering  
- Model training with multiple algorithms and cross-validation  
- Deployment-ready **Streamlit app** with a custom-designed interface  

---

## 🧠 Key Insights  

- **Petal length and width** are the strongest predictors (correlation > 0.9).  
- **Sepal dimensions** show weak correlation with species.  
- **SVM** achieved the highest accuracy of **96.7%**.  
- PCA visualization shows clear separation among species clusters.  

---

## ⚙️ Tech Stack  

- **Python**  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, plotly  
- **Model:** Support Vector Machine (SVM)  
- **Deployment:** Streamlit  
- **Visualization:** Plotly, Matplotlib  

---

## 🚀 How to Run  

1. Clone the repository  
   ```bash
   git clone https://github.com/fidasaleem/Codsoft_IrisClassification.git
   cd Codsoft_IrisClassification

  2. Install dependencies
     ```bash
     pip install -r requirements.txt

  3. Run the Streamlit app
     ```bash
     streamlit run Iris.py

## 🎨 App Features

- Custom background and themed UI
- Adjustable sliders for feature inputs
- Real-time probability plot and prediction
- Highlighted results with dynamic color

## 🏁 Conclusion

This project demonstrates how machine learning can be effectively applied to classic datasets for classification and visualization, while showcasing deployment-ready interactive apps for end-users.
