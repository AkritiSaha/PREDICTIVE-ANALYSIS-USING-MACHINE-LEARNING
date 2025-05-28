COMPANY : CODTECH IT SOLUTIONS

NAME: AKRITI SAHA

INTERN ID: :CT04DM725

DOMAIN: DATA ANALYTICS

DURATION: 4 WEEEKS

MENTOR: NEELA SANTHOS KUMAR

# 🌸 Iris Flower Species Prediction - ML Project

This project is part of my internship at **CodTech** and focuses on **Predictive Analysis using Machine Learning**. The goal is to classify iris flowers into three species based on their petal and sepal measurements.

 
Dataset Used
The classic [Iris dataset] contains 150 samples from three species of Iris flowers:
- **Setosa**
- **Versicolor**
- **Virginica**

Features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)


 Problem Statement

 Build a classification model to predict the species of Iris flowers using machine learning techniques.


 ML Workflow

1. **Data Loading & Preprocessing**
   - Used `seaborn`'s built-in iris dataset
   - Converted categorical labels to numeric using `LabelEncoder`

2. **Exploratory Data Analysis**
   - Plotted pair plots, box plots, and heatmaps
   - Checked for feature correlation

3. **Feature Selection**
   - All four features were retained due to their high predictive importance

4. **Model Building**
   - Model used: **Random Forest Classifier**
   - Split data into train-test sets (80-20)
   - Features were scaled using `StandardScaler`

5. **Model Evaluation**
   - Metrics: Accuracy, Classification Report, Confusion Matrix
   - Visualized feature importance


 Technologies Used

- Python 
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn  



 Results

- **Accuracy:** ~100% on test data
- **Most influential features:** Petal Length & Petal Width
- The model performs excellent classification across all three species.



Conclusion

This project showcases how to use basic machine learning tools to analyze data and build a reliable prediction model from scratch. It's a great beginner-friendly application of supervised classification using the Iris dataset.




