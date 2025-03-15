# Spam-Mail-Detection
This project builds a Spam Mail Detection system using NLP and Machine Learning. It classifies emails as spam or ham (not spam) using the Multinomial Naïve Bayes (MNB) algorithm. The model is trained with TF-IDF (Term Frequency-Inverse Document Frequency) to enhance feature extraction and improve classification accuracy..



## **Key Features**

1.) Data Preprocessing: Managing missing data and encoding categorical labels.

2.) Feature Extraction: Transforming text into numerical vectors using TfidfVectorizer.

3.) Model Training: Implementing Multinomial Naïve Bayes (MNB) for email classification.

4.) Performance Analysis: Evaluating accuracy, generating classification reports, and using a confusion matrix.

5.) Hyperparameter Optimization: Enhancing model performance with GridSearchCV.

6.) Email Classification: Predicting whether user-input emails are spam or legitimate.

## **Technologies Used**  

- **Python** – Core programming language for implementation.  
- **Pandas & NumPy** – Efficient data handling and manipulation.  
- **Scikit-learn** – Machine learning models, performance metrics, and feature extraction.  
- **Matplotlib & Seaborn** – Data visualization for insights and analysis.

## **Dataset**
This project utilizes a labeled email dataset categorized into two classes:

- Ham (1): Genuine, non-spam emails
- Spam (0): Unwanted or junk emails

### **Installation & Setup**
1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/spam-mail-detection.git  
cd spam-mail-detection
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Upload Dataset

Place mail_data.csv in the working directory.

4️⃣ Run the Script
```
python spam_detection.py
```

### **Code Overview**
1️⃣ Data Preprocessing
Load the dataset using Pandas.
Handle missing values.
Convert labels (spam, ham) into binary format (1 = Ham, 0 = Spam).

2️⃣ Feature Extraction
Transform email text into numerical form using TfidfVectorizer.

3️⃣ Model Training
Split data into training (80%) and testing (20%) sets.
Train a Multinomial Naïve Bayes (MNB) model on extracted features.

4️⃣ Performance Evaluation
Measure accuracy on training and test sets.
Generate a classification report and confusion matrix.
Optimize model performance using GridSearchCV.

5️⃣ Spam Mail Prediction
The model takes an email input and classifies it as spam or ham.

🔹 Example:

input_email = ["Congratulations! You have won a lottery. Call now to claim!"]  
input_features = feature_extraction.transform(input_email)  
prediction = model.predict(input_features)  
print("Spam mail" if prediction[0] == 0 else "Ham mail")  

### **Results & Visualization**

✅ Training Accuracy: ~98%

✅ Testing Accuracy: ~96%

✅ Performance Metrics: Confusion Matrix & Classification Report

✅ Data Visualization:

Confusion Matrix Heatmap
Feature Importance Analysis using TF-IDF
This project effectively detects spam emails, enhancing email security and filtering accuracy. 🚀







