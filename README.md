# TITLE:
SpamSlayer

# BRIEF EXPLANATION ABOUT THE WORKING MODEL:
This model basically works on Naive Bayes Machine learning Algorithm. It is trained by a Spam Ham Dataset. It gets text input from the user and successfully predicts whether it is SPAM or HAM. It checks each word from the input with the pre-trained Dataset and calculates the probability of each word and finally totals the probabilities in order to get the probability of the full sentence. Then, it says whether it is SPAM or HAM.

# ML ALGORITHM USED:
I used Naive Bayes ML Algorithm to build this model.
Naive Bayes finds the probabilites of each word in the sentence whether it is SPAM or HAM relating to the Dataset. Finally it adds the probabilities of all words to find the total probability of the sentence whether it is SPAM or HAM.

# TECHNOLOGY USED FOR BUILDING THE FRONT-END:
I used Streamlit Technology to build the front-end for this model. Streamlit is basically a Built-in python Framework. It allows us to code the front-end in python itself. It also allows us to do modifications according to our interest.

# STEP-BY-STEP EXPLANATION OF THE CODE:
## Backend(Model's code):
### Step - 1:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
```
Importing the necessary libraries.

### Step - 2:
```python
def load_data():
    data = pd.read_csv("Spam_Ham_Dataset.csv")
    print(data.columns) 
    return data
```
Loading the dataset

### Step - 3:
```python
def preprocess_data(data):
    X = data['Message']  # Email content
    y = data['Category']  # Labels: 'spam' or 'ham'
    
    vectorizer = CountVectorizer()
    X_cv = vectorizer.fit_transform(X)  # Convert text to numeric
    
    X_train_cv, X_test_cv, y_train, y_test = train_test_split(X_cv, y, test_size=0.2, random_state=42)
    return X_train_cv, X_test_cv, y_train, y_test, vectorizer
```
Preprocessing the data.
Setting X variable for the Message column of the Dataset and Y variable for the Category column of the dataset.
Using Count Vectorizer to convert the text data into computer readable binary data.
Splitting the model for Training and Testing using X_train, Y_train, X_test, Y_test as variables and train_test_split as the function

### Step - 4:
```python
def train_model(X_train_cv, y_train):
    model = MultinomialNB()
    model.fit(X_train_cv, y_train)
    return model
```
Training the model

### Step - 5:
```python
def evaluate_model(model, X_test_cv, y_test):
    y_pred = model.predict(X_test_cv)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)  # Save as a dictionary
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
```
Evaluating the model by Printing the Accuracy score, Confusion matrix and Classification report.

### Step - 6:
```python
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON compatibility
        "classification_report": class_report
    }
    with open("metrics.pkl", "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)
    print("Evaluation metrics saved successfully.")
```
Saving the metrics for displaying in the front-end.

### Step - 7:
```python
def save_model_and_vectorizer(model, vectorizer):
    with open("spam_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print("Model and vectorizer saved successfully.")
```
Saving the trained model as pickle files so that you don't have to run the whole code and train it again and again. We can use the pickle files itself

### Step - 8:
```python
def main():
    # Step 1: Load the data
    data = load_data()
    
    # Step 2: Preprocess the data
    X_train_cv, X_test_cv, y_train, y_test, vectorizer = preprocess_data(data)
    
    # Step 3: Train the model
    model = train_model(X_train_cv, y_train)
    
    # Step 4: Evaluate the model
    evaluate_model(model, X_test_cv, y_test)
    
    # Step 5: Save the model and vectorizer
    save_model_and_vectorizer(model, vectorizer)

# Run the main function
if __name__ == "__main__":
    main()
```
Defining the main function to call all the functions declared before. And then finally running the main function

## Front-end:
### Step - 1:
```python
import streamlit as st
import pickle
import pandas as pd
import numpy as np
```
Importing the necessary libraries

### Step - 2:
```python
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
metrics = pickle.load(open("metrics.pkl", "rb"))
```
Loading the pickle files for model, vectorizer and saved metrics.

### Step - 3:
```python
st.title("HP's - Mail Spam Filter")
st.write("This app classifies emails as **Spam** or **Ham** using a Naive Bayes classifier.")
```
Giving the title and description of the model

### Step - 4:
```python
# Display evaluation metrics
st.subheader("Model Evaluation Metrics")
st.write(f"**Accuracy**: {metrics['accuracy']:.2f}")

# Confusion Matrix
st.write("**Confusion Matrix**:")
conf_matrix = np.array(metrics['confusion_matrix'])
st.dataframe(pd.DataFrame(conf_matrix, columns = ["Predicted Ham", "Predicted Spam"], index = ["Actual Ham", "Actual Spam"]))

# Classification Report
st.write("**Classification Report**:")
class_report = pd.DataFrame(metrics['classification_report']).transpose()
st.dataframe(class_report)
```
Displaying the evaluation metrics, Accuracy, Confusion matrix and Classification Report respectively.

### Step - 5:
```python
email_content = st.text_area("Enter the Email content:")
```
Creating an input text box

### Step - 6:
```python
if st.button("Check Email"):
    if email_content.strip():  # Ensure the input isn't empty
        email_vector = vectorizer.transform([email_content])
        prediction = model.predict(email_vector)[0]
        st.success(f"The email is classified as: **{prediction.upper()}**")
    else:
        st.error("Please enter some content to classify.")
```
Writing code for the "Check Email" button.
If the input is empty, we will display a warning message.
If input is present, then we will display the output generated by the model's code.


# OUTPUT:

![Screenshot 2025-02-12 120348](https://github.com/user-attachments/assets/79c9cd35-fafd-4acd-b356-8d2da0fa5b2e)
![Screenshot 2025-02-12 120332](https://github.com/user-attachments/assets/7d64c7f3-4240-4966-b6cc-4e88a81db8c7)
![Screenshot 2025-02-12 120618](https://github.com/user-attachments/assets/a9a63b09-9670-4df4-98cc-10ec08444977)

# RESULT:
Thus, the given input text is predicted and displayed as SPAM or HAM successfully!
