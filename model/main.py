# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

def data_processing(path):

    # Mapper for categorical Loan Grades
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

    # Import data from path
    data = pd.read_csv(path)
    
    # Drop "cb_person_default_on_file" column
    data = data.drop(["cb_person_default_on_file"], axis = 1)

    # Replace missing values with mean feature value
    data['person_emp_length'].fillna(data['person_emp_length'].mean())
    data['loan_int_rate'].fillna(data['loan_int_rate'].mean())

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Covert categorical columns to numerical columns
    data['loan_grade'] = data['loan_grade'].map(grade_map)
    data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True, dtype=int)

    return data

def create_model(data):
    
    # Create feature matrix and target vector
    X = data.drop(['loan_status'], axis=1)
    y = data['loan_status']

    # Split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 23)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators = 100, random_state = 23)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f'Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Classification Report: \n {classification_report(y_test, y_pred)}')

    return model, scaler

def main():

    # Load data
    data = data_processing("data\credit_risk_dataset.csv")

    # Create model
    model, scaler = create_model(data)

    # Export model
    with open('model\model.pkl', 'wb') as f: #(filename, write+binary)
        pickle.dump(model, f)

    with open('model\scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    main()
