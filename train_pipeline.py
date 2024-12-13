import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
import time


def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

    train_precision = metrics.precision_score(y_train, model.predict(X_train), average='weighted', zero_division=1)
    test_precision = metrics.precision_score(y_test, model.predict(X_test), average='weighted', zero_division=1)

    train_recall = metrics.recall_score(y_train, model.predict(X_train), average='weighted', zero_division=1)
    test_recall = metrics.recall_score(y_test, model.predict(X_test), average='weighted', zero_division=1)

    end_time = time.time()
    print(f"Evaluation took {end_time - start_time:.2f} seconds")

    print("Training Set Metrics:")
    print("Training Accuracy {}: {:.2f}%".format(name, train_accuracy * 100))
    print("Training Precision {}: {:.2f}%".format(name, train_precision * 100))
    print("Training Recall {}: {:.2f}%".format(name, train_recall * 100))

    print("\nTest Set Metrics:")
    print("Test Accuracy {}: {:.2f}%".format(name, test_accuracy * 100))
    print("Test Precision {}: {:.2f}%".format(name, test_precision * 100))
    print("Test Recall {}: {:.2f}%".format(name, test_recall * 100))


# Function to train the model and save it
def train_and_save_model(num_rows=None):
    start_time = time.time()
    print("Loading the dataset...")
    df = pd.read_csv("/Users/shekhargoud/Desktop/creditcard_prediction/dataset/creditcard_2023.csv")

    # Drop unwanted columns
    df.drop(["id"], axis=1, inplace=True)

    # If num_rows is specified, select a subset of the data
    if num_rows:
        df = df.head(n=num_rows)
        print(f"Using a subset of {num_rows} rows for training.")
    else:
        print("Using the entire dataset for training.")
    df = df.sample(frac=1).reset_index(drop=True)

    # Separate features and target
    X = df.drop(["Class"], axis=1)
    y = df["Class"]
    print("Applying label encoding on target column")
    # Apply Label Encoding to the target variable y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    pipeline = Pipeline([
        ('preprocessor', StandardScaler()),  # Apply StandardScaler to all features in X
        ('classifier', XGBClassifier(
            colsample_bytree=0.9,
            learning_rate=0.2,
            max_depth=5,
            n_estimators=200,
            subsample=0.9
        ))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Fit the model using the pipeline
    pipeline.fit(X_train, y_train)

    # Save the model
    with open('model_pipeline.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    end_time = time.time()
    print(f"Model training and saving took {end_time - start_time:.2f} seconds")
    print("Model evaluation...\n")
    # Evaluate the model
    evaluate_classification(pipeline, "XGBoost", X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    num_rows = None  # Set the number of rows for training (e.g., num_rows = 1000000)
    train_and_save_model(num_rows)
curl -X POST http://127.0.0.1:5001/train

