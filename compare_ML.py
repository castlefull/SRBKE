import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Load the prepared data
# Assuming `labels_df` is available and has columns ['detected_objects', 'label']
# Uncomment and adjust this line based on how you load your dataframe
labels_df = pd.read_csv(r"./labels_dataframe_tfidf)conf025.csv")

def evaluate_classifiers(labels_df):
    """Evaluate multiple classifiers on the prepared dataset."""
    # Combine all feature columns into a single text representation
    detected_objects = labels_df.drop(columns=['label']).apply(lambda row: ', '.join(row.index[row == 1]), axis=1)

    # Vectorize the detected objects
    vectorizer = TfidfVectorizer(max_features=20, min_df=0.01, max_df=0.85)
    tfidf_features = vectorizer.fit_transform(detected_objects).toarray()
    
    # Encode string labels to numeric values
    label_encoder = LabelEncoder()
    labels_df['encoded_label'] = label_encoder.fit_transform(labels_df['label'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels_df['encoded_label'], test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', probability=True),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LDA': LinearDiscriminantAnalysis()
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Classification Report for {name}:")
        report = classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred), output_dict=True)
        print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)))
        results[name] = report

    return results


# Call the evaluation function
# Adjust this line to use the correct labels_df loaded from your pipeline
results = evaluate_classifiers(labels_df)
