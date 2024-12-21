import pandas as pd
import matplotlib.pyplot as plt
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
import numpy as np

# Confidence별 데이터 경로 설정
file_paths = {
    0.25: r"./labels_dataframe_tfidf)conf025.csv",
    0.50: r"./labels_dataframe_tfidf)conf050.csv",
    0.75: r"./labels_dataframe_tfidf)conf075.csv"
}

# 결과 저장용 딕셔너리
f1_scores = {model: [] for model in [
    'Logistic Regression', 'Decision Tree', 'Random Forest', 
    'SVM', 'Naive Bayes', 'KNN', 'Artificial Neural Network', 
    'XGBoost', 'LDA'
]}


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

# Confidence 별로 결과 수집
for confidence, path in file_paths.items():
    print(f"Evaluating models for confidence {confidence}...")
    # Load the dataset
    labels_df = pd.read_csv(path)

    # Evaluate classifiers and get results
    results = evaluate_classifiers(labels_df)

    # Extract weighted avg F1-Score for each model
    for model, report in results.items():
        f1_scores[model].append(report['weighted avg']['f1-score'])

# 데이터 프레임 생성
confidence_values = list(file_paths.keys())
table_data = pd.DataFrame(f1_scores, index=confidence_values)
table_data.index.name = 'Confidence'

# CSV 파일로 저장
output_path = "f1_scores_by_confidence.csv"
table_data.to_csv(output_path)

print(f"F1-Score 데이터가 '{output_path}'에 저장되었습니다.")
# 데이터 표 추가
table = plt.table(
    cellText=table_data.values,
    rowLabels=table_data.index,
    colLabels=table_data.columns,
    cellLoc='center',
    loc='bottom',
    bbox=[0.0, -0.3, 1.0, 0.25]
)

# 표 스타일 설정
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(table_data.columns))))

plt.subplots_adjust(left=0.1, bottom=0.4)  # 그래프와 표 간격 조정
plt.show()