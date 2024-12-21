import os
import pandas as pd
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_label_from_filename(filename):
    """파일명에서 실험실 유형 라벨을 추출"""
    if "chem" in filename.lower():
        return "chem"
    elif "bio" in filename.lower():
        return "bio"
    elif "rad" in filename.lower():
        return "rad"
    else:
        return "unknown"

def detect_objects(image_path, model):
    """YOLO 모델로 객체 탐지"""
    results = model(image_path, conf=0.75)
    detected_objects = []
    if results and hasattr(results[0], "boxes") and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls)
            detected_objects.append(model.names[class_id])
    else:
        detected_objects.append("no_objects_detected")
    return detected_objects

def setup_yolo_pipeline(image_dir):
    """YOLO 파이프라인 설정"""
    # YOLO 모델 로드
    yolo_model = YOLO(r"./YOLOv8_training/experiment19/weights/best.pt")
    
    # 데이터 준비
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    data = [{"image_name": file, "label": extract_label_from_filename(file)} for file in image_files]
    labels_df = pd.DataFrame(data)

    # 객체 탐지 및 결과 저장
    detected_objects_list = []
    for image_name in labels_df['image_name']:
        image_path = os.path.join(image_dir, image_name)
        detected_objects = detect_objects(image_path, yolo_model)
        detected_objects_list.append(", ".join(detected_objects))
    
    labels_df['detected_objects'] = detected_objects_list
    labels_df = labels_df[~labels_df['detected_objects'].str.contains("no_objects_detected", na=False)]
    labels_df.to_csv("labels_dataframe.csv", index=False)
    return yolo_model, labels_df

def train_random_forest_classifier(labels_df):
    """TF-IDF와 Random Forest로 장면 분류 모델 학습"""
    vectorizer = TfidfVectorizer(max_features=20, min_df=0.01, max_df=0.85, stop_words=['no_objects_detected'])
    
    # TF-IDF 벡터화 수행
    tfidf_features = vectorizer.fit_transform(labels_df['detected_objects']).toarray()
    
    # NumPy 배열을 DataFrame으로 변환
    tfidf_df = pd.DataFrame(tfidf_features, columns=vectorizer.get_feature_names_out())
    tfidf_df['label'] = labels_df['label'].values  # 라벨 추가
    
    # DataFrame 저장
    tfidf_df.to_csv("labels_dataframe_tfidf)conf075.csv", index=False)
    
    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels_df['label'], test_size=0.2, random_state=43)

    # Random Forest 모델 학습
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # 예측 및 결과 출력
    y_pred = classifier.predict(X_test)
    print("Random Forest 분류 성능 보고서:")
    print(classification_report(y_test, y_pred))

    return vectorizer, classifier


if __name__ == "__main__":
    image_dir = r"./dataset_places365/data"
    yolo_model, labels_df = setup_yolo_pipeline(image_dir)
    vectorizer, classifier = train_random_forest_classifier(labels_df)
    # Random Forest 및 TF-IDF 저장
    joblib.dump(classifier, "random_forest_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("모델과 벡터라이저가 저장되었습니다.")