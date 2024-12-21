import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


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


def detect_objects(image_path, model, output_dir=None):
    """YOLO 모델로 객체 탐지 및 결과 저장"""
    results = model(image_path, conf=0.5)
    detected_objects = []

    if results and hasattr(results[0], "boxes") and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls)
            detected_objects.append(model.names[class_id])

        if output_dir:
            result_image_path = os.path.join(output_dir, os.path.basename(image_path))
            annotated_image = results[0].plot()
            cv2.imwrite(result_image_path, annotated_image)
    else:
        detected_objects.append("no_objects_detected")

        if output_dir:
            shutil.copy(image_path, os.path.join(output_dir, os.path.basename(image_path)))

    return detected_objects


def classify_lab(image_path, model, vectorizer, classifier, confidence_threshold=0.6):
    """이미지 경로를 받아 실험실 유형을 예측"""
    detected_objects = detect_objects(image_path, model)
    detected_objects_str = ", ".join(detected_objects)

    tfidf_vector = vectorizer.transform([detected_objects_str]).toarray()
    lab_probabilities = classifier.predict_proba(tfidf_vector)[0]
    max_probability = max(lab_probabilities)

    if max_probability >= confidence_threshold:
        lab_type = classifier.classes_[lab_probabilities.argmax()]
    else:
        lab_type = "unknown"

    return lab_type, detected_objects


def main():
    # 경로 설정
    image_dir = r"./dataset_places365/data"
    output_dir = r"./dataset_places365/result"
    os.makedirs(output_dir, exist_ok=True)

    # YOLO 모델 로드
    yolo_model = YOLO(r"./YOLOv8_training/experiment19/weights/best.pt")

    # 데이터프레임 생성
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    data = [{"image_name": file, "label": extract_label_from_filename(file)} for file in image_files]
    labels_df = pd.DataFrame(data)

    # 객체 탐지 및 결과 저장
    detected_objects_list = []
    for image_name in labels_df['image_name']:
        image_path = os.path.join(image_dir, image_name)
        detected_objects = detect_objects(image_path, yolo_model, output_dir)
        detected_objects_list.append(", ".join(detected_objects))
    
    labels_df['detected_objects'] = detected_objects_list
    labels_df = labels_df[~labels_df['detected_objects'].str.contains("no_objects_detected", na=False)]

    # TF-IDF 벡터화
    custom_stop_words = ['no_objects_detected']
    vectorizer = TfidfVectorizer(max_features=20, min_df=0.01, max_df=0.85, stop_words=custom_stop_words)
    labels_df = labels_df[labels_df['detected_objects'] != 'no_objects_detected']
    tfidf_features = vectorizer.fit_transform(labels_df['detected_objects']).toarray()

    # 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels_df['label'], test_size=0.2, random_state=43)

    # Random Forest 분류기 학습
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # 평가
    y_pred = classifier.predict(X_test)
    print("분류 성능 보고서:")
    print(classification_report(y_test, y_pred))

    # 새 이미지로 실험
    #new_image_path = r"C:\Users\windows11\study\path_to_places365\chem\chemistry_lab\00002977.jpg"
    #predicted_lab, detected_objects = classify_lab(new_image_path, yolo_model, vectorizer, classifier)
    #print(f"탐지된 객체: {', '.join(detected_objects)}")
    #print(f"예측된 실험실 유형: {predicted_lab}")


if __name__ == "__main__":
    image_dir = r"C:\Users\windows11\study\path_to_places365\data"
    yolo_model, labels_df = setup_yolo_pipeline(image_dir)
    vectorizer, classifier = train_random_forest_classifier(labels_df)
    labels_df.to_csv("labels_dataframe.csv", index=False)