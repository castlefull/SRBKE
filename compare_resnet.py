import os
import pandas as pd
import time
import torch
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from torchvision import models, transforms
from PIL import Image


# ResNet 모델 로드
from torchvision.models import ResNet50_Weights
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.eval()

# YOLO 관련 함수
def detect_objects(image_path, model):
    """YOLO 모델로 객체 탐지"""
    results = model(image_path, conf=0.5)
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

    return yolo_model, labels_df

def train_random_forest_classifier(labels_df):
    """TF-IDF와 Random Forest로 장면 분류 모델 학습"""
    vectorizer = TfidfVectorizer(max_features=20, min_df=0.01, max_df=0.85, stop_words=['no_objects_detected'])
    tfidf_features = vectorizer.fit_transform(labels_df['detected_objects']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels_df['label'], test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("Random Forest 분류 성능 보고서:")
    print(classification_report(y_test, y_pred))

    return vectorizer, classifier

# ResNet 관련 함수
def preprocess_image(image_path):
    """ResNet 입력 전처리"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

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

def train_resnet_labels(image_dir):
    """ResNet 학습 데이터를 파일명 기반으로 준비"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    labels = [extract_label_from_filename(file) for file in image_files]
    return image_files, labels

def evaluate_resnet(image_files, labels, image_dir):
    """ResNet 성능 평가"""
    y_true = []
    y_pred = []
    for file, true_label in zip(image_files, labels):
        image_path = os.path.join(image_dir, file)
        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            outputs = resnet_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = extract_label_from_filename(file)  # 예측값을 파일 이름 기반으로 변환
            y_true.append(true_label)
            y_pred.append(predicted_label)

    # 성능 평가
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred)
    print("ResNet 성능 보고서:")
    print(report)
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

if __name__ == "__main__":
    # YOLO 파이프라인 실행
    image_dir = r"./dataset_places365/data"
    yolo_model, labels_df = setup_yolo_pipeline(image_dir)
    vectorizer, classifier = train_random_forest_classifier(labels_df)

    # ResNet 학습 및 평가
    resnet_train_files, resnet_train_labels = train_resnet_labels(image_dir)
    evaluate_resnet(resnet_train_files, resnet_train_labels, image_dir)
