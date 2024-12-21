import os
import cv2
import joblib
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


def detect_objects(image, model):
    """YOLO 모델로 객체 탐지"""
    results = model(image)
    detected_objects = []

    if results and hasattr(results[0], "boxes") and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls)
            detected_objects.append(model.names[class_id])
    else:
        detected_objects.append("no_objects_detected")

    return detected_objects


def classify_lab(detected_objects, vectorizer, classifier, confidence_threshold=0.6):
    """탐지된 객체로 실험실 유형 분류"""
    detected_objects_str = ", ".join(detected_objects)

    tfidf_vector = vectorizer.transform([detected_objects_str]).toarray()
    lab_probabilities = classifier.predict_proba(tfidf_vector)[0]
    max_probability = max(lab_probabilities)

    if max_probability >= confidence_threshold:
        lab_type = classifier.classes_[lab_probabilities.argmax()]
    else:
        lab_type = "unknown"

    return lab_type


def process_video(video_path, model, vectorizer, classifier, output_path=None):
    """영상 처리: 객체 탐지 및 실험실 유형 예측"""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    if output_path:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    lab_predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 객체 탐지
        detected_objects = detect_objects(frame, model)

        # 실험실 유형 예측
        lab_type = classify_lab(detected_objects, vectorizer, classifier)
        lab_predictions.append(lab_type)

        # 결과 표시
        text = f"Lab Type: {lab_type}"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 저장
        if output_path:
            out.write(frame)

        # 화면에 표시
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # "unknown" 제외한 최종 결과 계산
    filtered_predictions = [lab for lab in lab_predictions if lab != "unknown"]
    if filtered_predictions:
        most_common_lab = Counter(filtered_predictions).most_common(1)[0]  # 가장 자주 예측된 실험실 유형
        print(f"최종 실험실 유형: {most_common_lab[0]}")
        print(f"발생 빈도: {most_common_lab[1]}회")
    else:
        most_common_lab = ("unknown", 0)
        print("모든 예측이 'unknown'입니다.")

    return most_common_lab[0], filtered_predictions.count(most_common_lab[0])


def main():
    # 경로 설정
    video_path = r"./input_video.mp4"
    output_path = r"./output_video.avi"
    yolo_model_path = r"./YOLOv8_training/experiment19/weights/best.pt"

    # YOLO 모델 로드
    yolo_model = YOLO(yolo_model_path)

    # 저장된 모델 및 벡터라이저 로드
    classifier = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # 영상 처리 및 최종 결과
    most_common_lab, occurrence = process_video(video_path, yolo_model, vectorizer, classifier, output_path)

    if most_common_lab != "unknown":
        print(f"영상 처리 완료. 최종 실험실 유형: {most_common_lab} ({occurrence}회 발생)")
    else:
        print("유효한 실험실 유형이 예측되지 않았습니다.")


if __name__ == "__main__":
    main()
