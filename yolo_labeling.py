import os
import cv2
from ultralytics import YOLO

# YOLO 모델 불러오기 (PyTorch Hub 사용)
model = YOLO(r"C:\Users\windows11\Downloads\CBR LNP.v1i.yolov8\YOLOv8_training\experiment19\weights\best.pt")
# 이미지 폴더 경로 설정
image_folder = r"C:\Users\windows11\study\path_to_places365\train_256_places365standard\biology_laboratory"
output_folder = r"C:\Users\windows11\study\path_to_places365\train_256_places365standard\biology_laboratory_label"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# YOLO 라벨 포맷 저장 함수
def save_yolo_labels(file_path, detections, image_shape):
    height, width = image_shape[:2]
    with open(file_path, 'w') as f:
        for detection in detections:
            cls, x_center, y_center, bbox_width, bbox_height = detection
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# 이미지 파일 탐색 및 처리
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
for file_name in os.listdir(image_folder):
    if not any(file_name.lower().endswith(ext) for ext in image_extensions):
        continue

    # 이미지 경로
    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로드 실패: {file_name}")
        continue

    # YOLO 탐지
    results = model.predict(image, save=False, verbose=False)

    # 탐지된 객체 정보 가져오기
    detections = []
    for result in results:  # 결과 객체에서 탐지 정보 추출
        boxes = result.boxes  # 탐지된 박스
        for box in boxes:
            x_center, y_center, bbox_width, bbox_height = box.xywh[0]  # 중심점 및 크기
            cls = int(box.cls[0])  # 클래스 ID
            # YOLO 라벨 포맷으로 변환 (상대 좌표값)
            x_center /= image.shape[1]
            y_center /= image.shape[0]
            bbox_width /= image.shape[1]
            bbox_height /= image.shape[0]
            detections.append((cls, x_center, y_center, bbox_width, bbox_height))

    # 라벨 저장 경로
    label_file_name = os.path.splitext(file_name)[0] + '.txt'
    label_file_path = os.path.join(output_folder, label_file_name)

    # YOLO 포맷으로 저장
    save_yolo_labels(label_file_path, detections, image.shape)

    print(f"처리 완료: {file_name} -> {label_file_name}")
