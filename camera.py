import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO(r"C:\Users\windows11\Downloads\CBR LNP.v1i.yolov8\YOLOv8_training\experiment19\weights\best.pt")

# 클래스 매핑 정의
class_map = [
    'Chemical_respirator', 'PCR', 'autoclave', 'biohazard_symbol', 'cell_culture', 
    'centrifuge', 'chemical_hazard', 'Geiger_counter', 'Lead_Apron_Gloves', 
    'chromatography_equipment', 'emergency_shower', 'glass_apparatus', 'lab_animal', 
    'laboratory_chemicals', 'manipulation_arm', 'microscope', 
    'radiation_hazard_symbol', 'radioactive_hot_cell'
]

# 점수 테이블 정의
score_table = {
    'Chemical_respirator': (3, 1, 2), 'PCR': (0, 3, 0), 'autoclave': (0, 3, 0),
    'biohazard_symbol': (0, 5, 0), 'cell_culture': (0, 4, 0), 'centrifuge': (2, 3, 0),
    'chemical_hazard': (5, 0, 0), 'Geiger_counter': (1, 0, 5), 'Lead_Apron_Gloves': (2, 0, 3),
    'chromatography_equipment': (4, 2, 0), 'emergency_shower': (2, 1, 1),
    'glass_apparatus': (3, 2, 0), 'lab_animal': (0, 4, 0), 'laboratory_chemicals': (4, 3, 3),
    'manipulation_arm': (0, 0, 2), 'microscope': (0, 3, 0),
    'radiation_hazard_symbol': (0, 0, 5), 'radioactive_hot_cell': (0, 0, 5)
}

def classify_lab(detected_objects):
    chemical_score = biological_score = radiological_score = 0
    for obj, detected in detected_objects.items():
        if detected:
            scores = score_table.get(obj, (0, 0, 0))
            chemical_score += scores[0]
            biological_score += scores[1]
            radiological_score += scores[2]

    scores = {
        "chemical": chemical_score,
        "biological": biological_score,
        "radiological": radiological_score
    }
    return max(scores, key=scores.get)

# 웹캠 열기
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    # YOLOv8 탐지
    results = model(frame)

    # 탐지된 객체 초기화
    detected_objects = {obj: False for obj in class_map}

    # YOLOv8 결과 처리
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Flatten the nested list and convert to int
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                cls = int(box.cls.item())
                class_name = class_map[cls]
                detected_objects[class_name] = True

                # 프레임에 경계 상자와 클래스 이름 추가
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 탐지된 객체로 실험실 유형 분류
    lab_type = classify_lab(detected_objects)
    cv2.putText(frame, f"Lab Type: {lab_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 결과 출력
    cv2.imshow("YOLO Detection", frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
