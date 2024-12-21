import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# YOLOv8 모델 로드
model = YOLO(r"C:\Users\windows11\Downloads\CBR LNP.v1i.yolov8\best.pt")

# 클래스 매핑 정의
class_map = ['Chemical_respirator', 'PCR', 'autoclave', 'biohazard_symbol', 'cell_culture', 'centrifuge', 
             'chemical_hazard','Geiger_counter','Lead_Apron_Gloves','cartridge','chromatography_equipment',
             'emergency_shower','human','glass_apparatus','lab_animal','laboratory_chemicals','manipulation_arm',
             'microscope','radiation_hazard_symbol','radioactive_hot_cell']

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
    for obj, count in detected_objects.items():
        scores = score_table.get(obj, (0, 0, 0))
        chemical_score += scores[0] * count
        biological_score += scores[1] * count
        radiological_score += scores[2] * count

    scores = {
        "Chemical": chemical_score,
        "Biological": biological_score,
        "Radiological": radiological_score
    }
    return max(scores, key=scores.get)

# 동영상 파일 열기
video_path = r"C:\Users\windows11\Downloads\Bromobenzene _ Organic Synthesis - Trim.mp4" # 여기에 동영상 파일 경로를 입력하세요
cap = cv2.VideoCapture(video_path)

all_detected_objects = Counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 탐지
    results = model(frame)

    # YOLOv8 결과 처리
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls.item())
                if cls < len(class_map):
                    class_name = class_map[cls]
                    all_detected_objects[class_name] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("YOLO Detection", frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 최종 결과 시각화
final_lab_type = classify_lab(all_detected_objects)

# 결과 UI 생성
result_ui = np.zeros((600, 800, 3), dtype=np.uint8)
cv2.putText(result_ui, "Detection Results", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

y_offset = 80
for obj, count in all_detected_objects.most_common():
    cv2.putText(result_ui, f"{obj}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 30

cv2.putText(result_ui, f"Lab Type: {final_lab_type}", (20, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Final Results", result_ui)
cv2.waitKey(0)
cv2.destroyAllWindows()
