import os
import multiprocessing
from ultralytics import YOLO

# CUDA 메모리 할당 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def main():
   
    model = YOLO('yolo11n.pt')

    # data.yaml 파일 경로

    # 모델 학습 설정
    model.train(
        data=r"C:\Users\windows11\Downloads\newfolder\yolo.v1i.yolov11\data.yaml",        # 데이터셋 구성 파일 경로
        epochs=300,               # 학습 에폭 수
        imgsz=640,               # 입력 이미지 크기
        batch=-1,                # 배치 크기
        name="yolo11_custom",    # 결과 저장 폴더 이름
        device=0                 # GPU 사용 (0번 GPU), CPU를 사용하려면 'cpu'로 설정
    )

if __name__ == "__main__":
    # Windows에서 안전한 프로세스 생성을 위해 'spawn' 방식 사용
    multiprocessing.set_start_method("spawn", force=True)
    main()
