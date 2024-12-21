import os
import multiprocessing
from ultralytics import YOLO

# CUDA 메모리 할당 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def main():
    # YOLOv8m 모델 로드 (사전 훈련된 가중치)
    model = YOLO('yolov8m.pt')

    # data.yaml 파일 경로
    DATA_YAML_PATH = r"C:\Users\windows11\Downloads\CBR LNP.v1i.yolov8\data.yaml"

    # 모델 학습 설정
    model.train(
        data=DATA_YAML_PATH,
        epochs=200,
        batch=16,  # 배치 크기 감소
        imgsz=640,
        workers=4,
        project='YOLOv8_training',
        name='experiment1',
        device=0,  # GPU 사용
        amp=True,  # 혼합 정밀도 학습 활성화
        cache='disk',  # 이미지 캐싱으로 학습 속도 향상
        optimizer="Adam",  # Adam 옵티마이저 사용
        patience=50,  # 조기 종료 patience
        save_period=10,  # 10 에폭마다 체크포인트 저장
        lr0=0.001,  # 초기 학습률
        lrf=0.01,  # 최종 학습률 계수
        momentum=0.937,  # SGD 모멘텀/Adam beta1
        weight_decay=0.0005,  # 옵티마이저 가중치 감소
        warmup_epochs=3.0,  # 웜업 에폭
        warmup_momentum=0.8,  # 웜업 초기 모멘텀
        warmup_bias_lr=0.1,  # 웜업 초기 편향 학습률
        box=7.5,  # 박스 손실 가중치
        cls=0.5,  # 분류 손실 가중치
        dfl=1.5,  # DFL 손실 가중치
        close_mosaic=10,  # 마지막 10 에폭에서 모자이크 증강 비활성화
    )

if __name__ == "__main__":
    # Windows에서 안전한 프로세스 생성을 위해 'spawn' 방식 사용
    multiprocessing.set_start_method("spawn", force=True)
    main()
