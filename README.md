# OCR-ORT Project

## 프로젝트 개요

이 프로젝트는 **ONNX Runtime Web (WASM)**을 활용하여 브라우저 환경에서 로컬 머신러닝 추론을 수행하는 웹 애플리케이션입니다.
서버로 이미지를 전송하지 않고 클라이언트 사이드에서 **ML(MobileNetV3, MobileNetV4, EfficientNet-Lite0)** 모델을 이용하여 이미지를 분류(영수증/비영수증 등)합니다.

## 데모 동작

![Demo GIF](sample_demo.gif)

## 데모 링크

https://ocr-test-onnxruntime-web-production.up.railway.app

## 주요 기능

### 1. 단일 이미지 분석

- 사용자가 업로드한 이미지를 즉시 분석
- ML 모델을 통한 고속 추론
- 분류 결과와 신뢰도(Confidence) 표시

### 2. 배치 처리 (Batch Processing)

- 여러 개의 이미지를 한 번에 선택하여 순차적으로 분석
- **실시간 처리 현황** 표시 (썸네일, 진행 상태)
- 처리 실패 시에도 중단 없이 다음 이미지 계속 처리

### 3. 상세 분석 통계

배치 처리 완료 후 다음과 같은 상세 통계를 제공합니다:

- **분류 결과 분포**: 각 라벨별 개수 및 비율 (%)
- **평균 성능**: 평균 신뢰도 및 추론 시간
- **성능 벤치마크**: 가장 빠른/느린 추론 시간과 해당 파일명

## 기술 스택

- **Framework**: [Next.js 15](https://nextjs.org/) (App Router)
- **Language**: TypeScript
- **ML Engine**: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM Backend)
- **Model**: MobileNetV3, MobileNetV4, EfficientNet-Lite0 (ONNX format)
- **Styling**: Inline CSS (Clean & Minimal Design)

## 프로젝트 구조

```
ocr_ort/
├── public/
│   ├── model/          # ONNX 모델 및 라벨 파일
│   └── wasm/           # ONNX Runtime WASM 바이너리
├── src/
│   └── app/
│       ├── page.tsx    # 메인 로직 (추론, UI, 배치 처리)
│       └── layout.tsx  # 앱 레이아웃
└── ...
```

## 문제점(260118)

**기본 요약**
| | MobileNetV3 | MobileNetV4 | EfficientNet-Lite0
|-------|-------|-------|-------|
학습 시간 | 174초 | 165초 | 297초
ONNX 파일 용량 | 6.4MB | 10.2MB | 13.7MB

**Confusion Matrix(파이썬 환경)**
→ 전체 학습 데이터 463개(영수증), 463개(비영수증)
→ 전체 테스트 이미지 156개(영수증), 156개(비영수증)

|                      | MobileNetV3 | MobileNetV4 | EfficientNet-Lite0 |
| -------------------- | ----------- | ----------- | ------------------ |
| Confusion Matrix(TP) | 154         | 156         | 156                |
| Confusion Matrix(TN) | 154         | 155         | 154                |
| Confusion Matrix(FP) | 2           | 1           | 2                  |
| Confusion Matrix(FN) | 2           | 0           | 0                  |

**모바일 실제 테스트(맥에서 브라우저 테스트)**
→ 영수증 사진 53개 업로드
→ 배치로 1번에 1개씩 처리

|                | MobileNetV3 | MobileNetV4 | EfficientNet-Lite0 |
| -------------- | ----------- | ----------- | ------------------ |
| 영수증(예측)   | 22          | 26          | 17                 |
| 비영수증(예측) | 31          | 27          | 36                 |
| 평균 추론 시간 | 49ms        | 70ms        | 131ms              |

## 개선할 부분(260118)

**1. 정확도 개선 작업 필요**

- 이를 위해 데이터 증강(Data Augmentation)을 적용하여 학습 데이터의 양을 늘리고, 모델의 일반화 성능을 향상시킬 필요가 있음.

**2. OCR 까지 한번에 로컬 처리 가능한지 여부 확인**

- 모바일 구동 가능 모델을 이용해 OCR 까지 한번에 처리하는 방안 모색

**3. 트레이드-오프 분석**

- 정확도와 속도 사이의 적절한 균형을 맞출 수 있는 모델 선정 및 서버 호출과의 비용을 고려한 분석 필요
