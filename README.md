# UX Heatmap

DeepGaze IIE 기반 AI 시선 예측 히트맵 플랫폼

UI 스크린샷을 업로드하면 사용자의 시선이 어디에 집중될지 예측합니다.

## 구조

```
index.html     — 프론트엔드 (드래그앤드롭 업로드, 히트맵 오버레이, 비교 모드)
server.py      — Flask API 서버 (DeepGaze IIE 모델 구동)
requirements.txt
```

## 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 서버 실행 (첫 실행 시 모델 다운로드 ~500MB)
python server.py

# 3. 브라우저에서 index.html 열기
```

## 기능

- 이미지 드래그앤드롭 / 다중 업로드
- DeepGaze IIE 모델 기반 시선 예측
- 히트맵 오버레이 / 비교 / 히트맵 단독 보기
- 컬러맵 선택 (Jet, Hot, Inferno, Viridis, Turbo)
- 투명도 / 블러 강도 조절
- 주목도 지표 (최대, 평균, 집중 영역 비율)
- 결과 이미지 다운로드

## 시스템 요구사항

- Python 3.9+
- GPU 권장 (CUDA) — CPU에서도 동작하나 느림
