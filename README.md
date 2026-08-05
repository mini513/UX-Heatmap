# UX Heatmap

DeepGaze IIE 기반 AI 시선 예측 히트맵 플랫폼

UI 스크린샷을 업로드하면 사용자의 시선이 어디에 집중될지 예측합니다.

## 구조

```
index.html          — 프론트엔드 (드래그앤드롭, 클립보드 붙여넣기, 히트맵 오버레이, 비교 모드)
server.py           — Flask API 서버 (DeepGaze IIE 모델 구동)
setup.py            — 오프라인 원클릭 설치 스크립트
requirements.txt    — Python 의존성
vendor/             — deepgaze_pytorch 소스코드 백업 (오프라인 사용)
```

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# deepgaze_pytorch는 PyPI에 없으므로 GitHub에서 설치
pip install git+https://github.com/matthias-k/DeepGaze.git
pip install git+https://github.com/openai/CLIP.git
pip install einops boltons

# 2. 서버 실행 (첫 실행 시 모델 다운로드 ~885MB)
python server.py

# 3. 브라우저에서 index.html 열기
```

## 오프라인 설치 (원본 저장소가 사라진 경우)

이 저장소에는 모든 소스코드가 백업되어 있습니다:

```bash
# 원클릭 설치 — 패키지 + vendor 소스 + 모델 가중치 자동 다운로드
python setup.py
```

**setup.py가 하는 일:**
1. Flask, PyTorch 등 pip 패키지 설치
2. `vendor/deepgaze_pytorch` 소스코드를 site-packages에 복사
3. 모델 가중치 5개 파일(총 885MB) 다운로드
4. 설치 검증

### 모델 가중치 수동 다운로드

자동 다운로드가 안 될 경우, 아래 파일들을 `~/.cache/torch/hub/checkpoints/`에 저장:

| 파일 | 크기 | URL |
|------|------|-----|
| deepgaze2e.pth | 400MB | [GitHub Releases](https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth) |
| resnet50_finetune_*.pth.tar | 195MB | [Bitbucket](https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar) |
| efficientnet-b5-b6417697.pth | 117MB | [GitHub Releases](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth) |
| densenet201-c1103571.pth | 77MB | [PyTorch](https://download.pytorch.org/models/densenet201-c1103571.pth) |
| resnext50_32x4d-7cdf4587.pth | 96MB | [PyTorch](https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth) |

## 기능

- 이미지 드래그앤드롭 / 다중 업로드 / Ctrl+V 클립보드 붙여넣기
- DeepGaze IIE 모델 기반 시선 예측 (MIT Saliency Benchmark 1위)
- 히트맵 오버레이 / 비교 / 히트맵 단독 보기
- 컬러맵 선택 (Jet, Hot, Inferno, Viridis, Turbo)
- 투명도 / 블러 강도 조절
- 주목도 지표 (최대, 평균, 집중 영역 비율)
- 결과 이미지 다운로드

## 시스템 요구사항

- Python 3.9+
- GPU 권장 (CUDA) — CPU에서도 동작하나 느림 (~10~30초/이미지)

## 라이선스

- DeepGaze: MIT License (Matthias Kümmerer, Bethge Lab, University of Tübingen)
- CLIP: MIT License (OpenAI)
- 프론트엔드/서버 코드: 자유 사용
