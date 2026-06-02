# Problem 2: 외부 모델 + 데이터셋으로 α,β-CROWN 실행 기록

## 1. 실행 설정

- 모델: `models/toy_binary_mlp.pt`
- 모델 타입: PyTorch (2-layer MLP)
- 데이터셋: `data/toy_dataset.csv` (2D 합성 데이터, label 2 클래스)
- 실험용 기준점: `data/toy_reference_point.json`
- 설정 파일: `configs/toy_linf_robustness.yaml`

## 2. 실행 준비/문제

- `build_toy_external_model.py`로 모델 및 데이터 생성은 완료
- 검증 실행은 `test.py --run`으로 수행
- 현재 레포지토리 환경에는 α,β-CROWN 저장소/실행 파일이 없어
  `abcrown.py` 미탐색 상태

## 3. 수집된 결과

`results/verification_result.json` 내용:
- `status`: `environment_not_ready`
- 사유: `abcrown.py not found. Clone and install alpha-beta-CROWN first.`

## 4. 다음 단계

- 사용자 환경에서 α,β-CROWN 설치 후 위 설정을 동일하게 실행하면,
  `results/verification_result.json`의 `status`를 `verified`/`falsified`/`timeout`
  중 하나로 갱신해 성능 및 결과를 보고서에 반영

## 5. 확인 포인트

- 모델 포맷: PyTorch checkpoint
- 데이터 포맷: CSV
- 속성: L_inf 반경 epsilon=0.05 근방의 로컬 강건성(템플릿)
