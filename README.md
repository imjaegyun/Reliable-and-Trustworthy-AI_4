# Reliable and Trustworthy AI Assignment 4

이 저장소는 과제 4의 α,β-CROWN 실험을 정리한 저장소입니다. 과제 3에서 사용한 Marabou 방식과 비교하기 위해, α,β-CROWN에 맞는 YAML 설정과 외부 PyTorch 모델을 준비했습니다.

## 파일 구성

- `assignment4.pdf`: 과제 원본
- `docs/problem1_model_directory_review.md`: α,β-CROWN 모델/설정 구조 조사 내용
- `scripts/build_toy_external_model.py`: 외부 toy 데이터셋과 PyTorch MLP 생성 스크립트
- `custom/toy_model_data.py`: α,β-CROWN에서 불러올 커스텀 모델/데이터 로더
- `configs/toy_linf_robustness.yaml`: L_inf 강건성 검증 설정
- `test.py`: α,β-CROWN 실행 커맨드 구성 및 결과 JSON 저장 스크립트
- `run_all.sh`: conda 환경 준비부터 실행까지 처리하는 쉘 스크립트
- `environment.yml`, `requirements.txt`: 재현용 환경 파일
- `results/verification_result.json`: 실행 결과 또는 환경 점검 결과
- `report.pdf`: 최종 한국어 보고서

## 실행 방법

가장 간단한 방법은 `run_all.sh`를 사용하는 것입니다.

```bash
chmod +x run_all.sh
./run_all.sh
```

위 명령은 conda 환경을 만들고, 필요한 Python 패키지를 설치한 뒤, 모델/데이터 파일이 없으면 생성하고 `test.py`를 dry-run으로 실행합니다. 실제 α,β-CROWN 검증까지 실행하려면 α,β-CROWN 저장소 경로를 넘겨야 합니다.

```bash
ABCROWN_ROOT=/path/to/alpha-beta-CROWN ./run_all.sh --run
```

또는 인자로 직접 지정할 수 있습니다.

```bash
./run_all.sh --run --abcrown-root /path/to/alpha-beta-CROWN
```

## 수동 실행

```bash
conda env create -f environment.yml
conda activate assignment4-abcrown
python -m pip install -r requirements.txt
python scripts/build_toy_external_model.py --seed 42
python test.py --config configs/toy_linf_robustness.yaml
python test.py --config configs/toy_linf_robustness.yaml --run --abcrown-root /path/to/alpha-beta-CROWN
```

현재 저장소 안에는 α,β-CROWN 본체를 포함하지 않았습니다. 따라서 `abcrown.py`를 찾지 못하는 환경에서는 `results/verification_result.json`에 `environment_not_ready`가 기록됩니다. α,β-CROWN을 설치한 뒤 `--run`과 `--abcrown-root`를 함께 지정하면 같은 설정으로 실제 검증 결과를 갱신할 수 있습니다.

## 실험 요약

외부 모델은 2차원 합성 데이터를 분류하는 작은 PyTorch MLP입니다. 입력은 `[0, 1]` 범위로 정규화했고, 검증 속성은 기준 샘플 주변 `L_inf` 반경 `epsilon=0.05`에서 예측이 유지되는지 확인하는 로컬 강건성 문제로 잡았습니다.
