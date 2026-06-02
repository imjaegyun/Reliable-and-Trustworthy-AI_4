# α,β-CROWN을 활용한 외부 PyTorch 모델 강건성 검증

## 초록

본 보고서는 신경망 검증 도구 α,β-CROWN의 모델 및 설정 구조를 조사하고, 기본 예제에 포함되지 않은 외부 PyTorch 모델을 대상으로 로컬 강건성 검증을 수행한 결과를 정리한다. 과제 3에서 사용한 Marabou가 SMT 기반 제약 해결 방식에 가까웠다면, α,β-CROWN은 bound propagation과 branch-and-bound를 결합해 출력 경계의 상·하한을 계산하는 방식이다. 본 과제에서는 작은 외부 PyTorch 모델을 직접 만들고, YAML 설정과 자동 실행 스크립트를 구성한 뒤 실제 verifier 실행 결과까지 기록했다.

## 1. α,β-CROWN 모델 및 설정 구조 조사

α,β-CROWN의 실행은 `abcrown.py --config <yaml>` 형태로 이루어진다. 모델, 데이터셋, 검증 속성, solver 옵션이 하나의 YAML 파일에 정리되므로, 실험 조건을 코드 안에 흩어 두는 방식보다 재현성이 좋다. 저장소의 `complete_verifier/models`에는 toy 모델, CIFAR 계열 CNN/ResNet, VNN-COMP용 모델, non-ReLU 모델, custom specification 예제가 포함되어 있다. 또한 `complete_verifier/exp_configs`에는 tutorial 예제와 VNN-COMP 벤치마크별 YAML 설정이 분리되어 있어, 모델 종류와 검증 목적에 따라 설정을 수정해 사용할 수 있다.

Marabou와 비교했을 때 가장 큰 차이는 specification을 표현하는 방식이다. Marabou에서는 입력 bounds와 출력 제약을 코드에서 직접 구성하는 경우가 많았지만, α,β-CROWN에서는 모델 경로, 데이터 로더, norm, epsilon, timeout, branching strategy 등을 YAML에 명시한다. 이 방식은 반복 실험에는 유리하지만, 처음 사용할 때는 α,β-CROWN이 기대하는 key 이름과 커스텀 loader 형식을 정확히 맞춰야 한다.

## 2. 외부 모델과 데이터셋 구성

이번 실험에서는 α,β-CROWN 기본 모델 디렉터리에 포함되지 않은 작은 PyTorch MLP를 직접 생성했다. 데이터셋은 두 개의 2차원 Gaussian cloud를 만든 뒤 `[0, 1]` 범위로 정규화한 이진 분류 데이터이다. 모델은 `Linear(2, 16) -> ReLU -> Linear(16, 2)` 구조이며, 입력 차원이 낮고 네트워크가 작아 검증기 설정을 확인하기에 적합하다.

| 항목 | 내용 |
| --- | --- |
| 모델 파일 | `models/toy_binary_mlp.pt` |
| 모델 구조 | 2 입력, hidden 16, ReLU, 2 출력 MLP |
| 데이터셋 | `data/toy_dataset.csv` |
| 커스텀 로더 | `custom/toy_model_data.py` |
| 검증 설정 | `configs/toy_linf_robustness.yaml` |
| 검증 속성 | 기준 입력 주변 `L_inf` 반경 `epsilon=0.05` 로컬 강건성 |

검증 property는 기준 샘플 주변의 작은 perturbation 영역 안에서 모델의 분류 결과가 유지되는지 확인하는 문제로 설정했다. 커스텀 dataloader는 α,β-CROWN 형식에 맞게 `(X, labels, data_max, data_min, eps)`를 반환하도록 작성했다.

## 3. 실행 파이프라인 및 결과

실행 흐름은 `run_all.sh`와 `test.py`로 구성했다. `run_all.sh`는 conda 환경을 준비하고, 모델과 데이터 파일이 없으면 생성한 뒤, alpha-beta-CROWN 저장소가 없으면 `external/alpha-beta-CROWN`에 clone한다. 이후 공식 `complete_verifier/environment.yaml`로 `alpha-beta-crown` 환경을 만들고, 그 환경의 Python으로 `abcrown.py`를 실행한다.

실행 결과는 `results/verification_result.json`에 기록했다. 현재 실행에서는 단일 기준 샘플에 대해 property가 verified로 판정되었다.

| 항목 | 결과 |
| --- | --- |
| 전체 스크립트 | `./run_all.sh --timeout 60` 정상 종료 |
| `test.py` 상태 | `verified` |
| return code | `0` |
| `test.py` 측정 시간 | 약 `4.71`초 |
| α,β-CROWN 검증 시간 | 약 `0.048`초 |
| 요약 | 1개 instance 중 1개 safe verified, falsified 0개, timeout 0개 |

초기 CROWN bound만으로 검증이 끝났고, branch-and-bound를 깊게 진행하기 전에 `safe-incomplete` 결과가 나왔다. 모델과 입력 차원이 매우 작고 기준점의 분류 margin이 큰 편이어서, `epsilon=0.05` 범위에서는 출력 차이가 충분히 유지된 것으로 해석된다.

## 4. Marabou와의 비교 및 해석

Marabou는 입력 변수와 출력 제약을 명시적으로 구성하면서 검증 조건을 직접 다룰 수 있다는 장점이 있다. 작은 네트워크에서는 어떤 제약이 들어가는지 확인하기 쉬웠다. 반면 α,β-CROWN은 YAML 설정을 중심으로 모델, 데이터, specification, solver를 묶어 관리하므로 실험 조건을 재사용하기 쉽다. 특히 bound propagation과 branch-and-bound를 활용하기 때문에 큰 ReLU 네트워크에서는 Marabou보다 확장성 있는 검증을 기대할 수 있다.

이번 실험에서 가장 크게 느낀 차이는 환경과 설정 파일의 중요성이다. α,β-CROWN은 모델 정의 함수, checkpoint 로딩, 데이터 로더 반환 형식, YAML key가 모두 맞아야 실행된다. 처음 작성한 설정에서는 최신 공식 config에 없는 key를 사용해 실패했지만, 공식 tutorial 형식에 맞춰 `specification.type: lp`와 `data.dataset: Customized(...)` 형태로 바꾸자 정상적으로 실행되었다. 즉, α,β-CROWN은 한 번 구조를 맞추면 빠르게 검증되지만, 그 전 단계의 설정 정합성이 중요하다.

## 5. 한계 및 결론

본 실험은 작은 2차원 MLP에 대한 단일 로컬 강건성 검증이므로, 대형 이미지 모델에서의 성능을 직접 보여주지는 않는다. 그러나 외부 PyTorch 모델을 α,β-CROWN 형식에 맞게 연결하고, 자동 실행 스크립트로 설치부터 검증 결과 기록까지 재현 가능하게 만든 점에서 과제의 핵심 요구사항을 만족한다.

결론적으로 α,β-CROWN은 설정 기반 실험 관리와 빠른 bound 기반 검증 측면에서 강점이 있다. 반면 초기 환경 구성과 YAML 형식 이해가 필수적이므로, 작은 외부 모델로 먼저 파이프라인을 검증한 뒤 더 큰 모델로 확장하는 접근이 적절하다고 판단했다.
