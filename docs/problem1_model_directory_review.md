# Problem 1: α,β-CROWN models/구성 탐색 요약

이 문서는 과제4 요구사항에 맞게 α,β-CROWN의 실행 설계 기준을 정리한 것이다.

## 1. 모델/설정 형식의 핵심

- 입력은 YAML 기반 설정 파일 1개로 처리된다.
- 모델은 크게 두 가지 형식을 지원한다.
  - PyTorch 모델
  - ONNX 모델
- `abcrown.py --config <파일>` 실행으로 통합 검증을 수행한다.
- 출력 상태는 보통 `verified`, `falsified`, `timeout/unknown` 계열로 정리된다.

## 2. 과제 목표에 필요한 탐색 포인트

- `models` 폴더에서는 실험용으로 재사용 가능한 구성 예시가 존재한다.
- `exp_configs` 폴더에는 데이터셋/알고리즘/모델 유형별 템플릿이 존재한다.
- 과제 요구에 맞게,
  - 내 모델을 외부 모델로 분리
  - ONNX 또는 PyTorch 형식으로 변환/저장
  - YAML에서 모델 경로, 데이터셋 로더, specification(norm 타입/epsilon), solver 파라미터
    를 명시
  하는 방식이 필요함.

## 3. Marabou와의 비교 포인트(계획)

- Marabou: SMT 기반 제약해결기 특성.
- α,β-CROWN: bound propagation + branch-and-bound 기반으로 대형 모델에 대해
  더 빠른 불완전/완전 조합 검증이 가능한 구성이 강점.
- 과제에서는 설정 파일 추적, 모델 형식(Pytorch/ONNX), norm 기반 강건성(spec) 작성,
  실행 시간/결과 상태/타임아웃의 실측 비교가 핵심이다.
