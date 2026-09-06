# PLQ 한국어 사용법

PLQ는 광자 회로와 논리 큐비트, 양자 오류 정정을 연결하는 Python 시뮬레이터입니다. 실제 장비를 제어하거나 실험 데이터 없이 특정 장비의 성능을 예측하는 프로그램은 아닙니다.

## 1. 설치

Python 3.11 이상에서 실행합니다.

```bash
git clone https://github.com/ht13255/PLQ.git
cd PLQ
python -m venv .venv
```

| 운영체제 | 가상환경 활성화 |
| --- | --- |
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

```bash
python -m pip install -e .
python -m plq hom --overlap 0.7 --transmission 0.9
```

`coincidence_probability`가 약 `0.20655`이면 첫 실행이 성공한 것입니다. 설치 상태는 `python -m plq doctor`로 확인합니다. 선택 SDK는 필요한 때만 설치해도 됩니다.

## 2. 가장 짧은 광자 회로 코드

```python
from plq import Circuit

result = Circuit(2).bs(0, 1).loss(0, 0.9).loss(1, 0.9).run([1, 1])
print(result.probabilities())
```

- `Circuit(2)`: 광학 모드 두 개.
- `.bs(0, 1)`: 두 모드를 50:50 빔 스플리터로 혼합.
- `.loss(..., 0.9)`: 해당 모드의 광자 생존 확률 90%.
- `.run([1, 1])`: 각 입력 모드에 광자 한 개씩 준비해 실행.

출력 `(2, 0)`은 첫 모드에 두 광자가 있는 경우이고, `(0, 0)`은 진공입니다. 손실된 경우도 확률에서 사라지지 않습니다. 일반적인 중첩·혼합 상태에는 `FockState.amplitudes`와 `FockState.mixture`를 사용합니다.

## 3. 열잡음과 계산 오차

```python
from plq import Circuit

result = Circuit(1).thermal_loss(
    0, transmission=0.7, mean_photons=0.2, tail_tolerance=1e-13
).run([0])

print(result.probabilities())
print("생략한 열환경 확률:", result.bath_omitted_probability)
print("수치 계산의 trace 차이:", result.numerical_trace_error)
```

`mean_photons`는 해당 환경 모드의 평균 광자 수입니다. 온도 자체나 검출기의 암계수가 아닙니다. 환경에서 들어온 광자가 잘리지 않도록 계산 기저를 확장합니다. 무한한 열분포에서 생략한 확률은 정규화로 숨기지 않고 별도로 보고합니다.

`tail_tolerance`는 각 열환경에서 생략할 확률의 한도이며, 실험과의 오차 한도는 아닙니다. 매우 드문 herald 결과를 비교한다면 생략 확률이 그 성공 확률보다 충분히 작아야 합니다. `python examples/thermal_accuracy.py`는 환경의 광자 수 한도를 늘리며 수렴을 확인합니다.

## 4. 작은 코드의 오류율을 완전 열거로 계산

```python
from plq import five_qubit_code, MemoryNoise, exact_pauli_memory

result = exact_pauli_memory(
    five_qubit_code(), MemoryNoise(px=0.001, py=0.001, pz=0.15)
)
print(result.logical_error_rate)
```

결과는 약 `0.03643846138`입니다. 기본 디코더는 같은 논리 효과를 내는 오류들의 확률을 합산해 복구를 선택합니다. 샷을 무작위로 뽑지 않으므로 표본 오차가 없습니다. 부동소수점 오차와 입력 잡음 모델의 불확실성은 남습니다.

이 함수는 **독립 Pauli 잡음, 한 번의 이상적 신드롬 측정과 복구**에 해당합니다. 여러 라운드·erasure·측정 잡음에는 `simulate_memory`, 구체적인 Clifford 측정 회로에는 Stim을 사용합니다. 이 결과를 실제 광자 하드웨어의 오류율이나 임계값으로 해석하면 안 됩니다.

```bash
python examples/exact_memory.py
python -m plq memory examples/configs/exact_memory.json --output results/exact.json
```

## 5. 설정 파일과 추가 기능

```bash
python -m plq optics examples/configs/optics.json --output results/optics.json
python -m plq optics examples/configs/thermal.json --output results/thermal.json
python -m plq memory examples/configs/memory.json --output results/memory.json
python examples/optical_logical.py
```

광학 JSON의 `max_photons`를 생략하면 입력으로부터 초기 총 광자 수 한도를 계산합니다. 직접 지정한 한도 때문에 광원 분포 일부가 제외되면 `source_omitted_probability`에 기록됩니다. memory 설정 안의 코드 파일 경로는 해당 JSON 파일의 위치가 기준입니다.

| 필요한 기능 | 추가 설치 |
| --- | --- |
| Quandela Perceval | `python -m pip install -e ".[perceval]"` |
| PennyLane | `python -m pip install -e ".[pennylane]"` |
| Stim + PyMatching | `python -m pip install -e ".[qec]"` |
| 테스트와 고정밀 기준 계산 | `python -m pip install -e ".[test,reference]"` |
| 전체 | `python -m pip install -e ".[all]"` |

```bash
python scripts/validate.py --core
```

전체 SDK 설치 후에는 `python scripts/validate.py`로 통합 검증합니다. `ResourceLimitError`는 명시한 계산 예산을 넘었다는 뜻입니다. `Precision(max_dimension=..., max_kraus_bytes=...)` 또는 디코더의 `max_patterns`를 높일 수 있지만, 행렬과 중간 배열이 사용할 메모리를 먼저 계산해야 합니다. `atol`을 줄이는 것만으로 complex128이 임의 정밀도 연산으로 바뀌지는 않습니다.

세부 규칙은 [API](API.md), [물리 모델](PHYSICS.md), [확장 방법](EXTENDING.md), [검증 보고서](VALIDATION.md)에 있습니다.
