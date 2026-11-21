# AI가 어떻게 학습하는지 수학으로 직접 풀어보기

A2-1 필수 과제. NumPy로 행렬 변환과 미분, 역전파, 경사하강법, 손실의 확률 모형을 실험한다. Matplotlib으로 실험 결과를 그린다.

행렬 곱과 기울기가 신경망 학습에서 어떤 역할을 하는지는 [역전파 노트북](notebooks/backprop_derivation.ipynb)의 고정 예제로 살펴본다. [확률과 손실 노트북](notebooks/probability_loss.ipynb)은 MSE와 교차엔트로피가 각각 어떤 확률 가정에서 나오는지 보여 준다.

## 실행

저장소 루트에서 Python 가상환경을 만든 뒤 `requirements.txt`를 설치한다. 아래 방법은 Windows PowerShell과 Python 3.12.10에서 확인했다.

```powershell
$a2Env = Join-Path $env:USERPROFILE 'Documents/Codex/venvs/a2-1-py312'
python -m venv $a2Env
$a2Python = Join-Path $a2Env 'Scripts/python.exe'
& $a2Python -m pip install -r requirements.txt
& $a2Python -X utf8 -m src.linear_algebra
& $a2Python -X utf8 -m src.calculus
& $a2Python -X utf8 -m src.optimizer
```

macOS·Linux에서는 가상환경을 활성화한 뒤 같은 모듈을 실행한다.

```bash
python -m venv ~/Documents/Codex/venvs/a2-1-py312
source ~/Documents/Codex/venvs/a2-1-py312/bin/activate
python -m pip install -r requirements.txt
python -m src.linear_algebra
python -m src.calculus
python -m src.optimizer
```

세 모듈은 `outputs/`에 01-06번 PNG를 저장하고 결과를 터미널에 출력한다. 노트북은 같은 Python 환경의 Jupyter 또는 VS Code에서 열고 모든 셀을 실행한다. [역전파 노트북](notebooks/backprop_derivation.ipynb)은 손계산과 NumPy 결과를 확인한다. [확률과 손실 노트북](notebooks/probability_loss.ipynb)은 07-08번 PNG를 저장한다.

## 필수 산출물과 결과

| 파일 | 결과 및 증거 |
| --- | --- |
| [src/linear_algebra.py](src/linear_algebra.py) | [회전·스케일링·전단](outputs/01_transformations.png)의 최대 면적비 상대 오차는 $8.882\times10^{-16}$ (기준 1%). Power Iteration 고유값 4.6180339887, NumPy 대비 상대 오차 $1.923\times10^{-16}$ (기준 5%). [SVD](outputs/02_svd.png)는 요청 $k=10,50,100$에서 실제로 10, 50, 64개 성분을 비교한다. |
| [src/calculus.py](src/calculus.py) | $f'(3)$의 중심차분 결과 6.0000000000, 절대 오차 $3.931\times10^{-11}$ (기준 $10^{-4}$). [Gradient와 접선](outputs/03_gradient.png)의 내적은 실험 점에서 0. |
| [notebooks/backprop_derivation.ipynb](notebooks/backprop_derivation.ipynb) | 입력 2개, 은닉 2개, 출력 1개의 순전파와 역전파 유도. shape를 표시하고 13개 값이 손계산과 소수점 넷째 자리까지 일치함을 확인. 파라미터 기울기를 중심차분과 비교한 최대 절대 오차는 $1.036\times10^{-10}$. |
| [src/optimizer.py](src/optimizer.py) | [원형 함수](outputs/04_gd_momentum_circle.png)에서 학습률 0.1로 GD 100회 후 원점 거리 $1.4404\times10^{-9}$ (기준 0.1). [학습률 실험](outputs/05_learning_rate.png)은 수렴·진동·발산을 비교한다. [타원형 함수](outputs/06_gd_momentum_ellipse.png)의 학습률 0.01에서 손실 0.01 이하를 유지하기 시작한 반복은 GD 194회, Momentum 97회. |
| [notebooks/probability_loss.ipynb](notebooks/probability_loss.ipynb) | [정규분포 PDF](outputs/07_normal_pdf.png), [베르누이 PMF](outputs/08_bernoulli_pmf.png), 정규·베르누이·카테고리 우도 유도. 소프트맥스 확률 합의 오차 $1.110\times10^{-16}$ (기준 $10^{-6}$). 소프트맥스와 Cross-Entropy의 기울기 $p-t$를 중심차분으로 확인했으며 최대 절대 오차는 $3.399\times10^{-11}$. |
| [README.md](README.md) | 개요, 실행 방법, 결과 요약. |

SVD 실험은 [64×64 회색조 입력 배열](data/grace_hopper_gray64.npy)을 NumPy로 읽는다. 이 배열은 Matplotlib에 포함된 Grace Hopper의 미 해군 공식 초상(`grace_hopper.jpg`, NH 96919-KN)을 회색조로 바꾸고 중앙을 기준으로 64×64에 맞춰 만든 것이다.

## 원본의 조건과 실험에서의 처리

| 원본 내용 | 이 저장소의 실행 결과 | 근거 |
| --- | --- | --- |
| 고정 예제에 정답 레이블이 비어 있음 | $y=1$로 명시 | 원본 7쪽의 참고 예제에서 사용하는 레이블이다. |
| $z_2=0.6071$, $\partial L/\partial z_1=[-0.0440,-0.0513]$ | $z_2=0.6072$, $\partial L/\partial z_1=[-0.0440,-0.0517]$ | 중간값의 자릿수를 유지하고 최종 결과를 반올림했다. 파라미터 기울기를 [중심차분](notebooks/backprop_derivation.ipynb)으로 독립 확인했다. |
| $x^2+y^2$에서 학습률 0.5 이상이면 발산 | 0.5는 1회에 원점 도달, 1.0은 두 점 사이 왕복, 1.1은 발산 | GD는 좌표에 $1-2\eta$를 곱한다. [학습률 그림](outputs/05_learning_rate.png) |
| $x^2+y^2$, 학습률 0.1에서 100회 후 GD $(0.08, 0.08)$, Momentum $(0.02, 0.02)$가 더 빠르게 수렴 | GD는 $(1.0185\times10^{-9},1.0185\times10^{-9})$, Momentum은 $(-0.01425706,-0.01425706)$. 손실 0.01 이하를 유지하기 시작한 시점은 GD 20회, Momentum 77회 | Momentum은 원점을 지나 왕복한다. [원형 함수 그림](outputs/04_gd_momentum_circle.png) |
| $x^2+10y^2$, 학습률 0.05에서 GD는 지그재그로 느리게, Momentum은 부드러운 곡선으로 빠르게 수렴 | 0.05에서 GD는 한 번에 $y=0$에 도달한다. 손실 0.01 이하를 유지하기 시작한 시점은 GD 38회와 Momentum 97회다. 0.01에서는 GD 194회와 Momentum 97회다. | 0.05에서 GD의 $y$ 좌표 배율은 $1-20\times0.05=0$이다. 0.01에서 Momentum 경로는 원점 부근에서 진동한다. [타원형 함수 그림](outputs/06_gd_momentum_ellipse.png) |
| Power Iteration 51회, 허용 오차 $10^{-6}$, $\lambda=4.6180$, $v=[0.8507,0.5257]$ | 33회에 잔차 $\lVert Av-\lambda v\rVert_2\le10^{-9}$ 충족. 고유값과 고유벡터는 4자리 반올림 기준으로 일치 | 원본은 허용 오차를 제시하고 종료 판정식은 생략한다. [반복 구현](src/linear_algebra.py) |
| SVD 성분 수 $k=10,50,100$ | 64×64 입력에서 실제로 10, 50, 64개 성분을 사용한다. $k=100$ 요청에서는 전체 성분으로 복원한다. | 64×64에서는 최대 64개 성분을 쓸 수 있다. [SVD 그림](outputs/02_svd.png) |

원본의 결과 예시는 재현 결과를 비교하기 위한 참고값이다.
