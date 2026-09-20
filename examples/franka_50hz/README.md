# fix3 50Hz — 클러스터 제출 안내

이번 패키지는 **fix3의 성공 데모 한 개**를 50Hz 전체 1,291프레임으로 변환한 새 실험입니다. 다른 장면의 데이터셋 전체를 변환한 패키지가 아닙니다. 기존 fix3 체크포인트를 이어 학습하지 않고 pi05_base에서 새로 학습합니다.

## 맞춘 설정

| 항목 | 값 |
|---|---|
| 학습 설정 | `pi05_franka_pnp_overfit1_fix3_50hz` |
| 데이터셋 | `lithyeon/franka_pnp_overfit1_fix3_50hz` |
| 데이터·실행 속도 | 50Hz, 액션 하나당 0.02초 |
| 미래 액션 개수 | 50개, 시간 간격 0.02초 → 약 1초 분량 |
| 학습 배치 / 스텝 | 32 / 20,000 |
| 증강 / EMA | 이미지 증강 없음 / 0.999 |
| 학습률 | warmup 1,000, 최대 2.5e-5, 마지막 2.5e-6 |
| 정규화 | 이 데이터 전체 1,291프레임과 미래 50개 delta 액션으로 새로 계산 |
| 평가 설정 | `control_stride=1`, `replan_every=25`, 정확 초기 상태 복원 |

`replan_every=25`는 모델이 만든 명령 25개를 0.5초 동안 실행하고 다시 관측한다는 뜻입니다. 매초 50번 모델을 호출하는 설정이 아닙니다.

## Hugging Face에서 받기

비공개 저장소: https://huggingface.co/datasets/lithyeon/franka_pnp_overfit1_fix3_50hz

클러스터 로그인 노드에서 Hugging Face 계정에 로그인하고 패키지를 받습니다.
기존 openpi 환경이 있다면 그 환경의 Python으로 실행할 수 있습니다.
로그인 토큰은 터미널의 비밀번호 입력란에만 입력합니다.

```bash
python -c 'from huggingface_hub import login; login()'
python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lithyeon/franka_pnp_overfit1_fix3_50hz",
    repo_type="dataset",
    allow_patterns=["training/*"],
    local_dir="franka-50hz-download",
)
PYCODE
cd franka-50hz-download/training
```

`huggingface_hub`가 없다면 먼저 `python -m pip install huggingface_hub`로 설치합니다.
다운로드에는 데이터, norm stats, 학습 코드가 모두 들어 있는 압축 파일이 포함됩니다.
아래 1번의 체크섬 확인과 압축 해제부터 이어 진행합니다.

## 1. 패키지를 클러스터로 옮기기

`franka_50hz_training_bundle.tar.gz`와 `.sha256` 파일을 클러스터의 작업 디렉터리에 복사합니다. 기존 openpi 폴더에 덮어쓰지 말고 별도 디렉터리에 풉니다.

```bash
sha256sum -c franka_50hz_training_bundle.tar.gz.sha256
tar -xzf franka_50hz_training_bundle.tar.gz
cd openpi-franka-50hz
```

소스 코드, 잠금 파일, 데이터 압축 파일, 정규화 통계, 제출·검증 스크립트가 포함되어 있습니다. 시뮬레이터와 베이스 모델 가중치는 포함하지 않습니다. 클러스터 학습에는 Isaac Sim이 필요 없습니다.

## 2. 로그인 노드에서 준비하기

Python 3.11 환경을 사용합니다. 의존성과 베이스 모델을 처음 받는 단계는 네트워크 연결이 필요합니다.

```bash
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen

# 데이터와 모델 캐시는 계산 노드에서도 보이는 경로여야 합니다.
export HF_LEROBOT_HOME="$HOME/.cache/huggingface/lerobot"
# 모델 캐시 위치를 바꾸려면 준비 단계부터 OPENPI_DATA_HOME을 설정합니다.
# export OPENPI_DATA_HOME=/shared/path/openpi-cache

.venv/bin/python scripts/install_franka_50hz_data.py data/franka_pnp_overfit1_fix3_50hz.tar.gz
JAX_PLATFORMS=cpu .venv/bin/python scripts/prepare_franka_50hz.py
```

설치기는 데이터 파일을 모두 체크섬으로 확인합니다. 같은 데이터가 이미 설치되어 있으면 검증하고, 다른 내용이면 덮어쓰지 않고 중단합니다. `prepare`는 정규화 확인과 베이스 모델·토크나이저 사전 다운로드를 합니다. 로그인 노드에서 사용한 `JAX_PLATFORMS=cpu`를 GPU 잡의 환경변수로 export하지 마세요.

전송 후 전수 검사를 다시 실행하려면:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_franka_50hz.py
```

## 3. 잡 제출하기

```bash
export CHECKPOINT_BASE_DIR="$HOME/openpi_runs"
export EXP_NAME=overfit1_fix3_50hz_8g
export GPU_COUNT=8
sbatch scripts/slurm/franka_50hz.sbatch
```

기본 Slurm 요청은 기존 제출 설정을 바탕으로 한 `A100-80GB / hpgpu / GPU 8개 / CPU 16개 / RAM 160G / 24시간`입니다. 현재 클러스터의 큐 가용성·계정 제한과 전체 학습 소요 시간은 이 로컬 환경에서 확인하지 않았습니다. 사이트 제한에 맞춰 `sbatch --partition=... --qos=... --time=...`로 바꿀 수 있습니다. 4GPU를 사용하면 `GPU_COUNT=4 sbatch --gres=gpu:4 ...`로 두 값을 함께 바꿉니다.

잡은 다음 순서로 실행됩니다.

1. 데이터·정규화 체크섬, GPU 할당, 저장 공간 확인.
2. **실제 전체 pi0.5 모델 + 배치 32 + horizon 50으로 2번 학습 업데이트.** 메모리 부족, NaN, 가중치 로딩 오류가 나면 여기서 중단.
3. 검사를 통과하면 새 모델을 초기화하고 본 학습 20,000스텝 시작.

검사에 쓴 모델을 본 학습으로 이어 쓰지 않습니다. 검사 과정의 업데이트는 본 학습 스텝 수에 포함되지 않습니다.

사전 학습 검사만 실행하려면 `SMOKE_ONLY=1 sbatch ...`를 사용합니다. 저장 공간은 최소 약 152GiB 여유를 검사합니다. 베이스 모델 캐시·데이터는 별도이며, 계정 quota는 파일시스템 잔여 공간과 별도로 확인해야 합니다.

## 4. 중단 시 이어 학습하기

같은 50Hz run에 대해서만:

```bash
RESUME=1 sbatch scripts/slurm/franka_50hz.sbatch
```

`EXP_NAME`, `CHECKPOINT_BASE_DIR`, `GPU_COUNT`, 데이터 캐시 환경변수를 처음과 같게 유지합니다. 이전 10Hz 체크포인트는 이 실험의 resume 대상이 아닙니다. `--overwrite`는 사용하지 않습니다.

## 5. 학습 결과 가져오기

학습을 재개할 수 있는 전체 체크포인트는 최신 것 하나를 유지합니다. 시뮬에서 사용할 가중치+정규화는 다음 세 시점에 별도로 보존합니다.

```text
$CHECKPOINT_BASE_DIR/pi05_franka_pnp_overfit1_fix3_50hz/<EXP_NAME>_inference/5000
$CHECKPOINT_BASE_DIR/pi05_franka_pnp_overfit1_fix3_50hz/<EXP_NAME>_inference/10000
$CHECKPOINT_BASE_DIR/pi05_franka_pnp_overfit1_fix3_50hz/<EXP_NAME>_inference/19999
```

이 `_inference` 폴더 안의 **해당 스텝 디렉터리 전체**를 가져옵니다. `params`와 `assets`를 함께 보존해야 합니다. 이 경량 내보내기는 학습 resume용이 아닙니다.

## 6. 시뮬 평가하기 — 로컬 시뮬 머신에서

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py --port 8004 policy:checkpoint \
  --policy.config=pi05_franka_pnp_overfit1_fix3_50hz \
  --policy.dir=/path/to/downloaded/19999

# 별도 터미널
POLICY_PORT=8004 SIM_GPU=6 EPISODES=5 RUN_TAG=fix3_50hz_19999 \
  bash scripts/eval_franka_overfit1_50hz.sh
```

현재 로컬 시뮬 프로젝트의 `Tools/eval/pi05_eval.py`에는 50Hz metadata 확인과 horizon 50 지원을 반영했습니다. 새 스크립트는 기존 10Hz 서버나 주기가 다른 서버를 거부합니다. 다른 시뮬 머신에 적용할 때는 패키지의 `simulator/pi05_eval.py`도 해당 시뮬 프로젝트의 evaluator 위치에 반영해야 합니다. 해당 머신의 `evalenv.py`에는 `initial_state_hdf5` 복원 기능이 있어야 합니다. 클러스터 학습만 한다면 이 시뮬 파일은 사용하지 않습니다.

## 검증의 범위

로컬 검사 결과는 `validation/` 또는 원본 체크아웃의 `diagnostics/overfit1_fix3_50hz/`에 있습니다. 데이터 전수 검사, 새 데이터 액션으로 실제 시뮬 2회 성공, 전체 베이스 모델의 50-action 추론, 작은 모델의 실제 gradient update를 확인했습니다. 전체 배치의 **대형 모델 역전파와 GPU 메모리 적합성은 클러스터 잡 앞의 필수 검사**가 확인합니다.

50Hz 일치는 확인된 실행 불일치를 제거합니다. 새로 학습한 정책의 폐루프 성공률은 학습 후 시뮬에서 별도로 확인해야 합니다. 사전 검사 통과만으로 성공률을 보장할 수는 없습니다.
