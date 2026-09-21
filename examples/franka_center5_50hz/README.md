# center5_v2: 50개 성공 데모, 50Hz

기존 center5_v2 성공 데모 50개를 원본 주기 그대로 변환한 데이터입니다.
69,898프레임, 카메라 영상 150개(320×320, 50fps), 각 액션 간격 20ms입니다.
상태는 액션 실행 전의 절대 관절 위치이고 액션은 절대 관절 목표 7개와 그리퍼 명령입니다.
마지막 프레임까지 보존하며 미래 액션은 각 에피소드 끝에서 마지막 액션으로 패딩합니다.

- Hugging Face (비공개): https://huggingface.co/datasets/lithyeon/franka_pnp_center5_v2_50hz
- 학습 설정: `pi05_franka_pnp_center5_v2_50hz`
- 미래 액션: 50개 × 20ms = 1초
- norm stats: 전체 69,898프레임의 상태와 50개 미래 delta 액션으로 새로 계산
- 기존 center5_v2 설정처럼 이미지 증강 ON, batch 32, 20,000스텝, pi05_base에서 새 학습
- 시뮬 평가: `control_stride=1`, `replan_every=25`, `expected_action_fps=50`

## 클러스터에서 받기

기존 저장소를 사용하면 먼저 `franka_pnp` 브랜치의 최신 코드를 받습니다.
로컬 수정이 있으면 보존하고 별도 폴더에 clone 하세요.

```bash
git clone --branch franka_pnp https://github.com/llishyun/openpi.git openpi-center5-50hz
cd openpi-center5-50hz
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
.venv/bin/python -c 'from huggingface_hub import login; login()'

# 계산 노드에서도 보이는 경로를 사용합니다.
export HF_LEROBOT_HOME="$HOME/.cache/huggingface/lerobot"
export OPENPI_FRANKA_CONFIG=pi05_franka_pnp_center5_v2_50hz
.venv/bin/python scripts/download_franka_center5_50hz.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/prepare_franka_50hz.py
```

norm stats는 Git의 해당 설정 assets 폴더와 Hugging Face의 `training/norm_stats.json`에 동일하게 보관됩니다.
준비 스크립트는 데이터·norm stats 체크섬을 검사하고 베이스 모델·토크나이저를 미리 받습니다.
`JAX_PLATFORMS=cpu`를 GPU 학습 잡 전체에 export하지 마세요.

## 잡 제출

```bash
export CHECKPOINT_BASE_DIR="$HOME/openpi_runs"
export EXP_NAME=center5_v2_50hz_8g
export GPU_COUNT=8
sbatch scripts/slurm/franka_center5_50hz.sbatch
```

기본값은 A100-80GB 8개, hpgpu, CPU 16개, RAM 160G, 24시간입니다.
클러스터 큐/계정 제한에 맞게 Slurm 옵션을 조정하세요. 4GPU는 `GPU_COUNT=4`와 `--gres=gpu:4`를 함께 지정합니다.
잡은 데이터 검증과 실제 전체 모델·배치32의 2스텝 학습 검사를 통과해야 본 학습을 시작합니다.
전체 체크포인트용 약 152GiB의 여유 공간을 검사하며 계정 quota는 별도로 확인해야 합니다.

기존 10Hz 체크포인트를 이어 학습하지 않습니다. 같은 50Hz 실험의 재개에만 `RESUME=1`을 사용하세요.
시뮬용 가중치와 norm stats는 `<CHECKPOINT_BASE_DIR>/pi05_franka_pnp_center5_v2_50hz/<EXP_NAME>_inference/{5000,10000,19999}`에 저장됩니다.

## 검증 범위

전수 액션·상태 일치, 영상 디코딩/50Hz 타임스탬프, 에피소드 경계의 미래 액션 패딩,
정규화 재계산, 실제 학습 데이터 로더 검사를 수행합니다. 상세 결과는 이 폴더의 `audit.json`에 기록합니다.
원본은 수집 당시 성공한 50개 데모입니다. 이 데이터로 학습한 정책의 시뮬 성공률은 학습 후 평가해야 합니다.
