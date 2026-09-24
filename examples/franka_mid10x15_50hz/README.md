# mid10x15: 450개 성공 데모, 50Hz

큰 선반 중앙 10 × 15 cm 영역을 3×3 셀로 나눠 셀당 50개, 총 450개 성공 데모를 원본 주기 그대로 변환한 데이터입니다.
645,964프레임, 카메라 영상 1,350개(320×320, 50fps), 각 액션 간격 20ms입니다.
수집 레시피는 center5_v2와 같습니다(그리퍼 닫힘 대기 4 s, 파지 후 settle 상한 2 s, 성공 후 1 s 추가 녹화,
액션 실행 전 상태 정렬, 첫 프레임 재렌더). 마지막 프레임까지 보존하며 미래 액션은 각 에피소드 끝에서 마지막 액션으로 패딩합니다.
셀·라운드·캔 좌표는 `meta/scenes.jsonl`에 에피소드별로 기록되어 있습니다.
변환기: CVLAB-RLBench-PnP `isaaclab_arena_pnp_franka/Tools/datagen/suite_to_lerobot_50hz.py` (hdf5 → 50 Hz 데이터셋 단일 패스;
이전 두 단계 흐름 big100_to_lerobot + build_franka_center5_50hz 와 바이트 단위 동일 출력 확인).

- Hugging Face (비공개): https://huggingface.co/datasets/lithyeon/franka_pnp_mid10x15_50hz
- 학습 설정: `pi05_franka_pnp_mid10x15_50hz` (center5_v2_50hz와 동일 레시피, 데이터셋만 교체)
- 미래 액션: 50개 × 20ms = 1초
- norm stats: 전체 645,964프레임의 상태와 50개 미래 delta 액션으로 새로 계산
- 이미지 증강 ON, batch 32, 20,000스텝, pi05_base에서 새 학습
- 시뮬 평가: `control_stride=1`, `replan_every=25`, `expected_action_fps=50`

## 클러스터에서 받기

```bash
git clone --branch franka_pnp https://github.com/llishyun/openpi.git openpi-mid10x15-50hz
cd openpi-mid10x15-50hz
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
.venv/bin/python -c 'from huggingface_hub import login; login()'

export HF_LEROBOT_HOME="$HOME/.cache/huggingface/lerobot"
export OPENPI_FRANKA_CONFIG=pi05_franka_pnp_mid10x15_50hz
.venv/bin/python scripts/download_franka_center5_50hz.py      # 실험은 OPENPI_FRANKA_CONFIG로 선택
JAX_PLATFORMS=cpu .venv/bin/python scripts/prepare_franka_50hz.py
```

## 잡 제출

```bash
export CHECKPOINT_BASE_DIR="$HOME/openpi_runs"
export EXP_NAME=mid10x15_50hz_8g
export GPU_COUNT=8
sbatch scripts/slurm/franka_mid10x15_50hz.sbatch
```

기본값은 A100-80GB 8개, hpgpu, CPU 16개, RAM 160G, 24시간입니다. 4GPU는 `GPU_COUNT=4`와 `--gres=gpu:4`를 함께 지정합니다.
잡은 데이터 검증과 전체 모델·배치32의 2스텝 학습 검사를 통과해야 본 학습을 시작합니다.
시뮬용 가중치와 norm stats는 `<CHECKPOINT_BASE_DIR>/pi05_franka_pnp_mid10x15_50hz/<EXP_NAME>_inference/{5000,10000,19999}`에 저장됩니다.
