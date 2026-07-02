# lfm-graph-preflexor

**포크 안내 (Fork Notice)**
이 저장소는 [`lamm-mit/graph-preflexor-grpo`](https://github.com/lamm-mit/graph-preflexor-grpo) 의 포크이며, 원본 논문
[Graph-Native Reinforcement Learning Enables Traceable Scientific Hypothesis Generation through Conceptual Recombination (arXiv 2607.00924v1)](https://arxiv.org/abs/2607.00924v1)
의 재현 + 확장 프로젝트입니다. 원본 코드는 `../graph-preflexor-grpo/` 에 그대로 보존되어 있으며, 이 디렉토리는 원본의 수정된 파생본입니다.

> This repository is a fork of [`lamm-mit/graph-preflexor-grpo`](https://github.com/lamm-mit/graph-preflexor-grpo).
> It reproduces the paper above and extends it with two backbones (Qwen3-8B control + LFM2.5-8B-A1B extension),
> an H200 × 8 distributed setup, and an API-free reward pipeline that replaces `gpt-5-mini`/`grok-4` judging
> with a Claude-as-judge file queue + embedding-similarity fallback.

---

## 확장 내용 (What this fork changes vs upstream)

| 항목 | 원본 (upstream) | 이 포크 (this fork) |
|------|-----------------|---------------------|
| 하드웨어 | A100 80GB × 1 | H200 × 8 (4 train DDP + 4 vLLM rollout) |
| 보상 판정 (correctness, graph_utility) | OpenAI / xAI API (`gpt-5-mini`, `grok-4`) | **API-free**: BGE-base-en-v1.5 코사인 유사도 (primary) + Claude opus-4.7 역할의 파일 큐 judge (calibration) |
| 학습 차단 | API 호출이 병목 | **Non-blocking**: embedding reward로 학습 지속, judge는 오프라인 batch |
| 백본 | Qwen3-8B (paper) | Qwen3-8B (control, paper 재현) + **LFM2.5-8B-A1B** (확장) |
| 평가 | Claude opus-4.7 API judge | Claude opus-4.7 역할의 수동 judge 큐 (`scripts/run_judge_worker.sh`) |
| 데이터 준비 | 노트북 기반 | `scripts/01_download_data.py`, `02_prepare_data.py` |
| 체크포인트 업로드 | 수동 | `scripts/06_upload_to_hub.py` → HuggingFace `LLM-OS-Models` org |

원본의 5-sentinel 추론 포맷(`<brainstorm>→<graph>→<graph_json>→<patterns>→<synthesis>`)과 6-component 보상 설계(`correctness` 0.30 / `format` 0.15 / `graph_utility` 0.25 / `graph_networkx` 0.10 / `graph_diversity` 0.10 / `graph_structure` 0.10)는 그대로 보존합니다.

---

## 디렉토리 구조 (Layout)

```
graph-preflexor-lfm25/
├── configs/                       per-model env files
│   ├── orpo_qwen3_8b.env          ORPO cold-start (paper reproduction)
│   ├── grpo_qwen3_8b.env          Graph-GRPO (paper reproduction)
│   ├── orpo_lfm25_8b.env          ORPO cold-start (LFM2.5 extension)
│   └── grpo_lfm25_8b.env          Graph-GRPO (LFM2.5 extension)
├── data/
│   ├── raw/                       downloaded from HF (gitignored)
│   │   ├── train_10K.jsonl        lamm-mit/graph_reasoning_10K
│   │   └── benchmark_100.jsonl    lamm-mit/graph-preflexor-grpo-benchmark
│   └── processed/                 prepared splits (gitignored)
│       ├── orpo.jsonl             10000 rows: prompt / chosen / rejected
│       ├── grpo.jsonl             10000 rows: prompt / answer / question
│       └── eval.jsonl             100 rows:   question / category / doi / title
├── src/
│   ├── programmatic_rewards.py    format, networkx, diversity, structure (ported)
│   ├── embedding_rewards.py       BGE cosine sim → [0,1] (replaces API judges)
│   ├── combined_reward.py         6-component weighted + Claude blend
│   ├── judge_queue.py             atomic file queue pending → done → archive
│   ├── orpo_train.py              torchrun DDP ORPO (TRL)
│   └── grpo_train.py              torchrun DDP GRPO (TRL) + vLLM rollout server
├── scripts/
│   ├── 01_download_data.py        ✓ run once
│   ├── 02_prepare_data.py         ✓ run once
│   ├── 03_run_orpo.sh             cold start (4 GPUs)
│   ├── 04_run_grpo.sh             RL refinement (4 train + 4 vLLM)
│   ├── 05_eval_benchmark.py       100-q benchmark: --generate then --collect
│   ├── 06_upload_to_hub.py        push to HuggingFace LLM-OS-Models org
│   ├── run_vllm_replicas.sh       launch 4 vLLM api_servers (rollout target)
│   ├── stop_vllm_replicas.sh
│   ├── judge_worker.py            renders pending, reads JSON, writes done
│   └── run_judge_worker.sh        bash wrapper for the above
├── judge/
│   ├── README.md                  rubric for both judging modes
│   ├── pending/                   trainer → Claude (gitignored)
│   ├── done/                      Claude → trainer (gitignored)
│   ├── archive/                   consumed batches (gitignored)
│   └── templates/
│       ├── grpo_reward_template.json
│       └── eval_metric_template.json
├── checkpoints/                   output adapters (gitignored)
├── logs/                          training + judge logs (gitignored)
└── README.md                      this file
```

---

## 재현 절차 (Reproduction recipe)

### 1. 환경 (Requirements)

- Python 3.10+
- 8× H200 80GB (또는 동급 VRAM; A100 80GB × 8도 가능)
- PyTorch 2.x, CUDA 12.x
- `pip install torch transformers trl peft accelerate vllm sentence-transformers datasets huggingface_hub networkx`

`.env` 파일에 `HF_TOKEN` 이 있어야 함 (허깅페이스 업로드 시 사용). 이 프로젝트의 런처들은 상대경로 `../../.env` 에서 토큰을 읽습니다.

### 2. 데이터 준비 (Data — already done in this repo's state)

```bash
python scripts/01_download_data.py     # → data/raw/{train_10K,benchmark_100}.jsonl
python scripts/02_prepare_data.py      # → data/processed/{orpo,grpo,eval}.jsonl
```

현재 상태: 10000 + 10000 + 100 rows 준비 완료.

### 3. ORPO 콜드 스타트 (Stage 1 — SFT-style warmup)

```bash
bash scripts/03_run_orpo.sh configs/orpo_qwen3_8b.env    # 논문 재현 (control)
bash scripts/03_run_orpo.sh configs/orpo_lfm25_8b.env    # LFM2.5 확장
```

입력: `data/processed/orpo.jsonl` → 출력: `checkpoints/orpo_<model>/`

### 4. Graph-GRPO (Stage 2 — RL refinement)

```bash
# 4-1. vLLM rollout 서버들을 GPU 0-3에 실행 (ports 8123-8126)
bash scripts/run_vllm_replicas.sh configs/grpo_qwen3_8b.env

# 4-2. 트레이너는 GPU 4-7에서 DDP로 실행
bash scripts/04_run_grpo.sh configs/grpo_qwen3_8b.env
```

입력: `data/processed/grpo.jsonl` + ORPO merged 체크포인트 → 출력: `checkpoints/grpo_<model>/`

이 단계에서 트레이너는 매 `JUDGE_QUEUE_EVERY_STEPS` (기본 50 step) 마다
`judge/pending/batch_<timestamp>.jsonl` 에 completion 샘플을 씁니다.
트레이너는 embedding 보상으로 계속 학습하며, judge 완료를 블로킹하지 않습니다.

### 5. Claude-as-Judge 호출 (오프라인, 원할 때)

```bash
bash scripts/run_judge_worker.sh                 # 모든 pending batch 처리
bash scripts/run_judge_worker.sh --list-only     # 큐 깊이만 확인
bash scripts/run_judge_worker.sh --batch 20260702T...   # 특정 batch만
```

worker는 각 entry를 렌더링해서 JSON 템플릿을 출력하고, 결과를 stdin으로 받아
`judge/done/<same_name>.jsonl` 로 검증하여 씁니다. 다음 GRPO step에서 트레이너가
`done/` 을 drain하여 보상에 blend(`claude_blend_alpha=0.5`).

### 6. 100-q 벤치마크 평가

```bash
# 6-1. 모델 추론 (pending eval_metric entries 생성)
python scripts/05_eval_benchmark.py --generate checkpoints/grpo_qwen3_8b/merged

# 6-2. Claude judge (3 metrics × 100 questions × N models)
bash scripts/run_judge_worker.sh

# 6-3. 결과 수집 + 평균
python scripts/05_eval_benchmark.py --collect
```

3 metrics (`reasoning_quality`, `intellectual_depth`, `reasoning_traceability`) 각각 [0, 10].
rubric은 `judge/README.md` 참조.

### 7. HuggingFace 업로드

```bash
python scripts/06_upload_to_hub.py checkpoints/grpo_qwen3_8b/merged
```

- 어댑터 (LoRA 병합된 full model) → `LLM-OS-Models/graph-preflexor-qwen3-8b`
- eval JSON (sibling dataset) → `LLM-OS-Models/graph-preflexor-qwen3-8b-eval`

---

## 연구 질문 (Research questions)

1. **재현성**: 논문의 Qwen3-8B 결과(reasoning_traceability ≥ 7)를 같은 데이터/같은 보상 설계로 재현할 수 있는가?
2. **아키텍처 일반화**: LFM2.5-8B-A1B (A1B MoE) 가 같은 그래프-네이티브 RL 파이프라인에서 Qwen3-8B (8B dense) 와 비교해 어디까지 성능을 낼 수 있는가?
3. **API-free 판정**: Claude opus-4.7 역할의 수동 파일 큐 judge 가 embedding 유사도 보상과 함께 학습 신호로 작동하는가? 보정 차이(calibration diff)는 얼마인가?

---

## 보상 설계 (Reward design)

6-component weighted 보상 (논문 Eq. 4 그대로):

| Component | Weight | Source |
|-----------|--------|--------|
| `correctness` | 0.30 | BGE cosine(candidate, gold) → rescale [0,1] (원본: OpenAI judge) |
| `format` | 0.15 | 5-sentinel 태그 + graph_json 파싱 가능 여부 (원본 그대로) |
| `graph_utility` | 0.25 | BGE cosine(graph_text_render, gold) → rescale [0,1] (원본: OpenAI 2-call judge) |
| `graph_networkx` | 0.10 | NetworkX 그래프가 유효한지 (원본 그대로) |
| `graph_diversity` | 0.10 | MiniLM 임베딩 기반 노드 다양도 (원본 그대로) |
| `graph_structure` | 0.10 | 그래프 밀도/연결성/커버리지 (원본 그대로) |

`graph_utility` 의 **information-bottleneck 설계** (논문 §4.1.3) 가 핵심입니다:
judge는 `graph_json` 만 보고 답을 재구성할 수 있어야 합니다. 이 포크에서는
`graph_json` 을 텍스트로 렌더링한 후 gold answer 와의 임베딩 코사인 유사도로 대체합니다.

Claude judge 가 `done/` 에 도착하면, `claude_blend_alpha=0.5` 로 `correctness`/`graph_utility` 에 블렌드됩니다:

```
final_score = (1 - alpha) * embedding_score + alpha * claude_score
```

---

## 라이선스 (License)

원본 `lamm-mit/graph-preflexor-grpo` 의 라이선스(Apache 2.0)를 따릅니다.
이 포크에서 추가된 코드는 같은 라이선스로 배포됩니다.

## 인용 (Citation)

```bibtex
@article{lamm2026graphpreplexor,
  title={Graph-Native Reinforcement Learning Enables Traceable Scientific Hypothesis Generation through Conceptual Recombination},
  author={Lamm, Ariel T. and others},
  journal={arXiv preprint arXiv:2607.00924v1},
  year={2026}
}
```

원본 코드: <https://github.com/lamm-mit/graph-preflexor-grpo>
원본 논문: <https://arxiv.org/abs/2607.00924v1>
