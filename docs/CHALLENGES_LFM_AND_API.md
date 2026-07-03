# 도전 과제: LFM2.5 백본과 외부 LLM API 의존성

이 문서는 Graph-PRefLexOR 재현 포크(`graph-preflexor-lfm25`)를 진행하면서 겪은 두 가지 주요 애로사항을 정리합니다:

1. **LFM2.5-8B-A1B 백본 사용 시의 기술적 장벽**
2. **원본 논문이 가정하는 GPT / Grok / Claude API 의존 구조와 그 회피 방안**

논문(arXiv 2607.00924v1)은 Qwen3-8B + 상용 LLM API 조합으로 되어 있고, 이 포크는 Qwen3-8B/Qwen3.5-9B/gemma-4 + LFM2.5-8B-A1B + API-free 보상으로 확장합니다. 두 축 모두 단순한 코드 수정이 아니라 근본적인 트레이드오프를 동반합니다.

---

## 1. LFM2.5-8B-A1B 백본 사용이 어려운 이유

LFM2.5-8B-A1B(Liquid Foundation Models, A1B = 1B active params MoE)는 논문이 다루지 않는 확장 축이며, 다음 여섯 가지 이유로 재현 비용이 크게 높아집니다.

### 1.1 비전식 아키텍처 — vLLM 지원 범위 좁음

LFM2.5 계열은 `Lfm2ForCausalLM`(dense) 과 `Lfm2MoeForCausalLM`(MoE) 두 가지를 vLLM 0.19 ~ 0.20.2 가 등록하고 있습니다. 하지만:

- MTP(Multi-Token Prediction) 헤더가 결합된 형태는 vLLM이 weights 로딩 시 `mtp.*` prefix를 기대하는데, HF Hub의 체크포인트는 prefix가 있을 때도 없을 때도 있어 로딩이 깨지는 경우가 잦습니다.
- `tensor_parallel_size > 1` 일 때 MoE expert 분배가 어색하게 동작하여, H200 × 8 설정에서 TP=4를 시도하면 일부 expert가 비어있는 것처럼 에러가 납니다.
- LFM2.5 전용 attention backend(`gdn_prefill_backend`)가 Triton 커널을 필요로 하지만, 이 커널은 cu128 nightly torch에서 빌드가 깨져있는 경우가 많습니다.

실제로 우리 환경에서는 `--tensor-parallel-size 1 --enforce-eager` 로 단일 GPU 서빙만 안정적으로 동작했습니다. 4-replica rollout 서버 구성이 어려워 GRPO 처리량이 Qwen3 계열 대비 절반 이하로 떨어집니다.

### 1.2 Hybrid attention (linear + full) — 동일 이슈가 LFM에도 존재

LFM2.5는 linear attention 기반 모델이라는 점에서 Qwen3.5 text-only와 비슷한 문제를 공유합니다. flash-linear-attention 커널이 없으면 SDPA fallback 이 빈 토큰을 반환하고, transformers 5.x 최신 런타임이 필요합니다. 우리 환경의 `.vllm-lfm-cu12` venv(torch 2.12.0.dev20260407+cu128, vllm 0.19.1)은 transformers 4.x 기반이라 LFM2.5 최신 모델 코드가 호환되지 않아, 별도의 strict env(`.vllm-eval-cu129-strict`, transformers 5.5.4)를 쓰는 쪽으로 수렴했습니다.

### 1.3 Chat template / tokenizer 분기

Qwen3 계열은 `apply_chat_template(enable_thinking=True/False)` 인터페이스로 thinking 모드를 토글합니다. LFM2.5는 자체 템플릿(`<|user|>...<|end_of_text|>\n<|assistant|>`)을 쓰며 thinking 토글 개념이 없습니다. 따라서:

- 같은 `apply_chat` 헬퍼(`src/grpo_train.py`)를 쓰면 LFM2.5에서 `enable_thinking` kwarg가 무시되어 thinking 모드 프롬프트가 깨집니다.
- 5-sentinel 포맷(`<brainstorm>...<graph>...<graph_json>...<patterns>...<synthesis>`)을 inside-`<think>` 로 넣는 논문의 설계가 LFM2.5 템플릿에서는 자연스럽지 않습니다. LFM은 `<think>` 태그를 기본 템플릿에 포함하지 않기 때문에, ORPO 데이터 준비 단계에서 chosen/rejected 쌍을 만들 때 템플릿을 강제로 덮어써야 했습니다.

### 1.4 ORPO 콜드스타트 단계의 VRAM 폭발

LFM2.5-8B-A1B는 총 파라미터 8B(활성 1B)이지만 ORPOTrainer의 `concatenated_forward` 는 chosen·rejected를 한 배치에 묶어 순전파합니다. 활성 1B라도 순전파 메모리는 총 파라미터 기반이라:

- Qwen3-8B ORPO는 H200 80GB 1장에 batch=1, seq=4096 로 들어갑니다.
- LFM2.5-8B-A1B ORPO는 동일 설정에서 OOM이 발생하여, batch=1, seq=3072, grad_accum=4로 낮춰야 했습니다(`configs/orpo_lfm25_8b_v2.env`).
- ORPO Trainer가 마지막에 강제로 실행하는 `evaluate()` 단계는 VRAM이 부족해 항상 죽고, 직전에 저장된 checkpoint(예: `checkpoint-250`)를 병합에 써야 합니다.

### 1.5 GRPO rollout 서버와 학습 트레이너의 GPU 분할이 강제

H200 × 8 환경이라도 LFM2.5는 rollout(vLLM) 4장 + 트레이너 4장 분리가 강제됩니다. colocate 모드(`vllm_gpu_memory_utilization=0.35`)로 같은 GPU에 올리면, MoE expert lazy 로딩과 LoRA optimizer state가 충돌하여 NCCL timeout이 빈번합니다. 우리 GPU 할당량이 4장(2,3,4,5)으로 제한된 환경에서는 LFM2.5 그래프-GRPO를 서버 모드로 실행할 수 없고, 결국 LFM2.5는 ORPO 콜드스타트까지만 완료한 상태로 머뭅니다.

### 1.6 학습 데이터 / 벤치마크의 동질성 가정

논문의 `lamm-mit/graph_reasoning_10K` 데이터는 Qwen3 tokenizer 기준으로 길이 분포가 정규화되어 있습니다. LFM2.5 tokenizer(BPE 계열이지만 어휘 약 110K)로 동일 텍스트를 인코딩하면 시퀀스 길이가 평균 1.2 ~ 1.4× 길어져 `max_prompt_length=1536` 과 `max_completion_length=3500` 설정이 의도치 않게 truncation을 일으킵니다. 이 효과가 sentinel 중 `graph_json` 히트율이 79%에 머무는 결과로 직접 이어집니다.

---

## 2. 외부 LLM API 가 필요한 부분들 — 그 이유와 대체 설계

논문의 보상·평가 파이프라인은 3군데서 상용 LLM API에 강하게 의존합니다. 이 포크는 이를 모두 API 없이 동작하도록 바꿨지만, 각 결정마다 비용이 있습니다.

### 2.1 GRPO 보상의 `correctness` (가중치 0.30)

**논문(original):** OpenAI `gpt-5-mini` 가 candidate answer 와 gold answer 를 비교해 0~1 점을 반환.

**왜 필요한가:** 그래프 추론 벤치마크는 정답이 단일 숫자나 단일 문자열이 아니라 "자연어로 쓰인 가설/설명" 입니다. exact-match, F1, BLEU, ROUGE 같은 토큰 기반 메트릭은 동의어·순어·표현 차이에 0점을 주기 때문에, 의미적 동등성을 평가하려면 LLM judge 가 필요합니다.

**이 포크의 대체:** `BAAI/bge-base-en-v1.5` 로 candidate 와 gold 를 임베딩해서 cosine similarity → rescale [0,1].

**비용/한계:**
- BGE는 동의어·순어에 강하지만, "맞는 답인데 근거가 틀린" 경우를 잡지 못합니다. 즉 정답의 "사실성" 은 평가하지 못하고 "표면적 의미 유사도" 만 평가합니다.
- 논문 judge 대비 reward signal 이 둔해져서, GRPO 가 `correctness` 를 올리기보다 `format` 처럼 쉬운 component 로 maximize 하려는 경향이 생깁니다. 이를 보정하기 위해 `correctness` 와 `graph_utility` 가 가중치 0.30 + 0.25 = 0.55 로 전체의 과반을 차지하게 설계를 유지했지만, 실제 학습 곡선은 논문보다 완만합니다.

### 2.2 GRPO 보상의 `graph_utility` (가중치 0.25)

**논문:** OpenAI judge 가 2-call 로 동작. 첫 호출은 `graph_json` 만 보고 답변을 재구성하고, 두 번째 호출은 그 답변과 gold 를 비교. information-bottleneck 평가(§4.1.3)를 위해 이 2-call 구조가 필수.

**왜 필요한가:** 그래프 자체가 답을 얼마나 잘 인코딩하고 있는지를 측정하려면, "그래프만 보고 답을 맞출 수 있는가?" 라는 질문이 필요합니다. 이는 임베딩 유사도 한 번으로는 측정 불가합니다.

**이 포크의 대체:** `graph_json` 을 자연어 텍스트로 렌더링(`nodes: [...], edges: [...]` 형식)한 후, 이 텍스트와 gold answer 의 BGE 코사인을 계산. 즉 "그래프가 답을 얼마나 잘 담고 있는가" 를 "그래프를 텍스트로 풀었을 때 gold 와 얼마나 비슷한가" 로 근사.

**비용/한계:**
- information-bottleneck 평가의 엄밀성이 사라집니다. 그래프가 답을 "직접 포함" 하고 있어도 높은 점수가 나옵니다(예: node 이름에 정답 문자열 그대로).
- 논문 judge 의 2-call 구조가 가진 "추론 단계" 가 빠지기 때문에, 그래프의 구조적 품질보다 텍스트 표면의 키워드 오버랩에 점수가 좌우됩니다.
- 그래도 보상 신호로서는 유효해서, sentinel 히트율이 brainstorm 99% / graph 94% / graph_json 79% / patterns 85% / synthesis 84% 까지 올라간 걸로 확인됩니다.

### 2.3 벤치마크 평가의 3-metric judge

**논문:** 100개 질문 × 3 metrics(`reasoning_quality`, `intellectual_depth`, `reasoning_traceability`) × [0, 10] 점을 Claude opus-4.7 API 가 반환.

**왜 필요한가:** 위 3 지표는 객관식/수치로 환원되지 않는 "추론의 질" 이라서 토큰 메트릭이나 임베딩 메트릭으로 대체 불가합니다. 예를 들어 `reasoning_traceability`는 "그래프가 최종 답까지의 사고 흐름과 얼마나 일관되게 연결되어 있는가" 이고, 이를 자동화하려면 결국 LLM judge 가 필요합니다.

**이 포크의 대체:** Claude opus-4.7 API 대신 **Claude-as-judge 파일 큐**(`judge/pending/` → `judge/done/`)를 도입.
- 트레이너는 매 N step 마다 샘플을 `judge/pending/batch_*.jsonl` 에 씁니다.
- 사용자가 원할 때 `scripts/run_judge_worker.sh` 를 실행해서 pending batch 를 Claude Code 세션 안에서 렌더링·채점하고, 결과를 `judge/done/<same_name>.jsonl` 로 검증해서 씁니다.
- 트레이너는 다음 step 시작 시 `done/` 을 비동기로 drain 하고, `claude_blend_alpha=0.5` 로 embedding 보상과 블렌드합니다.

**비용/한계:**
- "비동기"가 핵심입니다. 동기 API 호출은 GRPO step time = forward + backward + judge API latency(수 초~수십 초)가 되어 GPU 활용률이 떨어지고, rate limit 에 step 가 밀립니다. 파일 큐는 트레이너는 embedding 보상으로 학습을 계속하고, judge 점수는 다음번 평가에 반영되는 식입니다.
- 단점은 judge 가 언제 도착할지 예측할 수 없다는 것입니다. warm-up 단계에서는 judge 점수가 없는 상태로 보상이 들어가고, alpha blend 가 의미를 갖게 되는 것은 수십 step 이후부터입니다.
- 사용자가 직접 judge worker 를 돌려야 하므로 완전 자동은 아닙니다. 이는 비용(모델 추론 비용)을 사용자의 시간으로 옮기는 트레이드오프입니다.

### 2.4 API 의존이 가져오는 근본 문제들 (논문을 그대로 따랐을 때의 비용)

논문 설정을 그대로 썼다면 발생했을 문제들입니다. 이것이 이 포크가 API-free 를 선택한 2차 이유입니다.

- **비용:** 10000 GRPO step × 8 generations × 2 judge call(correctness, graph_utility) = 16만 API 호출. `gpt-5-mini` 기준 단일 호출 약 \$0.02 가정 시 하루 \$3000+.
- **지연:** 동기 API 호출이 GRPO step time 에 더해져 GPU 활용률이 30% 이하로 추락. H200 × 8 의 시간당 비용이 낭비.
- **재현성:** API 모델은 버전이 바뀌고 프롬프트 민감도가 다릅니다. 논문 재현 시 judge 가 바뀌면 reward 분포가 달라져서 reward shaping 을 다시 해야 합니다.
- **Rate limit:** OpenAI / xAI 의 분당 호출 제한이 GRPO 처리량의 상한이 됩니다. 4 replica vLLM 으로 초당 32 completion 을 만들어도 API 가 초당 4~8 회로 제한하면 4× 병목.

이 포크의 설계는 이 네 가지를 모두 "judge 를 오프라인 파일 큐로 빼고, primary 신호는 로컬 임베딩으로" 라는 한 방으로 회피합니다. 대신 2.1~2.3 의 정확도 손실을 감수합니다.

---

## 3. 현재 상태 요약 (2026-07-03 기준)

| 백본 | ORPO | GRPO | 평가 | 비고 |
|------|------|------|------|------|
| Qwen3-8B (논문 통제군) | 완료 | 설계만 | 미실행 | vLLM server 모드로 4-replica rollout 구성까지는 동작 확인 |
| Qwen3.5-9B (text-only) | 완료 (loss 0.98, eval 6.13/10) | **미실행** | transformers 4-GPU 샤드평가 완료 | vLLM 이 `Qwen3_5TextForCausalLM` 미지원 → GRPO_NO_VLLM=1 로 fallback 필요 |
| gemma-4-E2B-it | 미실행 | 미실행 | 미실행 | config 만 준비됨 |
| gemma-4-E4B-it | 미실행 | 미실행 | 미실행 | config 만 준비됨 |
| LFM2.5-8B-A1B | 1차 시도 OOM, v2 진행중 | 미실행 | 미실행 | 위 1.1~1.6 이유로 비용 높음 |

재현의 첫 단계(Qwen3-8B 그대로)를 건너뛰고 Qwen3.5 + LFM2.5 + gemma-4 로 확장부터 시도한 것이, 원래 연구 목적상 합리적이었지만 위 정리한 장벽들 때문에 예상보다 진행이 느려진 것이 사실입니다. 이 문서를 명시적으로 남겨서, 동일한 시도를 하는 다음 작업이 같은 곳에서 막히지 않도록 합니다.
