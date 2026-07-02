# Claude-as-Judge Rubric

This is the operating manual for the manual judge (Claude in opus-4.7 role).
The trainer writes `judge/pending/batch_<id>.jsonl`; the judge worker
(`scripts/run_judge_worker.sh`) renders each entry and prompts for a JSON
result, which it writes to `judge/done/batch_<id>.jsonl`.

## Two judging modes

### 1. `grpo_reward` (during GRPO training)

Each pending entry contains:
- `question`        the user prompt
- `gold_answer`     the reference post-`</think>` answer
- `candidate_answer` the model's post-`</think>` answer
- `graph_json`      the model's emitted `<graph_json>` payload (may be empty)

Judge two scores, each on **[0.0, 1.0]** (continuous, full range):

#### `correctness` — weight 0.30
How well does the candidate answer match the gold answer on:
- Factual accuracy
- Completeness of key points
- Specificity (concrete detail vs vague)
- Errors (incorrect claims reduce score)

| Range | Meaning |
|-------|---------|
| 0.95-1.0 | Fully correct, complete, specific |
| 0.80-0.94 | Correct with minor omissions |
| 0.60-0.79 | Mostly correct, some gaps |
| 0.40-0.59 | Partially correct, significant gaps |
| 0.20-0.39 | Few correct elements |
| 0.01-0.19 | Mostly wrong or irrelevant |
| 0.0 | Completely wrong / no answer |

#### `graph_utility` — weight 0.25
The information-bottleneck test: using **only** the `graph_json` (no other context
from the response), can the question be answered?

| Range | Meaning |
|-------|---------|
| 0.90-1.0 | Graph captured all key concepts and relations |
| 0.70-0.89 | Most information present, minor gaps |
| 0.50-0.69 | Core idea present, missing details |
| 0.30-0.49 | Some relevant nodes, major gaps |
| 0.10-0.29 | Barely useful |
| 0.0-0.09 | Graph not useful / graph_json absent |

**Penalize decorative graphs**: a graph that "looks" rich but doesn't contain
the actual concepts needed to answer should score low. This is the paper's
central reward design (§4.1.3) and the experimentally identified bottleneck.

### 2. `eval_metric` (final benchmark, 100 questions)

Each pending entry contains:
- `question`         benchmark question (open-ended)
- `full_thinking`    the complete model output including `<think>...</think>` block
- `gold_answer`      optional reference
- `category`         reasoning category (one of 5 paper §4.2)
- `source_text_excerpt` relevant passage excerpt (for grounding)

Judge **three** metrics, each on **[0, 10]** (continuous):

#### `reasoning_quality` (paper Fig. 2)
Logical coherence, mechanistic correctness, absence of contradictions. Are the
intermediate steps actually right, not just plausible-sounding?

| Range | Anchor |
|-------|--------|
| 9-10 | Exceptional; publication-quality reasoning |
| 7-8.9 | Strong; minor errors only |
| 5-6.9 | Adequate; some correct, some weak |
| 3-4.9 | Weak; mostly superficial |
| 0-2.9 | Wrong or absent reasoning |

#### `intellectual_depth` (paper Fig. 2)
Cross-domain linkage, mechanistic depth, identification of hidden variables,
non-monotonic tradeoffs. Does it surface structure that's not in the prompt?

| Range | Anchor |
|-------|--------|
| 9-10 | Multi-scale, cross-domain, novel synthesis |
| 7-8.9 | Substantive depth, some cross-links |
| 5-6.9 | Single-domain depth only |
| 3-4.9 | Surface-level only |
| 0-2.9 | Trivial or off-topic |

#### `reasoning_traceability` (paper Fig. 2) — **the largest delta in the paper**
Can each claim in the synthesis/final answer be traced back to a node/edge in
the `<graph_json>`? Are the `<patterns>` actually read off the graph, or are
they free-floating? Does `<synthesis>` integrate the graph, or just restate text?

| Range | Anchor |
|-------|--------|
| 9-10 | Every claim traceable; patterns genuinely graph-derived |
| 7-8.9 | Most claims traceable; minor gaps |
| 5-6.9 | Mixed; some decorative graph use |
| 3-4.9 | Graph mostly decorative; reasoning in prose |
| 0-2.9 | No traceable structure; pure prose |

## Output schema

```json
[
  {
    "id": "<batch_id>|<idx>",
    "scores": {
      "correctness": 0.78,           // for grpo_reward, [0,1]
      "graph_utility": 0.42          // for grpo_reward, [0,1]
    },
    "justification": "One sentence per axis."
  },
  {
    "id": "...",
    "scores": {
      "reasoning_quality": 7.5,      // for eval_metric, [0,10]
      "intellectual_depth": 6.8,
      "reasoning_traceability": 8.2
    },
    "justification": "..."
  }
]
```

## Bias controls

The paper notes: "evaluation is performed using Claude opus-4.7 as an
independent judge" specifically because OpenAI models were used in dataset
generation. We're extending the same protocol — Claude-as-judge — but with
two explicit guardrails:

1. **Judge identities are blinded per entry**. The `model` field is in metadata
   but the response text itself is what's graded. Do not infer the model from
   style; score only on the rubric.
2. **Use the full range**. Defaulting to 0.5 or 5.0 collapses the signal. If
   a response is mediocre, give it 0.4 / 4.0, not 0.5 / 5.0.

## Throughput expectations

- A `grpo_reward` batch is typically 16 entries. Plan ~30 seconds per entry,
  ~8 minutes per batch.
- An `eval_metric` batch is up to 100 entries (one model). Plan ~2 minutes
  per entry (the thinking is longer), ~3 hours per model.

The trainer never blocks on judge completion: it uses embedding reward as
primary and blends Claude scores when they arrive. So judges can be batched
at whatever cadence is convenient.
