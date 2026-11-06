# Scoreboard

Same 30 Phase 1 test questions for every version; greedy decoding, max 64 new tokens.

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
|---|---|---|---|---|---|
| Small model, base | 9.83 | 3.3% | 0% | 0.69 | GPT-2 small (124M parameters) |
| + continued pretraining | 2.67 | 0.0% | 0% | 0.68 | 3 passes (best: pass 3), learning rate 5e-05 |
| + SFT | 2.83 | 20.0% | 0% | 0.20 | 766 training pairs + 85 validation pairs |

Hand-graded answers: 21; the two graders agreed at first on 19 of 21 (90%).
