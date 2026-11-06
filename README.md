# PlanPath: Course-Advising Assistant for JUST CIS Students

Building LLM Systems: From Models to Production, team project.
Phase 2 trains GPT-2 small (124M parameters) on the JUST B.Sc. CIS study plan in two stages (continued pretraining, then supervised fine-tuning) and scores the base, continued-pretrained and fine-tuned models on the 30 Phase 1 test questions.

Data source: JUST B.Sc. CIS study plan no. 766, https://www.just.edu.jo/FacultiesandDepartments/it/Departments/cis/Pages/viewplan.aspx?planno=766

## Setup

Use Google Colab with a T4 GPU (Runtime > Change runtime type > T4 GPU). Clone this repository, then:

```
pip install transformers accelerate pyyaml pandas matplotlib pypdf beautifulsoup4 lxml requests
export CHECKPOINTS_ROOT=/content/drive/MyDrive/PlanPath/checkpoints
```

`CHECKPOINTS_ROOT` makes the scripts save checkpoints to Google Drive (mount Drive first).

All settings (model, paths, learning rates, epochs, block size, seed, generation length) are in `config.yaml`. The random seed is 42.

## How to train

```
python src/fetch_course_docs.py   # optional: syllabi and course descriptions from services.just.edu.jo
python src/prepare_cpt.py         # corpus -> data/cpt/train.jsonl and heldout.jsonl (90/10 split by document)
python src/make_sft_pairs.py      # SFT pairs -> data/sft_pairs.jsonl (no test-set courses)
python train_cpt.py               # continued pretraining
python train_sft.py               # supervised fine-tuning
```

Run `python train_cpt.py --fraction 0.05 --epochs 1` and `python train_sft.py --fraction 0.05 --epochs 1` first to test the pipeline on 5% of the data.

Continued pretraining uses 512-token blocks, learning rate 5e-5 and up to 3 passes, saves a checkpoint after each pass, and stops when validation loss stops falling. The pass with the lowest validation loss becomes the final model.

Supervised fine-tuning uses the chat template `<|user|>\n{question}\n<|assistant|>\n{answer}<|end|>`. The three markers are added as special tokens, and the loss is computed on answer tokens only.

## How to score

```
python score.py              # runs base, cpt and sft on the 30 test questions
python score.py --finalize   # after hand grading: final scores and SCOREBOARD.md
```

`score.py` calls all three models through the same function (`src/models.py: generate_answer`) with greedy decoding and the same maximum answer length. It writes one results file per model (`results/results_base.csv`, `results_cpt.csv`, `results_sft.csv`) with the question, expected answer, model answer, score and a note.

Scoring: questions with a short answer (a number, a course code, yes/no, a requirement group) are scored automatically by exact match after normalizing case and spaces; the rule for each question is in `src/scoring.py` (`ANSWER_KEYS`). The 7 longer answers are graded by hand by two team members in `results/grading_sheet.csv` (1, 0.5 or 0). Disagreements are settled in the `agreed` column, and `--finalize` reports how often the graders agreed at first. Refusal accuracy is the share of the 3 should-refuse questions (28-30) that the model declines.

## Where the checkpoints are

Checkpoints are not in Git. They are saved to Google Drive:

- Continued-pretrained model: `MyDrive/PlanPath/checkpoints/cpt/final`
- Fine-tuned model: `MyDrive/PlanPath/checkpoints/sft/final`
- Base model: `gpt2` from the Hugging Face Hub

## Repository

```
config.yaml              model names, paths and training settings
train_cpt.py             continued pretraining
train_sft.py             supervised fine-tuning
score.py                 runs the three models on the test questions and scores them
src/                     model loading, chat template, data preparation, scoring rules
data/cpt/                study plan source tables and cleaned continued-pretraining text (train and held-out)
data/sft_pairs.jsonl     question-answer pairs for fine-tuning
tests/test_questions.csv the 30 Phase 1 test questions
results/                 loss curves, perplexity, completions, one results file per model,
                         SFT overlap report and the 50 hand-checked SFT pairs
SCOREBOARD.md            one row per system version
```

No API keys are needed in this phase. Variable names are listed in `.env.example`.
