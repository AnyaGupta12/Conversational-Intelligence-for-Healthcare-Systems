# Capstone Project - Conversational Intelligence for Healthcare Systems

Submitted By : Anya Gupta ( CS24MTECH11020 )

The core workflow lives in `capstone-file.ipynb`.

## Repository contents

### Main notebook
- `capstone-file.ipynb` - end-to-end implementation: model loading, prompts, dataset ingestion, slot registry + question bank, simulation loop, metrics, baseline comparison, and CSV export.

### Dataset (MEDIQA-SYNUR)
Folder: `mediqa_synur_dataset/mediqa_synur_dataset/`

- `train.jsonl`, `dev.jsonl`, `test.jsonl` - line-delimited JSON records.
  - `transcript`: nurse dictation text
  - `observations`: structured observation list (id/name/value_type/value)
- `synur_schema.json` - ontology of observation concepts (~193 slots)
- `SYNUR_testset_input.jsonl` - test input file (if present)
- `README` - dataset card

### Derived / cached artifacts
- `converted_scenarios.json` - JSONL records converted into patient simulation scenarios (`name`, `opening_message`, `symptom_profile`) and linked back to structured observations.
- `slot_question_bank.json` - mapping of slot name → question template + keywords (used to drive doctor questions and intent recognition).
- `disease_slot_kb.json` - disease/condition - list of high-priority slot IDs mined from training data.
- `conversations_run9.csv` - exported conversations and metrics from a full run.

## How the pipeline works (high level)

### Agents
1. **Patient agent** (LLM): role-plays a scenario, optionally grounded to a hidden structured record (`slot_record`).
2. **Doctor agent** (LLM): follows strict rules (one question/turn) and uses slot hints to query missing observations.
3. **State Manager** (LLM): reads conversation history and outputs **strict JSON** containing persona, gathered symptoms, diagnosis hypotheses and confidence.

### Slot-filling strategy
- The observation schema (`synur_schema.json`) becomes a **slot registry**.
- A generated **slot question bank** provides canonical question templates.
- A dataset-mined **disease-slot knowledge base** provides global, non-patient-specific guidance for which slots are important per disease hypothesis.

### Evaluation
The notebook computes metrics per simulation, including:

- Diagnostic accuracy (with robust matching)
- Symptom coverage
- Conciseness (doctor words/turn)
- Diagnostic velocity
- Repetition rate
- Slot recall / slot precision (when slot grounding is enabled)

## Requirements

The notebook installs dependencies inline, typically:

- `transformers`, `accelerate`, `bitsandbytes`, `torch`
- `langchain`, `langchain_community`
- `pandas`, `numpy`

Note: Running the Llama 8B models generally requires a GPU.

## Running the project

Open and run `capstone-file.ipynb` top-to-bottom.

Key configuration parameters are near the bottom of the notebook (paths, number of records converted, turns per simulation, top-k slots, etc.).


## Notes / caveats

- Update your Hugging Face Token in the HF_Token variable for loading the necessary models
- Paths in the notebook include Kaggle-style defaults (e.g. `/kaggle/...`). If running locally, update the dataset paths to your local files.

## License

The MEDIQA-SYNUR dataset inside `mediqa_synur_dataset/` is distributed under its own license (see the dataset `README`).
