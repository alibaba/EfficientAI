# Repository Overview

This codebase provides:

## 1. Lazy Reasoning Statistics
Complete implementation of the Lazy Reasoning computational framework from our paper. Execute the provided scripts to obtain identical statistical results as reported in the publication.

python cal_bfcl_lazy_reasoning.py

Qwen3-8B
Para Lazy Reasoning Ratio: 3.7%
Irr Lazy Reasoning Ratio: 9.1%
Multi-Turn Lazy Reasoning Ratio: 45.1%

D-CORE-14B
Para Lazy Reasoning Ratio: 5.1%
Irr Lazy Reasoning Ratio: 5.5%
Multi-Turn Lazy Reasoning Ratio: 6.8%

## 2. Multi-turn Tool Use Task Decomposition
task_decompose_sample.json includes results from Algorithm 1:
- Task decomposition for multi-turn tool use queries (key: `sub_tasks`)
- Step-by-step reasoning for each subtask (key: `reasoning_steps`)

python cal_dcore_success_rate.py

Valid Sample Number: 330
Total Sample Number: 363
Success Rate: 0.9090909090909091