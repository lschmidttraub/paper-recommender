# Research interests

## Core interests

- LLM post-training: RLVR, on-policy knowledge distillation, self-distillation, preference optimization beyond DPO
- Test-time compute and inference-time adaptation
- AI safety + mechinterp
- Architectural innovations and performance optimizations

## Currently working on

- Synthetic data generation pipeline (GLM-5 + Qwen3 generators)
- Self-distillation for LLM post-training: on-policy student/teacher setups, relationship to RLVR, when distillation beats RL and vice versa

## Authors I follow

- Frontier labs: the three US labs as well as Deepseek and Kimi
- Neel Nanda, Evan Hubinger
- Jonas Hübotter, Andreas Krause

## Venues I care about

- NeurIPS, ICML, ICLR
- Also: arXiv preprints from frontier labs (Anthropic, DeepMind, OpenAI, Meta FAIR, Qwen, DeepSeek, Allen AI) — often more relevant than the venue version

## Keywords to boost

- RLVR, verifiable rewards, process reward models
- On-policy distillation, self-distillation, policy distillation
- Test-time compute, inference scaling, best-of-N, speculative reasoning
- HPC / systems for LLM training: Grace Hopper / GH200, SGLang, DeepGEMM, kernel-level optimizations

## Not interested in

- Prompt engineering papers with no training component
- LLM-as-judge eval papers that just propose a new benchmark without methodological novelty
- Pure theoretical analyses of transformers (expressivity, approximation theorems) with no empirical hook
- "We fine-tuned Llama on domain X" application papers unless the method generalizes
- Safety/alignment papers that are really just red-teaming demos without a technical contribution
- Retrieval-augmented generation systems papers unless they engage with training, not just plumbing
- Yet another DPO variant with marginal gains on AlpacaEval
- Multimodal papers focused on image/video generation quality (VLM reasoning is fine; diffusion-for-pixels is not)
