# SURF-0245-Multi-Agent-Social-Simulation-for-Public-Opinion-Governance
SURF  Project: LLM-powered multi-agent simulation system for social public opinion evolution and official governance strategy evaluation, built on Mesa &amp; Hawkes process.
## System Architecture (5-Module Independent Division for 5-Person Team)
1. OpinionModel (Module A, Project Lead): System scheduler, social network construction, global time-step loop, real-time public opinion metric collection
2. SocialAgent (Module B, Core Agent Layer): BDI four-layer belief reasoning, agent perception-memory-inference-action pipeline driven by LLM
3. HawkesEngine + LLM Toolchain (Module C, Algorithm Core): Hawkes event intensity sampling, LLM client encapsulation, prompt construction & response parsing
4. StrategyEvaluator (Module D, Intervention Control): Baseline simulation construction, multi-type official intervention injection, three-dimensional quantitative evaluation (behavior/content/network topology)
5. Utils (Module E, Data & Visualization): Unified experimental configuration, result export, sentiment & polarization trend visualization, text similarity calculation & standardized logging

## Core Technical Stack
Python | Mesa Multi-Agent Framework | Hawkes Point Process | LLM API (OpenAI/vLLM) | Matplotlib/Seaborn | NetworkX | YAML Config

## Research Value
1. Simulate interactive behaviors of multi-type social agents: ordinary netizens, opinion leaders, media, official departments
2. Quantify the impact of intervention timing, information release and emotional guidance on restraining negative public opinion polarization
3. Provide reproducible simulation experimental tools and quantitative analysis frameworks for public opinion governance research

## Development Iteration Standard (5-Sprint Agile Schedule)
S1: Build basic closed-loop simulation without LLM; S2: Single-agent LLM reasoning access; S3: Full LLM-driven agent group; S4: Official intervention strategy experiment; S5: Data analysis, visualization & final research report

## Repository Contents
- `/src`: Full modular source code of 5 core classes (OpinionModel, SocialAgent, HawkesEngine, StrategyEvaluator, Utils)
- `/docs`: Project research abstract, student learning brief, module division specification, function design document
- `/config`: YAML experimental parameter configuration files
- `/scripts`: Simulation running scripts, contrast experiment scripts
- `/output`: Exported CSV/JSON simulation data, visualized trend charts
- `/report`: Draft of academic research report, experimental analysis materials

## Pre-requisites for Contributors
Undergraduates majoring in CS/AI/Data Science; Basic Python programming; Familiarity with Mesa or rapid learning ability; Basic understanding of machine learning & public opinion communication.
All team members develop independently in their exclusive modules with standardized external interfaces to minimize cross-module communication costs.
