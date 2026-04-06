---
title: Adaptive API Rate Limit Optimization Environment
emoji: "🚦"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Adaptive API Rate Limit Optimization Environment

## 🌐 Live Deployment

- Space: https://huggingface.co/spaces/akashakuthota/api-rate-limit-env
- API Base URL: https://akashakuthota-api-rate-limit-env.hf.space

## ⚡ Quick Test

```bash
curl -X POST https://akashakuthota-api-rate-limit-env.hf.space/reset
```

## 🚀 Quick Start (For Evaluators)

Run locally:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Test:

```bash
curl -X POST http://localhost:7860/reset
```

Run baseline agent:

```bash
python inference.py
```

## 🧠 Problem Overview

This environment simulates a real-world API gateway where an agent must decide:

- Accept request
- Reject request
- Queue request

under constraints:

- limited rate capacity
- dynamic traffic bursts
- queue pressure
- request priorities

The goal is to maximize service quality while avoiding violations and system overload.

## 🏆 Key Contribution

This environment introduces deterministic burst traffic with delayed system-level penalties, enabling evaluation of long-term decision-making under constrained API systems.

Unlike existing OpenEnv tasks, it captures realistic backend trade-offs between throughput, priority handling, and system stability.

This enables more reliable differentiation between shallow heuristics and agents capable of long-term planning under constrained system conditions.

## ⚡ Why This Environment Stands Out

Unlike toy environments, this system includes:

- deterministic burst traffic rather than random noise
- queue aging and delayed penalties
- dynamic rate-limit windows
- sustained pressure scenarios
- system load accumulation that introduces delayed penalties over future steps

This forces long-term decision-making instead of greedy one-step behavior.

## 🔁 Environment Loop (Simplified)

```text
Observe -> Decide -> Apply -> Update Queue -> Reward -> Repeat
```

## Why This Environment Matters

This environment models real backend decision-making:

- preserving rate-limit capacity
- prioritizing critical requests
- preventing system overload
- managing queue intelligently

Applicable to:

- API gateways
- load balancers
- request schedulers
- microservices

## Architecture

### 1. Environment Core

- [environment/env.py](environment/env.py)
- Handles state transitions, traffic simulation, dynamic load behavior, queue pressure, and rewards.

### 2. Grader

- [grader/grader.py](grader/grader.py)
- Produces:
  - score
  - success_rate
  - violations

### 3. API Server

- [app/main.py](app/main.py)
- FastAPI endpoints for environment interaction.

### 4. Baseline Inference

- [inference.py](inference.py)
- Runs evaluation and emits structured logs.

## Project Structure

```text
project/
├── app/main.py
├── environment/env.py
├── grader/grader.py
├── inference.py
├── models.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
├── client.py
├── pyproject.toml
├── uv.lock
└── server/app.py
```

## Observation Space

Observation fields:

- `current_requests` (int)
- `rate_limit_remaining` (int)
- `time_window_remaining` (int)
- `request_priority` (`low | medium | high`)
- `queue_size` (int)
- `avg_queue_wait` (float)
- `recent_violations` (int)
- `system_load` (int)

## Action Space

- `0` → accept
- `1` → reject
- `2` → queue

## Reward Design (Deterministic)

### Positive

- `+10` -> high priority accepted
- `+5` -> medium priority accepted
- `+2` -> low priority accepted

### Penalties

- `-10` -> reject high priority
- `-5` -> rate-limit violation
- queue overflow, queue aging, delayed overload, and inversion penalties apply contextually
- inefficient actions apply smaller negative shaping penalties

### Bonus

- `+1` -> queue cleared effectively in favorable conditions

### Reward Logic

The environment uses dense reward shaping with delayed consequences:

- base rewards favor serving important work
- positive rewards are scaled by system health, which depends on remaining rate-limit capacity
- overload causes future penalties through `system_load`
- poor queue composition can trigger priority inversion penalties
- long queue residence increases downstream penalties through queue aging

## Task Modes


- `easy` -> low traffic
- `medium` -> mixed traffic and moderate capacity
- `hard` -> burst traffic, tight limits, system pressure

All tasks are deterministic and fixed-length episodes of 50 steps.

## API

`/reset` and `/step` are POST only.

### `POST /reset`

Request body (optional):

```json
{ "task": "easy" }
```

### `POST /step`

Request:

```json
{ "action": 0 }
```

### `GET /state`
Returns current observation.

### `GET /grade`

Returns grader output:

```json
{ "score": 0.73, "success_rate": 0.88, "violations": 2 }
```

## Grader

Current scoring uses normalized reward, high-priority success, violation penalty, and system stability.

Formula implemented in [grader/grader.py](grader/grader.py):

```text
score = 0.35 * normalized_total_reward
      + 0.30 * success_rate
      + 0.20 * system_stability
      - 0.15 * violation_penalty
```

## What This Evaluates


- prioritization
- rate-limit management
- queue control
- long-term planning
- overload avoidance

## What Makes It Challenging


- delayed penalties
- burst windows
- queue aging
- dynamic load buildup
- non-greedy optimal strategy

## Baseline Inference


Uses:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Defaults are used for `API_BASE_URL` and `MODEL_NAME`. `HF_TOKEN` is read from the environment, with `OPENAI_API_KEY` also accepted by the implementation as a compatibility fallback.

Logs format:

```text
[START]
[STEP]
[END]
```

## Local Run


```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
python inference.py
```

## Docker

Build:
```bash
docker build -t api-rate-limit-env .
```

Run:
```bash
docker run --rm -p 7860:7860 api-rate-limit-env
```

## Hugging Face Deployment

```bash
hf auth login
openenv push --repo-id <your-username>/api-rate-limit-env
```

## Final Validation Status


- `openenv validate` -> PASS
- Docker -> PASS
- inference -> PASS
- HF deploy -> LIVE
- `/reset` -> HTTP 200

## Validation Checklist


- deterministic environment
- OpenEnv spec
- 3 tasks
- grader valid
- inference reproducible
- HF API working
