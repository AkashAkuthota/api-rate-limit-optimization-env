
## API Rate Limit Env

Deterministic trajectory-based policy for API rate-limit optimization using search-optimized action plans.

## Key Features

- Trajectory-first execution with no runtime planning
- Search-optimized action sequences for each task
- Minimal safety overrides for stability
- OpenAI proxy-compliant inference
- Deterministic and reproducible results

## Tech Stack

- Python
- FastAPI
- OpenEnv
- OpenAI SDK

## How It Works

- Uses precomputed action sequences for `easy`, `medium`, and `hard`
- Executes each action step-by-step in the environment
- Applies only two safety rules during execution:
  - avoid `ACCEPT` when rate-limit capacity is `0`
  - avoid `QUEUE` when queue size is `8` or higher

## Results

- Easy: ~0.695
- Medium: ~0.386
- Hard: ~0.174

## Deployment

- Live API: https://akashakuthota-api-rate-limit-env.hf.space

## Project Structure

```text
app/
environment/
grader/
inference.py
openenv.yaml
Dockerfile
requirements.txt
README.md
```

## 🌐 Live Deployment

- Space: https://huggingface.co/spaces/akashakuthota/api-rate-limit-env
- API Base URL: https://akashakuthota-api-rate-limit-env.hf.space

## ⚡ Quick Test

```bash
curl -X POST https://akashakuthota-api-rate-limit-env.hf.space/reset
```

## API Rate Limit Env

Deterministic trajectory-based policy for API rate-limit optimization using search-optimized action plans.

## Key Features

- Trajectory-first execution with no runtime planning
- Search-optimized action sequences
- Minimal safety overrides for stability
- OpenAI proxy-compliant inference
- Deterministic and reproducible results

## Tech Stack

- Python
- FastAPI
- OpenEnv
- OpenAI SDK

## How It Works

- Uses precomputed action sequences per task
- Executes each sequence step-by-step in the environment
- Applies minimal safety rules to avoid avoidable violations

## Results

- Easy: ~0.695
- Medium: ~0.386
- Hard: ~0.174

## Deployment

- Hugging Face Space (Docker)
- Live API: https://akashakuthota-api-rate-limit-env.hf.space

