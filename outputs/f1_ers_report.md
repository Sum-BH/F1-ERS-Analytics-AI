# F1 ERS Deployment Analysis — Monza
**Generated:** 2025-09-21 19:14:46 UTC

## 1) Project Summary
This project demonstrates a full Data Science + ML + RL + GenAI pipeline for ERS (Energy Recovery System) deployment analysis on an F1 track (Monza). It includes:
- Synthetic telemetry generation (FIA realistic ERS budgets normalized per lap)
- Feature engineering and per-segment aggregation
- ERS deployment event extraction and efficiency computation (s saved per MJ)
- Transformer-style classifier demo (predict deploy decisions)
- Gymnasium + PPO RL skeleton and training to learn deploy policy
- RAG retrieval + local Mistral GGUF or Hugging Face fallback for an Assistant

## 2) Key Numbers (per-lap aggregates)
- Lap 0: lap_time = 95.700 s | estimated time saved = 2.221 s | pred_no_deploy = 97.921 s | efficiency = 0.600 s/MJ
- Lap 1: lap_time = 95.740 s | estimated time saved = 2.345 s | pred_no_deploy = 98.085 s | efficiency = 0.600 s/MJ
- Lap 2: lap_time = 95.668 s | estimated time saved = 2.170 s | pred_no_deploy = 97.838 s | efficiency = 0.600 s/MJ
- Lap 3: lap_time = 95.628 s | estimated time saved = 2.343 s | pred_no_deploy = 97.971 s | efficiency = 0.600 s/MJ
- Lap 4: lap_time = 95.636 s | estimated time saved = 2.616 s | pred_no_deploy = 98.252 s | efficiency = 0.600 s/MJ
- Lap 5: lap_time = 95.632 s | estimated time saved = 2.391 s | pred_no_deploy = 98.023 s | efficiency = 0.600 s/MJ

## 3) Deployment Events (top 10 by time_saved)
- Lap 5: 0.134 MJ deployed from 541m to 550m, duration 0.16s, time saved 0.081s, efficiency 0.600 s/MJ
- Lap 1: 0.133 MJ deployed from 3915m to 3920m, duration 0.08s, time saved 0.080s, efficiency 0.600 s/MJ
- Lap 5: 0.130 MJ deployed from 1622m to 1632m, duration 0.16s, time saved 0.078s, efficiency 0.600 s/MJ
- Lap 5: 0.129 MJ deployed from 1583m to 1593m, duration 0.16s, time saved 0.078s, efficiency 0.600 s/MJ
- Lap 5: 0.124 MJ deployed from 2993m to 3003m, duration 0.16s, time saved 0.074s, efficiency 0.600 s/MJ
- Lap 2: 0.123 MJ deployed from 3239m to 3249m, duration 0.16s, time saved 0.074s, efficiency 0.600 s/MJ
- Lap 1: 0.121 MJ deployed from 1646m to 1651m, duration 0.08s, time saved 0.073s, efficiency 0.600 s/MJ
- Lap 1: 0.120 MJ deployed from 5373m to 5378m, duration 0.08s, time saved 0.072s, efficiency 0.600 s/MJ
- Lap 1: 0.119 MJ deployed from 5508m to 5513m, duration 0.08s, time saved 0.071s, efficiency 0.600 s/MJ
- Lap 1: 0.119 MJ deployed from 2148m to 2153m, duration 0.08s, time saved 0.071s, efficiency 0.600 s/MJ

## 4) Visuals
The report includes the following images (in /outputs):
- segments_bars.png (saved at outputs/segments_bars.png)
- deploy_timeline.png (saved at outputs/deploy_timeline.png)
- race_summary.png (saved at outputs/race_summary.png)
- ers_eff_trend.png (saved at outputs/ers_eff_trend.png)
- ppo_epoch_rewards.png (saved at outputs/ppo_epoch_rewards.png)

## 5) ML & RL Summary
- Transformer classifier: lightweight demo (sequence to next-step-deploy prediction).
- PPO RL: environment (F1HybridEnv), trained for demo epochs. Check PPO reward curve in the visuals.

## 6) RAG & Assistant
RAG uses FAISS over deployment event summaries and the full report text. The Assistant will retrieve the most relevant pieces of evidence from the report and then answer using local LLM if available (or Hugging Face fallback).

## 7) How to interpret the outputs (short guide)
- `total_deploy_MJ` and `total_harvest_MJ` are in MJ (megajoules).
- `time_saved_s` is the estimated seconds saved from deployment, computed with a conservative gain factor (approx 0.6 s per MJ).
- `efficiency_s_per_MJ` is the seconds saved per MJ deployed — higher is better.

## 8) Next steps / Improvements
- Use real telemetry (FastF1) and per-driver models.
- Improve RL environment physics (use a physics-based longitudinal model).
- Add uncertainty estimates to time-saved and calibrate gain factor against real data.

## Appendix: snapshot of top segments
|   segment_idx |   avg_speed_kmh |   total_harvest_MJ |   total_deploy_MJ |   est_time_saved_s |   efficiency_s_per_MJ |
|--------------:|----------------:|-------------------:|------------------:|-------------------:|----------------------:|
|            19 |         157.031 |           0        |          0.763297 |           0.457978 |                   0.6 |
|            43 |         311.91  |           0.215522 |          0.746706 |           0.448023 |                   0.6 |
|            49 |         157.398 |           0        |          0.707231 |           0.424339 |                   0.6 |
|            52 |         335.625 |           0.23471  |          0.704325 |           0.422595 |                   0.6 |
|             4 |         243.184 |           0.296567 |          0.574444 |           0.344666 |                   0.6 |
|             1 |         312.058 |           0        |          0.556177 |           0.333706 |                   0.6 |
|            12 |         335.94  |           0.241572 |          0.545248 |           0.327149 |                   0.6 |
|            39 |         157.098 |           0        |          0.530298 |           0.318179 |                   0.6 |