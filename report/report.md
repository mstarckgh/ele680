# 24-hour-ahead electricity load forecasting: recurrent vs. convolutional networks and the role of seasonal baselines

## Abstract

We compare LSTM, GRU and temporal convolutional networks (TCN) of matched size, plus a simplified CNN-GRU hybrid, for one-shot 24-hour-ahead forecasting of 10 clients of the UCI ElectricityLoadDiagrams20112014 data. Trained directly, all networks are worse than the trivial forecast "repeat last week" (MAE 16.5-17.9 kW against 14.4 kW). Letting the network predict only a correction to last week's load removes the gap; the residual TCN is then reliably better than the residual GRU (0.52 kW, paired 95% CI [0.21, 0.94]) but its advantage over last week's load is small and concentrated in a few days. A plain average of the same hours over the last four weeks is better than last week's load and at least as accurate as every learned model we trained, including networks trained to correct that average. Longer inputs, L1 loss and the hybrid give no reliable benefit. The main lesson is methodological: against a strong seasonal baseline and with paired, day-level uncertainty, most architecture differences vanish.

## 1 Introduction

Electricity load has strong daily and weekly cycles, so a one-day-ahead forecast benefits from information up to a week or more back. Bai et al. [1] argue that generic convolutional networks (TCN) match or beat recurrent networks on sequence tasks, and Lai et al. [2] discuss why recurrent models struggle with long periodic patterns on this same dataset. We revisit that comparison on the UCI data [3] with a one-shot 24 h forecast, and ask whether the networks add anything beyond simple seasonal forecasts.

Our contributions, in the order the work developed:

1. A negative baseline result: with the standard setup, all three networks lose to "repeat last week" (Section 3.1).
2. A residual formulation in which the network predicts only a correction to a seasonal forecast, and a comparison of anchors: last week's load and the four-week mean (3.1, 3.3).
3. Stronger non-learned baselines (multi-week averages) and a paired, day-level evaluation with block-bootstrap intervals, which changes several conclusions that the mean MAE alone would suggest (3.2-3.4).
4. A test of the hybrid design suggested by the assignment, a simplified convolution + recurrence model inspired by LSTNet [2], which brings no gain over a GRU (3.2).

## 2 Methods and materials

**Data.** We use 10 of the 370 clients (MT_156, 159, 169, 232, 242, 248, 266, 281, 302, 307), drawn once with a fixed seed among clients with no zero-load hours since 2012 (`configs/clients.json`). The 15-minute readings are averaged to hourly values. The series from 2012-01-01 is split chronologically 70/10/20 into train/validation/test (the test period runs from late May to December 2014); z-scoring uses training statistics only and windows never cross split boundaries. Validation and test windows start at midnight, so every forecast covers one calendar day.

**Task and metrics.** Input: the last L hours of all 10 clients (L x 10). Output: the next 24 hours of all clients in one shot (24 x 10), from a linear head on the last time step. Metrics are computed in kW: MAE (primary) and RMSE, plus WAPE per client (client MAE divided by its mean load). The October 2014 clock change doubles one 01:00 hour; a secondary "noDST" MAE excludes the two affected days, and changes the MAEs of the paired-analysis runs by at most 0.05 kW.

**Models** (about 68k parameters each, so capacity is comparable): LSTM, 2 layers x 64 units (68,336 parameters); GRU, 2 layers x 72 units (67,200); TCN, 7 residual blocks of two weight-normalised causal dilated convolutions, kernel 3, dilations 1 ... 64, 38 channels (68,298; receptive field 1 + 4(2^n - 1) = 509 h), written from scratch following [1]. Training: Adam (lr 1e-3, halved on validation plateau), batch 64, MSE on normalised data, gradient clipping 1.0, dropout 0.1, at most 40 epochs, early stopping with patience 6. Every configuration is trained with 5 seeds.

**Residual variants.** The network output is added to a seasonal forecast taken from the input window, so the network only learns a correction: either the same 24 hours one week earlier (needs L >= 168), or their mean over the last four weeks (`residual: mean4`, needs L >= 672).

**Hybrid (CNN-GRU).** Inspired by the convolution + recurrence design of LSTNet [2], but a simplified model of our own: two causal convolutions (kernel 6, 32 channels, ReLU, dropout) with a 1x1 skip from the input, read by a 2-layer GRU (64 units), same head (67,856 parameters). It is not LSTNet: we omit its skip-recurrent component and its autoregressive highway; the seasonal residual plays a similar role to the highway but is a different mechanism.

**Baselines.** naive-24h (yesterday); naive-168h (same day last week); mean-2w, mean-4w and median-4w (mean or median of the same 24 hours over the last 2 or 4 weeks). The four-week baselines need 672 h of history.

**Paired evaluation.** A window needs L hours of history inside the test split, so a model with input length L is scored on days starting L hours into the test period: L = 672 sees 21 fewer days (191 vs 212) than L = 168, and mean MAEs across L are computed on different days. All comparisons that matter are therefore made on the days both models cover. For a model and a reference we report the mean daily MAE difference (diff = reference minus model, so positive favours the model; averaged over seeds), a 95% interval from a moving-block bootstrap (7-day blocks, because daily errors are autocorrelated), the share of days on which the model wins, and the range over seeds. These runs are retrained with forecasts saved and can differ slightly from the mean tables (e.g. 13.68 vs 13.52 kW for the TCN with L1 at L = 672), by about the seed spread. The intervals describe the single test period, not other years or clients.

**Reproducibility.** All configurations are generated by `scripts/make_configs.py`; `README.md` lists the commands for training (`src.train`), the tables (`src.plots`) and the paired analysis (`src.analyze`).

## 3 Results

### 3.1 Direct forecasts lose to last week; the residual formulation closes the gap

| test MAE [kW], L = 168 | direct, MSE | direct, L1 | residual, MSE | residual, L1 |
|---|---|---|---|---|
| LSTM | 17.87 +- 0.45 | - | - | - |
| GRU | 16.49 +- 0.25 | 16.12 +- 0.29 | 14.49 +- 0.13 | 14.50 +- 0.20 |
| TCN | 16.58 +- 0.34 | 16.42 +- 0.24 | 13.87 +- 0.13 | 13.79 +- 0.15 |
| CNN-GRU | 17.14 +- 0.39 | - | 14.48 +- 0.09 | - |

For reference, naive-24h has MAE 14.46 and naive-168h 14.43 kW (RMSE 30.7 and 28.3). Every directly trained network is worse than naive-168h; the LSTM is worst, GRU and TCN are indistinguishable. The gap is concentrated in client MT_159 (mean load 44 kW; WAPE 45-55% for the direct models against 30% for naive-168h). L1 loss helps the direct models slightly. The residual formulation helps far more: the GRU reaches the level of naive-168h and the TCN has a lower mean MAE by about 0.55 kW. Training is also faster with the residual (GRU: 9 epochs against 19). Section 3.2 shows how much of the TCN's mean gain is reliable.

### 3.2 Paired comparison over test days

| model | reference | days | diff [kW] | 95% CI | model better on | per-seed diff |
|---|---|---|---|---|---|---|
| GRU (res, L168) | naive-168h | 212 | -0.06 | [-0.26, 0.18] | 43% of days | -0.29 .. 0.11 |
| TCN (res, L168) | naive-168h | 212 | 0.46 | [0.02, 1.07] | 47% | 0.28 .. 0.54 |
| TCN (res+L1, L672) | naive-168h | 191 | 0.31 | [-0.03, 0.69] | 48% | 0.18 .. 0.48 |
| TCN (res, L168) | GRU (res, L168) | 212 | 0.52 | [0.21, 0.94] | 58% | 0.35 .. 0.60 |
| TCN (res+L1, L672) | GRU (res, L168) | 191 | 0.45 | [0.18, 0.75] | 60% | 0.32 .. 0.62 |
| CNN-GRU (res, L168) | GRU (res, L168) | 212 | 0.01 | [-0.14, 0.15] | 50% | -0.16 .. 0.11 |
| CNN-GRU (res, L168) | TCN (res, L168) | 212 | -0.51 | [-0.92, -0.20] | 40% | -0.68 .. -0.41 |

The residual GRU is indistinguishable from naive-168h. The residual TCN is better than the residual GRU, with an interval that excludes zero and wins on 58-60% of days: this is the one architecture effect that survives the paired test. Against naive-168h the TCN's mean gain barely excludes zero (L = 168) or includes it (L = 672 with L1), and the TCN is better on fewer than half of the days. The gain is concentrated: for the L = 168 TCN the median daily difference is -0.06 kW, and the mean gain falls from 0.46 to 0.12 kW when the ten days with the largest gains are removed. Those days lie mostly in 12-21 June 2014 (14 June: naive-168h 30.5 kW, TCN 15.4 kW), where repeating the previous week fails; we did not investigate the cause (a holiday or one-off event is plausible). The TCN is worse on 26 December (7.7 kW against 0.2 kW).

The hybrid is statistically identical to the GRU and clearly worse than the TCN: adding a convolutional front-end to a GRU does not reproduce the TCN's advantage.

### 3.3 Input length and receptive field

Residual test MAE [kW] by input length (Figure 1, `figures/residual_length.pdf`; the plain-model sweep is in `figures/input_length.pdf` and also shows no systematic benefit):

| residual model | L = 168 | L = 336 | L = 672 |
|---|---|---|---|
| GRU | 14.49 +- 0.13 | 14.54 +- 0.09 | 14.55 +- 0.40 |
| TCN | 13.87 +- 0.13 | 14.20 +- 0.20 | 14.02 +- 0.20 |
| CNN-GRU | 14.48 +- 0.09 | 14.69 +- 0.14 | 14.55 +- 0.32 |

Longer inputs do not help. On common days the residual GRU at L = 672 is worse than at L = 168 (-0.42 kW, CI [-0.71, -0.03]) and worse than naive-168h (-0.56, [-0.94, -0.07]); the residual TCN at L = 336 is slightly worse than at L = 168 (-0.20, [-0.38, -0.04]). The TCN with L1 loss at L = 672 has the lowest mean MAE (13.52), but against the L = 168 TCN the difference is +0.14 kW, CI [-0.05, 0.30]; since it also changes the loss, it does not isolate an effect of L. We therefore do not claim that a longer input or L1 loss improves the TCN: one week of history is the best or equal-best input length.

Receptive field (Figure 2, `figures/rf_residual.pdf`; residual TCN, L = 672, n = 3 ... 8 blocks, receptive field 29, 61, 125, 253, 509, 1021 h): MAE 14.47, 14.38, 14.37, **14.00**, 14.02, 14.03 kW (std 0.08-0.31). The error is flat, close to naive-168h, while the receptive field is shorter than one week, drops by about 0.37 kW once it exceeds 168 h, and does not improve further. The hidden width is fixed, so the parameter count grows with depth (33k to 77k); the plateau from n = 6 to 8 occurs while the parameter count still grows by 30%, so capacity alone does not explain the step. The pattern is consistent with the network needing to see the same hours of the previous week to learn useful corrections, but we did not test this further.

### 3.4 A multi-week average is at least as good as the learned models

| model | reference | days | diff [kW] | 95% CI | model better on |
|---|---|---|---|---|---|
| mean-4w | naive-168h | 191 | 0.76 | [0.14, 1.42] | 60% of days |
| TCN (res, L168) | mean-4w | 191 | -0.59 | [-1.17, 0.01] | 39% |
| TCN (res+L1, L672) | mean-4w | 191 | -0.45 | [-1.04, 0.15] | 42% |
| GRU (res, L168) | mean-4w | 191 | -0.90 | [-1.56, -0.28] | 38% |
| TCN (res on mean-4w, L672) | mean-4w | 191 | -0.47 | [-0.83, -0.00] | 39% |
| TCN (res on mean-4w + L1, L672) | mean-4w | 191 | -0.31 | [-0.71, 0.17] | 37% |
| GRU (res on mean-4w, L672) | mean-4w | 191 | -0.67 | [-1.15, -0.02] | 35% |

On the 191 days that all models cover, mean-4w has MAE 13.23 kW, against 13.68 for the best learned model in the paired runs (TCN, residual + L1, L = 672). It improves on naive-168h by 0.76 kW; mean-2w (+0.49) and median-4w (+0.69) behave similarly. Averaging removes the noise of one atypical week, the failure mode seen in June 2014. No learned model is better than mean-4w: the intervals of the TCNs include or barely exclude zero, and the GRU is clearly worse. We also trained the networks to correct mean-4w itself (`residual: mean4`, L = 672); their MAE is 13.70 (TCN), 13.54 (TCN + L1) and 13.90 kW (GRU) against 13.23, and the learned correction does not improve on the TCN anchored to last week (+0.28 kW, CI [-0.38, 0.97]). The correction therefore adds error instead of removing it; one possible explanation is overfitting to the training years, which we did not test (e.g. validation versus test error). Seed spread is large (up to 1.5 kW against mean-4w for one TCN seed), so single runs are unreliable.

### 3.5 Per client and per forecast hour

WAPE in % (mean load in kW in the header); rows marked * use the 191-day period, the others 212 days.

| model | MT_156 | MT_159 | MT_169 | MT_232 | MT_242 | MT_248 | MT_266 | MT_281 | MT_302 | MT_307 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean load [kW] | 89 | 44 | 43 | 82 | 178 | 155 | 268 | 111 | 117 | 767 |
| naive-24h | 24.5 | 48.5 | 5.9 | **6.2** | **6.7** | **5.0** | **6.3** | **4.7** | **5.0** | **6.0** |
| naive-168h | 14.4 | 30.0 | 6.8 | 7.6 | 8.1 | 6.3 | 7.2 | 5.8 | 6.4 | 6.8 |
| mean-4w* | **12.0** | **26.8** | **5.8** | 6.6 | 7.9 | 5.5 | 6.7 | 5.5 | 7.0 | 6.1 |
| GRU (res, L168) | 14.7 | 30.8 | 6.7 | 7.6 | 8.5 | 6.2 | 7.2 | 5.8 | 6.3 | 6.7 |
| TCN (res, L168) | 14.3 | 33.2 | 6.3 | 7.3 | 8.3 | 5.7 | 7.0 | 5.6 | 5.9 | 6.3 |

The two irregular clients (MT_156, MT_159) have by far the largest relative errors for every model, and they are where averaging over weeks helps most. "Repeat yesterday" has a lower WAPE than every learned model on 8 of the 10 clients (all but MT_156 and MT_159), and the lowest of all rows on 7 of them, but is much worse on the two irregular clients, so its pooled MAE equals naive-168h. All residual models are anchored to last week; for clients with a weak weekly cycle this anchor is worse than yesterday and the networks do not fully compensate. The pooled kW MAE also weights clients by load (MT_307 alone contributes about a third), so per-client relative errors and pooled errors can rank methods differently. Per forecast hour (Figure 3, `figures/horizon_residual.pdf`) all models and baselines share the same shape: errors are lowest at night (forecast hours 2-6, about 10 kW) and peak in the afternoon (16-17 kW), so differences between methods are small compared with the variation over the day.

## 4 Discussion

**Why do direct networks lose?** Naive-168h is a strong baseline because load is dominated by weekly repetition. A direct network must reconstruct last week's values from normalised inputs through a bottleneck (the last hidden state or time step, 38-72 units) before the head, which costs accuracy on the part of the signal that is easiest to predict; the residual formulation hands that part over directly. This is plausible but not isolated by our experiments: the benefit may also come from the easier optimisation of a low-variance target.

**Recurrent vs. convolutional.** Without the residual we do not reproduce a clear advantage of the TCN over the GRU as reported in [1], although the LSTM is worst. With the residual the TCN is reliably better than the GRU. It is not the longer window that produces this (L = 168 is as good as longer inputs); the step in the receptive-field experiment suggests that covering one week matters. The advantage may come from the convolutional inductive bias on the correction signal, or from capacity and optimisation; we did not separate these. The hybrid shows that the gain is not simply "convolutions help", and since it omits LSTNet's skip-recurrent and highway components, our result is no evidence against the full LSTNet [2]. Numbers are not comparable with [2], which uses other clients, splits and metrics.

**What do the networks add?** Little, in this setting. Past load is largely explained by its weekly (and multi-week average) profile; the corrections the networks learn do not reliably beat a four-week average, even when they are trained directly on it. This is a finding for the tested configurations and period, not a general claim that neural networks cannot improve on seasonal averages.

**Limitations.** Only 10 clients and a single chronological split (about 200 test days), so conclusions may depend on the test period; one hyperparameter setting per architecture, with capacity matched by parameter count rather than tuned; models see past load only, with no calendar, holiday or weather inputs, which probably handicaps the direct networks (not tested); training windows start at every hour but test windows at midnight, which may cost the direct models accuracy (not tested); the June 2014 gain concentration was not traced to a cause. Future work: rolling-origin evaluation over several periods, more clients, calendar and weather covariates, an equal tuning budget, and learned models that combine several seasonal anchors.

## 5 Conclusion

Directly trained LSTM, GRU and TCN forecasters lose to "repeat last week" for 24-hour-ahead load of 10 clients. Predicting a correction to last week's load removes the gap and makes the TCN reliably better than the GRU, but the TCN's advantage over the seasonal baseline is small and rests on a few days, and a four-week average is at least as accurate as every learned model we trained, including a convolution + recurrence hybrid, longer inputs, L1 loss and networks trained to correct the average. For this task, a strong seasonal baseline and paired, day-level evaluation matter more than the choice of architecture.

## References

[1] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.

[2] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, "Modeling long- and short-term temporal patterns with deep neural networks," in *Proc. 41st Int. ACM SIGIR Conf. Research and Development in Information Retrieval*, 2018.

[3] A. Trindade, "ElectricityLoadDiagrams20112014," UCI Machine Learning Repository, 2015. [Online]. Available: https://doi.org/10.24432/C58C86
