# Pretrained time-series foundation models (TSFMs) as backbones for few-shot classification of industrial / partial-discharge signals

Scope note: researched 2026-10-09. Model cards and repos change often; where a fact came only from an aggregator or secondary source it is flagged. "Not found" = not verified in any fetched primary source. Anything from 2024 or earlier (Lag-Llama, Timer 1.0, GPT4TS, Time-LLM, MOMENT-1, Moirai 1.x, original Chronos) should be treated as potentially superseded.

## Q1. Candidate TSFMs: architecture, size, context/patch, classification support, pretraining data, license, availability, fine-tuning options

### Takeaway
Only a handful of TSFMs are classification-native out of the box: Mantis / Mantis+ / MantisV2 (Apache-2.0, ~8M params, built for TSC), MOMENT (MIT, 40M–385M, has a classification head + embeddings), UniTS (MIT), TSPulse (Apache-2.0, ~1M), and GPT4TS/OFA. Every forecasting TSFM (Chronos/Chronos-Bolt/Chronos-2, TimesFM, Moirai, Timer/Sundial, Time-MoE, Toto, Lag-Llama, TiRex) can only be used for classification by pooling hidden states yourself. Licenses split sharply: Moirai (all versions) and TimesFM 3.0 weights are non-commercial / research-only; Chronos-2, TimesFM ≤2.5, Sundial/Timer, Time-MoE, Toto, Mantis, TSPulse, Lag-Llama are Apache-2.0; MOMENT and UniTS are MIT.

### Cited Findings

**Classification-native / general-purpose TSFMs**

- **Mantis (V1, "Mantis-8M")** — HF checkpoint `paris-noah/Mantis-8M`, 8.11M params, Apache-2.0 — [HF model card](https://huggingface.co/paris-noah/Mantis-8M)
- Mantis architecture: input instance-normalized and resized to 512 by default; fixed 32 tokens (input length must be a multiple of 32); token generator has three branches (conv patch features of normalized signal, conv features of first-order difference, per-patch mean/std via a Multi-Scaled Scalar Encoder) → 32 tokens of dim 256, class token, 6 pre-norm Transformer layers, 8 heads, RoPE; contrastive (InfoNCE, temperature 0.1) pretraining with Random Crop Resize augmentation; layer pruning can reduce it to ~2.2M params — [Mantis arXiv HTML 2502.15637](https://arxiv.org/html/2502.15637)
- Mantis family checkpoints: Mantis-8M, MantisPlus (`paris-noah/MantisPlus`), UTICA (`fegounna/Utica`, self-distilled, pin a revision), MantisV2 (`paris-noah/MantisV2`); Mantis+ and MantisV2 pretrained on CauKer-2M synthetic data; `pip install mantis-tsfm`; `MantisTrainer.transform(X)` for frozen embeddings (recommended `return_transf_layer`=2 for Mantis/UTICA/MantisV2, 1 for Mantis+; `output_token='combined'`; "feature normalization matters for linear classifiers"); `fit(X,y)` trains a head (default BatchNorm+Linear since v1.1.0); multichannel adapters `MultichannelProjector` (e.g., PCA) and `LinearChannelCombiner` (`fine_tuning_type='adapter_head'`); ICML 2026 paper combining V1/V2 (openreview gbJMAjXLZ4) — [Mantis GitHub](https://github.com/vfeofanov/mantis)
- MantisV2 / Mantis+ report (Feofanov et al., 19 Feb 2026): MantisV2 is a lighter refined encoder; adds a test-time method using intermediate-layer representations and refined output-token aggregation, plus self-ensembling and cross-model embedding fusion; claims to "consistently outperform prior time series foundation models" on UCR, UEA, HAR, EEG; notes "a substantial performance gap remained between frozen and fine-tuned encoders" — [arXiv 2602.17868](https://arxiv.org/abs/2602.17868)
- MantisV2 parameter count: not found.
- **MOMENT (AutonLab/CMU)** — T5-style encoder; univariate input fixed T=512, 64 patches of length 8; RevIN; masked reconstruction (~30% patches masked, MSE); sizes Small 6L/512d ≈40M, Base 12L/768d ≈125M, Large 24L/1024d ≈385M; multivariate handled channel-independently; series >512 are subsampled, shorter are left-zero-padded; "cannot differentiate vertically shifted series" because of per-signal normalization — [MOMENT arXiv HTML 2402.03885](https://arxiv.org/html/2402.03885)
- MOMENT-1-large HF card: ~0.3B params, MIT license, tasks forecasting / classification / anomaly detection / imputation / embeddings, pretrained on AutonLab/Timeseries-PILE; PEFT ECG-classification tutorial — [HF MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large)
- MOMENT repo: `pip install momentfm`; classification via `task_name="classification"`, `n_channels`, `num_class`; linear probing described; PEFT + multi-GPU in tutorial; MIT — [MOMENT GitHub](https://github.com/moment-timeseries-foundation-model/moment)
- Parameter-count discrepancy: arXiv 2511.23177 reports MOMENT small/base/large as 35.42M / 109.77M / 341.42M — [arXiv 2511.23177 HTML](https://arxiv.org/html/2511.23177) vs ≈40M/125M/385M in [MOMENT paper](https://arxiv.org/html/2402.03885) (likely counting with/without heads/embeddings; unresolved).
- **UniTS** (Harvard, NeurIPS 2024, arXiv 2403.00131): one shared model for classification, forecasting, anomaly detection, imputation across 38 datasets; checkpoints in a GitHub "ckpt" release; few-shot fine-tuning and prompt tuning scripts for classification; MIT license; built on Time-Series-Library. Parameter count not stated — [UniTS GitHub](https://github.com/mims-harvard/UniTS)
- **TSPulse (IBM Granite)** — ~1M params (1.08M), Apache-2.0 on the model card, base context 512 (classification accepts any length and resamples to 512), patch length 8 or 16 depending on variant, tasks classification / anomaly detection / imputation / similarity search; pretrained on public Zenodo/Monash-style data (electricity, weather, solar, wind, traffic, etc.); fine-tuning script "will be added later"; claims 5–16% gain on UEA — [HF granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
- TSPulse uses dual-space (time + frequency) masked reconstruction and disentangled detailed vs semantic embeddings (arXiv 2505.13033) — [fugumt summary of 2505.13033](https://fugumt.com/fugumt/paper_check/2505.13033v2) (secondary; the CC BY-NC-ND 4.0 tag seen there refers to the arXiv paper text, the model card says Apache-2.0).
- **NuTime** (Lin et al., TMLR 2024): Transformer over non-overlapping windows; each window = normalized shape + mean + std embedded by a numerically multi-scaled embedding; contrastive pretraining on >1M public sequences; evaluated on classification, few-shot, clustering, anomaly detection; code at github.com/chenguolin/NuTime — [arXiv 2310.07402](https://arxiv.org/abs/2310.07402). Param count and weight release: not found.
- **GPT4TS / OFA ("One Fits All")** (Zhou et al., NeurIPS 2023 Spotlight): keeps the self-attention and FFN layers of a pretrained LM frozen; tasks include classification, anomaly detection, forecasting, few-shot — [arXiv 2302.11939](https://arxiv.org/abs/2302.11939). Only layer norm and positional embeddings trainable (as described by a later paper) — [UTICA arXiv 2603.01348 PDF](https://arxiv.org/pdf/2603.01348) (secondary description).
- **Time-LLM** (Jin et al., ICLR 2024): frozen LLM, input patches "reprogrammed" with text prototypes + Prompt-as-Prefix; abstract covers forecasting only (few-/zero-shot forecasting) — [arXiv 2310.01728](https://arxiv.org/abs/2310.01728). Tan et al. list Time-LLM (LLaMA backbone) at ~6,642M params — [Tan et al. HTML](https://arxiv.org/html/2406.16964).

**Forecasting TSFMs (classification only via self-extracted embeddings)**

- **Chronos-2** (Amazon, Oct 2025, arXiv 2510.15821): encoder-only, T5-encoder-inspired, group attention for in-context learning across related series/covariates; 120M params; context 8192; max horizon 1024; Apache-2.0; trained on Chronos datasets subset + GIFT-Eval Pretrain subset + synthetic uni/multivariate data; card documents forecasting only (no embedding API documented) — [HF amazon/chronos-2](https://huggingface.co/amazon/chronos-2). Same card: Chronos-Bolt context 2048, original Chronos context 512.
- **Chronos-Bolt**: T5 encoder-decoder over patches, direct multi-step quantile decoding; Tiny 9M / Mini 21M / Small 48M / Base 205M; Apache-2.0; ~100B observations — [HF chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base). Patch size: not found.
- **TimesFM 2.5** (Google, released 15 Sep 2025): decoder-only, 200M params (+ optional 30M quantile head), context 16k (vs 2048 in 2.0, which was ~500M), frequency indicator removed, XReg covariates restored Oct 2025, LoRA fine-tuning example (HF Transformers + PEFT) added 9 Apr 2026; Apache-2.0 for code and weights up to 2.5; no embedding/classification support documented — [TimesFM GitHub](https://github.com/google-research/timesfm); pretraining data: GiftEvalPretrain, Wikimedia pageviews, Google Trends, synthetic — [HF timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch). Patch size: not found in fetched sources.
- **TimesFM 3.0** (31 Aug 2026): code Apache-2.0 but weights under "timesfm-non-commercial-license-v1.0" — [TimesFM GitHub](https://github.com/google-research/timesfm) (repo note) and secondary reports e.g. [wikidocs](https://wikidocs.net/424375), [datanorth](https://datanorth.ai/nl/nieuws/google-lanceert-timesfm-3). License text itself not verified on HF.
- **Moirai 1.1-R-large**: 0.3B params, CC-BY-NC-4.0, "research purposes only" — [HF moirai-1.1-R-large](https://huggingface.co/Salesforce/moirai-1.1-R-large)
- **Moirai-MoE-1.0-R-base**: 0.9B total params, CC-BY-NC-4.0; active params not stated — [HF moirai-moe-1.0-R-base](https://huggingface.co/Salesforce/moirai-moe-1.0-R-base)
- **Moirai 2.0-R-small**: decoder-only, quantile loss, multi-token prediction; 11.4M params; CC-BY-NC-4.0, research only; data GIFT-Eval pretrain/train subsets, Chronos mixup, KernelSynth synthetic, internal Salesforce ops data — [HF moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small); paper arXiv 2511.11698 (36M-series corpus, "thirty times smaller" than Moirai 1.0-Large) — [arXiv 2511.11698](https://arxiv.org/html/2511.11698v1). Conflict: TS-Arena lists 14M and a 2026 survey lists 36M — [search summary incl. TS-Arena](https://dag-upb-ts-arena.hf.space/models/family/moirai).
- **Sundial ("Timer 3.0", THUML)**: decoder-only, patch 16, 12 layers, 128M params, context up to 2880, TimeFlow (flow-matching) loss, pretrained on ~1T (1032B) points from UTSD + LOTSA + Chronos datasets; Apache-2.0; forecasting only; pin transformers==4.40.1 — [HF sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m)
- **Timer (timer-base-84m)**: decoder-only, 84M, patch 96, context up to 2880, 8 layers, 260B points (UTSD + LOTSA), Apache-2.0; cites Timer (2402.02368) and Timer-XL (2410.04803); fine-tuning via OpenLTM — [HF timer-base-84m](https://huggingface.co/thuml/timer-base-84m). Separate Timer-XL checkpoint: not found.
- **Time-MoE** (ICLR 2025 Spotlight): checkpoints TimeMoE-50M and TimeMoE-200M; claims scaling to 2.4B; max context 4096 (context+horizon ≤4096); Time-300B (>300B points, >9 domains); Apache-2.0; fine-tuning supported; forecasting only, "classification" is on the TODO list — [Time-MoE GitHub](https://github.com/Time-MoE/Time-MoE). HF card for the "200M" checkpoint shows 0.5B tensor params (probably total incl. experts; unverified) — [HF TimeMoE-200M](https://huggingface.co/Maple728/TimeMoE-200M)
- **Toto-Open-Base-1.0** (Datadog): decoder-only, Proportional Factorized Space-Time Attention, Student-T mixture head; ~151M params; Apache-2.0; >2T points (≈1T Datadog observability metrics + GIFT-Eval pretrain + Chronos data + ~1/3 synthetic); forecasting only; `pip install toto-ts` — [HF Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0)
- **Lag-Llama**: 2.45M params, trained context 32 (RoPE scaling for longer), Apache-2.0, probabilistic forecasting; fine-tuning script provided — [HF Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama)
- **TiRex** (NX-AI): 35M-param xLSTM forecaster; HF page links a TiRex classification model and the Auer et al. classification paper — [HF NX-AI/TiRex](https://huggingface.co/NX-AI/TiRex); license reported as "NXAI community license" (aggregator only) — [awesome.ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2FNX-AI%2Ftirex)

**Newer (2025–2026) classification-oriented models / methods**

- **TimEE** (Küken, Hoo, Mráz, Hutter, Purucker; 8 Jul 2026): 4.5M-param in-context classifier (support set + query → class distribution in one forward pass), meta-trained only on synthetic VARX-prior tasks; claims 1st in ROC AUC and 3rd in accuracy on UCR among compared methods (DTW-1NN, Catch22, MiniRocket, Hydra, InceptionTime, TS2Vec, NuTime, MOMENT, MantisV2, TiCT, Chronos-2, TiRex) — [arXiv 2607.07500](https://arxiv.org/html/2607.07500)
- **TIC-FM** (Fang et al., Jan 2026, rev. Jul 2026): in-context inference using the labeled training set as context; argues "frozen encoder + trained classifier" zero-shot protocols are biased by classifier choices; 128 UCR datasets, largest gains in extreme low-label settings; code public — [arXiv 2602.00620](https://arxiv.org/abs/2602.00620)
- **RocketPFN** (O'Rourke, Trisovic, Bertsimas; 19 Jun 2026): random-conv features + PFN-style in-context learning — [arXiv 2606.21786](https://arxiv.org/html/2606.21786)
- **UTICA** (multi-objective self-distillation pretraining for TSC, Mantis backbone) — [arXiv 2603.01348](https://arxiv.org/pdf/2603.01348); checkpoint `fegounna/Utica` — [Mantis GitHub](https://github.com/vfeofanov/mantis)
- Other 2026 titles surfaced but not read: UniShape shape-aware TSC foundation model [arXiv 2601.06429](https://arxiv.org/html/2601.06429); In-Context TSC with Random Convolutional Features [arXiv 2607.19234](https://arxiv.org/html/2607.19234); TS2TabPFN [arXiv 2608.04174](https://arxiv.org/pdf/2608.04174).
- **FlowState** (IBM): SSM encoder + functional basis decoder, sampling-rate-invariant forecasting TSFM (NeurIPS 2025 workshop; later version lists ICML 2026) — [arXiv 2508.05287](https://arxiv.org/pdf/2508.05287)

### Inferences
- Practical shortlist for a few-shot PD classification paper on 8× RTX 4090 (all fit trivially; the largest, MOMENT-Large ≈385M, can be fully fine-tuned in fp32/bf16 on one 24 GB card at moderate batch size — engineering estimate, not a sourced figure):
  1. **Mantis-8M and/or MantisV2** (primary backbone): classification-native, contrastive, Apache-2.0, strongest published few-shot industrial result (see Q2), built-in channel adapters, numpy-in/numpy-out API that plays well with CSV data.
  2. **MOMENT-1-base or -large** (the canonical "general-purpose, masked-reconstruction" TSFM; MIT; LoRA/full FT): the expected reviewer baseline and the natural contrast for a "pretraining objective matters under few labels" analysis.
  3. **One forecasting TSFM used as frozen feature extractor** — Chronos-2 (Apache-2.0, 120M, 8192 context) is the cleanest license-wise; TimesFM 2.5 (Apache-2.0, 16k context) is the alternative. Avoid Moirai (CC-BY-NC-4.0, research-only) and TimesFM 3.0 weights (non-commercial) unless the paper is purely academic and you state the license; TiRex licensing should be checked before use.
  4. Optional 4th: **TSPulse** (1M, Apache-2.0, time+frequency dual-space pretraining — conceptually attractive for impulsive signals) or an **in-context classifier** (TimEE / TIC-FM / RocketPFN) to show awareness of the 2026 state of the art.
- LLM-based models (GPT4TS/OFA, Time-LLM) are poor choices as primary backbones (see Q3); if included, include them only as baselines with the Tan et al. ablations.
- Most forecasting TSFMs expose no embedding API, so you must hook hidden states yourself; this makes layer choice a hyperparameter that must be fixed on a validation split, not tuned on test (see Q2).

### Gaps
- Patch sizes for Chronos-2, Chronos-Bolt, TimesFM 2.5, Toto, Moirai 2.0: not found in fetched primary sources.
- MantisV2, UniTS, NuTime, TimEE-architecture param details, and original Mantis-8M pretraining corpus size: not verified (original Mantis corpus not stated in fetched content).
- Moirai 2.0 parameter count conflicts across sources (11.4M HF card vs 14M vs 36M).
- TiRex license verified only via an aggregator.
- Time-LLM backbone named only through Tan et al.; not confirmed on the abstract page.

## Q2. Published evidence: TSFMs on UCR/UEA and on industrial fault diagnosis in few-shot settings (incl. arXiv 2511.23177)

### Takeaway
On full-label UCR, frozen TSFM embeddings (≈0.78–0.86 mean accuracy depending on classifier/protocol) still trail supervised ROCKET/InceptionTime/HC2 (≈0.88–0.90). Among TSFMs, contrastive/classification-oriented Mantis is consistently best, good forecasting models (TiRex, Chronos-2, Chronos-Bolt, TimesFM 2.0) come close when multi-layer embeddings are used, and reconstruction-based MOMENT is consistently weaker as a frozen extractor. In the only industrial few-shot study directly on point (2511.23177, PMSM stator faults), frozen Mantis + SVM reached 84.1% at 1% labels vs 40.5% for frozen MOMENT + SVM. No study yet sweeps labels-per-class with TSFMs and the ROCKET family side by side on industrial data — this is an open, publishable gap.

### Cited Findings

**arXiv 2511.23177 — Data-Efficient Motor Condition Monitoring with TSFMs** (Deyu Li, Xinyuan Liao, Shaowei Chen, Shuai Zhao; v1 28 Nov 2025, v2 2 Dec 2025; eess.SP preprint, no venue) — [arXiv abs](https://arxiv.org/abs/2511.23177), [HTML](https://arxiv.org/html/2511.23177)
- Data: public PMSM stator-fault dataset (Jung et al., 2023), three PMSMs (1.0/1.5/3.0 kW), 16 operating conditions, 45 classes; 3-phase current downsampled 2000 Hz → 512 Hz to match one vibration axis; windows of 512 steps (the fixed TSFM input length), 4 channels; ~960k samples; random 80/20 split; 1% and 5% label subsets (at 1%: ~80 normal and ~45 per fault class) — [HTML](https://arxiv.org/html/2511.23177)
- Methods: MOMENT and Mantis as frozen feature extractors → LogME scoring and SVM/RF/DT/k-NN; MOMENT also LoRA and full FT; Mantis full FT on CPU; baselines raw-signal SVM, CNN, LSTM, Transformer (MOMENT architecture w/o pretraining), TSSequencerPlus, MiniRocket (cited reference is MultiRocket), TSPerceiver, XCM, PatchTST; ResNet/ROCKET/InceptionTime not used — [HTML](https://arxiv.org/html/2511.23177)
- LogME (5%→80%): MOMENT 0.627→0.697; Mantis 0.877→0.953 — [HTML](https://arxiv.org/html/2511.23177)
- Frozen SVM accuracy: at 1% raw 39.7 / MOMENT 40.5 / Mantis 84.1; at 5% 52.7 / 53.9 / 90.9; at 80% 52.7 / 54.4 / 91.3 — [HTML](https://arxiv.org/html/2511.23177)
- Abstract claims "MOMENT achieves nearly twice the performance of conventional deep learning models using only 1% of the training data" and "Mantis surpasses state-of-the-art baselines by 22%, reaching 90% accuracy with the same data ratio" — [arXiv abs](https://arxiv.org/abs/2511.23177). Inconsistency: Table III gives Mantis-SVM 84.1% at 1%; 90.9% is the 5% figure — [HTML](https://arxiv.org/html/2511.23177)
- Explanation offered: MOMENT's masked-reconstruction representations "need larger datasets" and lag contrastive models when data are scarce; Mantis' contrastive pretraining yields discriminative features; MOMENT-large LoRA reaches >70% at 5%, close to its full FT; non-pretrained Transformer narrows but never closes the gap — [HTML](https://arxiv.org/html/2511.23177)

**UCR/UEA evidence**
- MOMENT paper (frozen embeddings + SVM, 91 UCR datasets with length <512): mean acc MOMENT 0.794, TS2Vec 0.851, T-Loss 0.833, ResNet (supervised) 0.825, FCN 0.809, TS-TCC 0.793, TNC 0.786, DTW 0.764, GPT4TS 0.566, TimesNet 0.572; no dedicated few-shot classification experiment — [MOMENT HTML](https://arxiv.org/html/2402.03885)
- Auer, Klotz, Böck, Hochreiter, "Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for TSC" (NeurIPS 2025 TSFM workshop): frozen embeddings + Random Forest on UCR (127) + UEA (30), excluding series >2048; multi-layer mean-pooled hidden states concatenated; per-variate processing for multivariate; augmentations "absolute sample statistics" (mean/std/min/max of 8 patches, to offset scale lost by instance norm) and first-difference embeddings. Accuracy (Stat+Diff): TiRex 0.80; Chronos-Bolt-Base 0.78; Moirai-Large 0.78; TimesFM 2.0 0.78; Mantis (no aug) 0.78; Chronos-Base 0.75; TimesFM 1.0 0.74; Toto 0.73; DTW 0.73; NuTime 0.67; MOMENT-Large 0.62. Positive but noisy correlation with GIFT-Eval forecasting skill; with a 1-NN classifier forecasting models no longer beat Mantis; no MiniRocket/supervised comparison — [arXiv 2510.26777 HTML](https://arxiv.org/html/2510.26777v1), [NeurIPS page](https://neurips.cc/virtual/2025/130477)
- Mantis paper (latest version; frozen + logistic regression/RF): mean acc UCR — Mantis 0.8195, TiConvNext 0.8029, TiRex 0.8013, Chronos2 0.8002, Catch22+ 0.7969, TiViT-H 0.7943, TabPFN 0.7806, MOMENT 0.7789, NuTime 0.7732, TabICL 0.7707; UEA-27 — Mantis 0.7420, NuTime 0.7199, TiConvNext 0.7105, TiViT-H 0.7091, MOMENT 0.6991, Chronos2 0.6967, Catch22+ 0.6963, TiRex 0.6952; no ROCKET-family comparison in the fetched part — [Mantis HTML](https://arxiv.org/html/2502.15637)
- RocketPFN paper: on 103 UCR datasets Mantis+TabPFN 0.859, MantisV2+TabPFN 0.851, MOMENT+TabPFN 0.845 vs RocketPFN 0.874–0.894; on a 92-dataset 30-resample protocol HC2 0.900, ROCKET 0.882, InceptionTime 0.880, TabPFN-on-flattened 0.847 (not directly comparable dataset sets); no low-label sweep — [arXiv 2606.21786 HTML](https://arxiv.org/html/2606.21786)
- TimEE: per-dataset ROC AUC examples (MiniRocket / MOMENT / MantisV2 / NuTime / TimEE): ArrowHead 0.978 / 0.945 / 0.940 / 0.912 / 0.975; ECG200 0.965 / 0.946 / 0.928 / 0.905 / 0.957; notes forecasting-TSFM transfer "requires non-trivial layer selection" (intermediate layers beat final layer; e.g., Chronos-2 layer 5, TiRex layer 6) — [arXiv 2607.07500 HTML](https://arxiv.org/html/2607.07500)
- NuTime evaluates 5-shot (100 episodes) TSC but against 1-NN ED/DTW, BOSS, ResNet — not ROCKET/InceptionTime — [search summary of NuTime PDF](https://arxiv.org/pdf/2310.07402)
- Bake-off context: MultiROCKET+Hydra and HIVE-COTE v2 significantly best on 112 UCR + 30 new datasets (DMKD 2024) — [arXiv 2304.13029](https://arxiv.org/pdf/2304.13029)

**Industrial / fault-diagnosis few-shot evidence beyond 2511.23177**
- Tokic et al., "TSFM in-context learning for TSC of bearing-health status" (ESANN 2026): labeled examples placed in the TSFM prompt as covariates/targets; frequency-domain reference signals converted to pseudo time series; servo-press bearing vibration; no numbers in abstract; TSFM not named on abs page — [arXiv 2511.15447](https://arxiv.org/abs/2511.15447)
- Li et al. (PolyU portal), "Exploring prior-structured low-rank adaptation for predictive maintenance based on TSFM": prior-structured adapter on frozen Timer-XL with a frequency-enhanced prior for "band-limited and impulsive signatures" and a degradation prior, condition-aware gate; bearing diagnosis + RUL; reported better few-shot data efficiency; higher F1 than vanilla Timer-XL LoRA in low-shot with slight reversal at larger k — [PolyU portal](https://research.polyu.edu.hk/en/publications/exploring-prior-structured-low-rank-adaptation-for-predictive-mai/), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1000934526000878) (full text not accessible; numbers not verified)
- UniFault (Eldele et al., arXiv 2504.01373, v2 Nov 2025): bearing-specific FD foundation model; harmonizes multivariate inputs into standardized univariate sequences; cross-domain temporal fusion; pretraining "over 6.9 million samples"; claims superior few-shot performance; weights/code release not confirmed — [arXiv 2504.01373](https://arxiv.org/abs/2504.01373)
- Label-scarce bearing preprint ("When Labels Are Scarce: An Oscillatory State Space Model for Vibration Diagnosis"): ~1 labeled second per class; Paderborn macro-F1 DualRes 48.9, MambaSL 45.7, MiniRocket 33.5; MiniRocket strongest on HUST under geometry shift — [ResearchGate](https://www.researchgate.net/publication/414687022_When_Labels_Are_Scarce_An_Oscillatory_State_Space_Model_for_Vibration_Diagnosis) (snippet-level, unreviewed)
- Chronos fine-tuned for power-grid event classification (PSML-5) with a quantum head on embeddings — [arXiv 2609.05408](https://arxiv.org/abs/2609.05408)
- Power-system forecasting benchmark (ERCOT): out-of-the-box TSFMs "not yet capable of bypassing specialized training" — [arXiv 2604.22077](https://arxiv.org/html/2604.22077v1)
- MOMENT embeddings used for few-shot RUL on aircraft engines (CMAPSS) — [ScienceDirect](https://www.sciencedirect.com/org/science/article/pii/S1526149225002085)

### Inferences
- The consistent ranking across three independent groups (Auer et al.; Mantis authors; RocketPFN authors; motor paper) is Mantis > good forecasters (TiRex/Chronos-2/Chronos-Bolt) > MOMENT as frozen extractors. The "reconstruction-based MOMENT underperforms contrastive models with little data" claim is supported by 2511.23177 and is consistent with UCR evidence, but it is confounded by model size, pretraining corpus and layer choice — a PD paper can contribute a controlled test (same head, same layer-selection protocol, label sweep).
- 2511.23177 used a random 80/20 split of windows cut from 48 recordings; windows from the same recording likely appear in both train and test, which can inflate accuracy. A PD paper should split by specimen/defect sample/recording session (analogous to the "group by inspection event" rule) — this would be a methodological improvement reviewers will notice.
- The comparison reviewers will expect and that nobody has published for industrial data: accuracy/macro-F1 vs labels-per-class (e.g., 1, 5, 10, 20, 50 shots), TSFM (frozen / LoRA / full FT) vs MiniRocket/MultiRocket-Hydra/InceptionTime, multiple seeds, grouped splits.
- In-context classifiers (TIC-FM, TimEE, RocketPFN) specifically target the low-label regime and are an emerging 2026 line; including at least one is a cheap way to be current.

### Gaps
- No paper found reporting TSFMs vs ROCKET/MiniRocket/InceptionTime/ResNet across a labels-per-class sweep (UCR or industrial).
- 2511.23177 baseline curves (CNN/LSTM/MiniRocket/PatchTST) only in figures; numbers not extractable.
- The Timer-XL PSA paper's venue, datasets and numbers could not be verified (403).
- MantisV2 paper's quantitative tables were not read (abstract only).

## Q3. Do LLM-based time-series models actually help? (Tan et al., NeurIPS 2024, and related critiques)

### Takeaway
Tan et al. showed that removing the LLM from Time-LLM, OneFitsAll/GPT4TS and CALF, or replacing it with a single attention/transformer layer, matches or beats the LLM versions on forecasting at a tiny fraction of the cost, including in 10% few-shot settings. For classification, MOMENT's benchmark already shows GPT4TS far behind (0.566 on 91 UCR under frozen-representation protocol). Reviewers now expect any LLM-based backbone to be accompanied by "w/o LLM / LLM2Attn / LLM2Trsf / random-init" ablations and compute accounting.

### Cited Findings
- Tan, Merrill, Gupta, Althoff, Hartvigsen, "Are Language Models Actually Useful for Time Series Forecasting?" — NeurIPS 2024 Spotlight; removing the LLM or replacing it with a basic attention layer "does not degrade forecasting performance" and often improves it; LLMs do not help in few-shot; do not represent sequential dependencies — [arXiv 2406.16964](https://arxiv.org/abs/2406.16964)
- Methods tested: Time-LLM (LLaMA), OneFitsAll/GPT4TS (GPT-2), CALF (GPT-2); ablations "w/o LLM", "LLM2Attn" (one randomly initialized attention layer), "LLM2Trsf" (one randomly initialized transformer block); 13 datasets (ETTh1/h2/m1/m2, Illness, Weather, Traffic, Electricity, Exchange, Covid Deaths, Taxi, NN5, FRED-MD) — [Tan et al. HTML](https://arxiv.org/html/2406.16964)
- Ablations beat Time-LLM in 26/26, CALF 22/26, OneFitsAll 19/26 (MAE+MSE); MAE across horizons 35/40, 31/40, 29/40 — [Tan et al. HTML](https://arxiv.org/html/2406.16964)
- Cost: Time-LLM ~6,642M params, 3,003 min training on Weather vs ablations ~0.245M params, ~2.17 min; inference 28.2× (Time-LLM), 2.3× (OFA), 1.2× (CALF) slower — [Tan et al. HTML](https://arxiv.org/html/2406.16964)
- Randomly initialized LLMs + fine-tuning beat pretrained LLMs 8 vs 3 times; few-shot (10%): Time-LLM vs w/o-LLM tie 8–8; CALF w/o-LLM won 10/14; input shuffling/masking does not hurt LLM methods more than ablations; classification was not tested (listed as future work); recommend simple patching+attention ("PAttn") baselines — [Tan et al. HTML](https://arxiv.org/html/2406.16964)
- MOMENT frozen-representation UCR benchmark: GPT4TS 0.566 vs MOMENT 0.794 vs TS2Vec 0.851 — [MOMENT HTML](https://arxiv.org/html/2402.03885)
- OFA/GPT4TS reported 74.00% average on 10 UEA datasets vs TimesNet 73.60% (secondary review of the original paper) — [liner review](https://liner.com/review/one-fits-all-power-general-time-series-analysis-by-pretrained); a re-run in aLLM4TS following the GPT4TS protocol averages ≈68.5% (secondary) — [aLLM4TS arXiv 2402.04852](https://arxiv.org/pdf/2402.04852)
- Counterpoint (2026): "Prompting Underestimates LLM Capability for TSC" — zero-shot prompting near chance, but linear probes on LLM hidden states raise average F1 from 0.15–0.26 to 0.61–0.67, sometimes matching specialized TS models; class-discriminative information in early layers — [arXiv 2601.03464](https://arxiv.org/abs/2601.03464)

### Inferences
- Recommended stance for a PD paper: do not use GPT4TS/Time-LLM as the main backbone. If a reviewer asks "why no LLM?", cite Tan et al. plus MOMENT's GPT4TS number. If an LLM model is included as a baseline, report: (a) w/o-LLM, LLM2Attn, LLM2Trsf, random-init variants; (b) params, GPU-hours, inference latency; (c) few-shot label sweep.
- The same ablation logic generalizes to non-LLM TSFMs and is what reviewers increasingly expect: "same architecture without pretraining" (2511.23177 did this with a random-init MOMENT Transformer), plus a frozen-vs-fine-tuned comparison.

### Gaps
- No published Tan-style ablation for classification (as opposed to forecasting) was found.
- Original OFA per-dataset UEA table not fetched directly.

## Q4. Known issues for high-frequency, impulsive, sparse signals (sampling-rate mismatch, context limits, normalization, patching) and published remedies

### Takeaway
Every classification-capable TSFM ingests short fixed windows (MOMENT 512, Mantis 512 / multiples of 32, TSPulse 512) after per-instance normalization; forecasting TSFMs allow 2k–16k contexts but were pretrained mostly on low-frequency business/IoT/weather data. For PD (ns–µs pulses at MHz–GHz sampling, whose amplitude/charge is itself diagnostic), naive resizing to 512 and instance normalization discard exactly the information that matters. Published remedies are: absolute-statistics and differencing augmentations, multi-layer embedding pooling, frequency-domain re-representation, frequency-aware adapters, sampling-rate-invariant architectures, and channel adapters — none yet validated on PD.

### Cited Findings
- MOMENT: fixed 512 input; longer series subsampled, shorter left-padded; RevIN; cannot distinguish vertically shifted series — [MOMENT HTML](https://arxiv.org/html/2402.03885)
- Mantis: inputs instance-normalized and resized (interpolated) to 512 by default, length must be multiple of 32 — [Mantis HTML](https://arxiv.org/html/2502.15637), [Mantis GitHub](https://github.com/vfeofanov/mantis); Mantis's token generator explicitly re-injects per-patch mean/std (Multi-Scaled Scalar Encoder) and a first-difference branch — [Mantis HTML](https://arxiv.org/html/2502.15637)
- TSPulse: any input length resampled to 512 for classification — [HF TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
- Context limits of forecasters: Chronos 512, Chronos-Bolt 2048, Chronos-2 8192 — [HF chronos-2](https://huggingface.co/amazon/chronos-2); TimesFM 2.5 16k — [TimesFM GitHub](https://github.com/google-research/timesfm); Timer/Sundial 2880 — [HF sundial](https://huggingface.co/thuml/sundial-base-128m); Time-MoE 4096 — [Time-MoE GitHub](https://github.com/Time-MoE/Time-MoE)
- Auer et al. excluded UCR/UEA datasets with length >2048 (55 datasets) and added "absolute sample statistics" (per-patch mean/std/min/max) "to offset the scale information lost to instance normalization", plus first-difference embeddings — [arXiv 2510.26777 HTML](https://arxiv.org/html/2510.26777v1)
- Intermediate layers outperform final-layer embeddings for forecasting TSFMs; layer selection is non-trivial — [TimEE HTML](https://arxiv.org/html/2607.07500); Mantis recommends layer 2 (layer 1 for Mantis+) — [Mantis GitHub](https://github.com/vfeofanov/mantis)
- Pretraining corpora are dominated by non-industrial/low-rate data: TimesFM 2.5 (GiftEval pretrain, Wikipedia pageviews, Google Trends, synthetic) — [HF TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch); Toto (~1T observability metrics) — [HF Toto](https://huggingface.co/Datadog/Toto-Open-Base-1.0); TSPulse (electricity, weather, solar, wind, traffic) — [HF TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1); Mantis+/V2 on synthetic CauKer — [Mantis GitHub](https://github.com/vfeofanov/mantis)
- "How Foundational are Foundation Models for Time Series Forecasting?": zero-shot generalization "tightly coupled with the distribution seen during pretraining" — [arXiv 2510.00742](https://arxiv.org/html/2510.00742v3)
- Long-context TSFM study: most approaches downsample long series and treat channels independently, limiting high-frequency and cross-channel modeling — [arXiv 2409.13530](https://arxiv.org/html/2409.13530v1)
- Industrial benchmark FactoryBench anti-alias filtered and resampled 83–500 Hz recordings to 10 Hz to fit model budgets, explicitly removing high-frequency content — [arXiv 2605.07675](https://arxiv.org/pdf/2605.07675)
- Noisy-periodic study: TSFM forecasting deteriorates with longer periods, higher noise, lower sampling rates; 512-step windows can miss a full cycle — [ResearchGate](https://www.researchgate.net/publication/387670393_Evaluating_Time_Series_Foundation_Models_on_Noisy_Periodic_Time_Series)
- Remedy: sampling-rate-invariant FlowState (SSM encoder + functional basis decoder; forecasting) — [arXiv 2508.05287](https://arxiv.org/pdf/2508.05287), [IBM Research](https://research.ibm.com/publications/flowstate-sampling-rate-invariant-time-series-foundation-model-with-dynamic-forecasting-horizons)
- Remedy: frequency-enhanced prior adapter for "band-limited and impulsive signatures" on frozen Timer-XL — [PolyU portal](https://research.polyu.edu.hk/en/publications/exploring-prior-structured-low-rank-adaptation-for-predictive-mai/)
- Remedy: convert frequency-domain references into pseudo time series for TSFM in-context classification — [arXiv 2511.15447](https://arxiv.org/abs/2511.15447)
- Remedy: FreqCondNorm, frequency-conditioned transformer FM for cross-domain predictive maintenance (only reference list seen) — [arXiv 2609.20535](https://arxiv.org/pdf/2609.20535)
- Remedy: channel adapters (PCA projector, learnable LinearChannelCombiner) — [Mantis GitHub](https://github.com/vfeofanov/mantis); per-variate embedding concatenation beat mean/max pooling — [Auer et al.](https://arxiv.org/html/2510.26777v1)
- Practice in 2511.23177: downsample current 2000→512 Hz to align with vibration, fixed 512-step windows; normalization not described — [HTML](https://arxiv.org/html/2511.23177)
- TF-C bearing benchmark (FD-A→FD-B) uses 64 kHz signals with 5,120-sample windows — [TF-C GitHub](https://github.com/mims-harvard/TFC-pretraining) — i.e., self-supervised (non-foundation) methods already handle longer high-rate windows than 512-input TSFMs.

### Inferences (PD-specific design advice; not validated in literature)
- Decide the "unit" fed to the TSFM explicitly and justify it: (a) single pulse waveform cropped around the peak (choose sampling so that the pulse + ringing fits 512 samples, rather than resizing a long record); (b) phase-resolved representation (e.g., 512 phase bins of a PRPD histogram or per-cycle max amplitude, which already is a multiple of 32); (c) multi-channel (UHF/HFCT/AE) handled with Mantis channel adapters or per-channel concatenation.
- Because instance normalization removes absolute amplitude (pC/mV), concatenate absolute per-patch statistics (Auer et al.) or append scalar features to embeddings; report an ablation with/without them.
- Report an explicit "resampling ablation" (interpolate-to-512 vs crop vs stride-pool) — this directly addresses a reviewer question about sampling-rate mismatch.
- Use multi-layer pooled embeddings with layer choice fixed on validation data, and state the protocol to avoid TIC-FM's critique of classifier-dependent evaluation.

### Gaps
- No published evaluation of TSFMs on MHz–GHz impulsive signals (PD, arcing, lightning) was found.
- No published comparison of resampling strategies for TSFM classification of transient pulses.

## Q5. Has anyone applied TSFMs to partial discharge or circuit-breaker data (2024–2026)?

### Takeaway
No. Across several search formulations, no 2024–2026 paper applying a general TSFM (MOMENT, Chronos, TimesFM, Moirai, Mantis, Timer, etc.) to partial-discharge or circuit-breaker data was found. Adjacent work exists: self-supervised / masked-autoencoder PD models trained from scratch on PRPD, few-shot metric-learning for HV circuit breakers, and Chronos for grid-event classification. This makes the PD + TSFM combination a genuine novelty claim (state it as "to our knowledge").

### Cited Findings
- KRMNet (Applied Intelligence, 2025): masked-autoencoder Transformer pretrained on unlabeled PRPD, then fine-tuned; 88.5% (noisy) / 90.2% (clean) accuracy — [Springer](https://link.springer.com/article/10.1007/s10489-025-06899-z)
- Self-supervised asynchronous federated learning for GIS PD diagnosis (Energies 18(12), 2025) — [RePEc/IDEAS](https://ideas.repec.org/a/gam/jeners/v18y2025i12p3078-d1676332.html)
- Self-supervised representation learning for GIS PD condition assessment (multi-task: diagnosis, localization, severity) — [Springer chapter](https://link.springer.com/chapter/10.1007/978-981-96-4063-8_32)
- Single vs mixed PD classification under switching voltage, AWA-CNN (arXiv 2605.21352; method not verified beyond title) — [arXiv 2605.21352](https://arxiv.org/pdf/2605.21352)
- PD deep-learning survey (Energies 2019; older) — [MDPI](https://www.mdpi.com/1996-1073/12/13/2485)
- HV circuit breaker few-shot cross-domain diagnosis, ITFGMN (Control Engineering Practice, 2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0967066125003399)
- SF6 HVCB shell-vibration fault diagnosis (IET GTD, 2026) — [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.70267)
- HVCB multimodal fusion diagnosis — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419656/)
- Closest TSFM-in-power-equipment classification: Chronos fine-tuned for PSML-5 grid events with a quantum head — [arXiv 2609.05408](https://arxiv.org/abs/2609.05408)

### Inferences
- The novelty positioning should be "first systematic evaluation of TSFMs for few-shot PD classification", not "first FM for PD" (KRMNet-style self-supervised PD models exist and must be cited as the domain-specific pretraining alternative).
- A from-scratch self-supervised PD encoder (MAE/contrastive on unlabeled PD) is a natural extra baseline: it tests whether generic TSFM pretraining beats in-domain self-supervision.

### Gaps
- IEEE Xplore (TDEI, TPWRD, Measurement, EPSR, MST) is poorly indexed by the search tool; a direct IEEE Xplore / CNKI search ("局部放电 基础模型 / 大模型 / 预训练") is recommended to confirm the negative result.

## Q6. Mandatory strong non-foundation baselines for small-data TSC and their implementations

### Takeaway
Reviewers will expect at minimum: MiniRocket (and/or MultiRocket+Hydra), ROCKET, InceptionTime, ResNet/FCN (all in aeon), 1-NN DTW, a feature-based baseline (Catch22 or QUANT), and self-supervised representation learners TS2Vec, TF-C and SoftCLT — plus PD-domain baselines (hand-crafted PD features + RF/SVM, CNN on PRPD) and a "same-architecture-without-pretraining" control. HIVE-COTE 2 is the accuracy ceiling but expensive.

### Cited Findings
- aeon classification API includes RocketClassifier, MiniRocketClassifier, MultiRocketClassifier, HydraClassifier, MultiRocketHydraClassifier, Arsenal; InceptionTimeClassifier, ResNetClassifier, FCNClassifier, LITETimeClassifier, TimeCNNClassifier, MLPClassifier etc.; HIVECOTEV2; Catch22Classifier, FreshPRINCEClassifier, TSFreshClassifier; QUANTClassifier (interval-based) — [aeon docs](https://www.aeon-toolkit.org/en/stable/api_reference/classification.html)
- Bake off redux (Middlehurst, Schäfer, Bagnall; DMKD 2024): MultiROCKET+Hydra and HIVE-COTE v2 significantly better than others; reproducible via tsml-eval + aeon — [arXiv 2304.13029](https://arxiv.org/pdf/2304.13029), [tsml-eval page](https://tsml-eval.readthedocs.io/en/stable/publications/2023/tsc_bakeoff/tsc_bakeoff_2023.html)
- ROCKET: competitive with HIVE-COTE, TS-CHIEF, InceptionTime at a fraction of compute — [arXiv 1910.13051](https://ar5iv.labs.arxiv.org/html/1910.13051); MultiRocket — [DMKD](https://link.springer.com/article/10.1007/s10618-022-00844-1); ROCKET standout in multivariate bake-off — [DMKD 2021](https://link.springer.com/article/10.1007/s10618-020-00727-3)
- TS2Vec (AAAI 2022, arXiv 2106.10466), MIT license, input n_instances×n_timestamps×n_features, UCR/UEA loaders, default repr dim 320 — [TS2Vec GitHub](https://github.com/zhihanyue/ts2vec)
- TF-C (Zhang, Zhao, Tsiligkaridis, Zitnik; NeurIPS 2022): time- and frequency-encoders with time–frequency consistency; transfer scenarios include FD-A→FD-B bearing fault (64 kHz, 5,120-sample windows, FD-B 60 train / 21 val / 13,559 test); MIT code license (FD data CC BY-NC 4.0) — [TF-C GitHub](https://github.com/mims-harvard/TFC-pretraining)
- SoftCLT (Lee, Park, Lee; ICLR 2024): soft contrastive learning extending TS2Vec and TS-TCC/CA-TCC; code for UCR-128/UEA-30 classification, semi-supervised 1%/5% labels, SleepEEG→Epilepsy/FD-B/Gesture/EMG transfer; license not stated — [SoftCLT GitHub](https://github.com/seunghan96/softclt)
- MOMENT-paper reference baselines for frozen-representation TSC: TS2Vec, T-Loss, TS-TCC, TNC, DTW, ResNet, FCN — [MOMENT HTML](https://arxiv.org/html/2402.03885)
- Tan et al. recommend patching + attention ("PAttn") simple baselines — [Tan et al. HTML](https://arxiv.org/html/2406.16964)
- 2511.23177 used raw-signal SVM, CNN, LSTM, non-pretrained Transformer, TSSequencerPlus, MiniRocket, TSPerceiver, XCM, PatchTST — [HTML](https://arxiv.org/html/2511.23177)

### Inferences
- Minimal defensible baseline table for a few-shot PD paper:
  - Classical/kernel (aeon, CPU, seconds–minutes): MiniRocket, MultiRocket+Hydra, ROCKET, 1-NN DTW, Catch22 or QUANT; optionally HC2 on the smallest label budgets only.
  - Supervised deep (aeon or own PyTorch): InceptionTime, ResNet, FCN (and/or LITETime).
  - Self-supervised pretrain-then-finetune: TS2Vec, TF-C (its FD bearing scenario is a good precedent), SoftCLT — pretrain on the unlabeled PD pool, so the comparison with generic TSFMs is "in-domain SSL vs out-of-domain foundation pretraining".
  - PD domain: statistical PRPD features + RF/SVM; 2D-CNN on PRPD image.
  - Controls: same backbone randomly initialized; frozen vs LoRA vs full FT; with/without absolute-statistics augmentation.
- With CSV data, aeon/sktime expect numpy arrays shaped (n_cases, n_channels, n_timepoints); Mantis takes numpy; MOMENT takes torch tensors (batch, channels, 512) — one loader can serve all (engineering inference).
- Report macro-F1 + balanced accuracy, mean ± std over ≥5 seeds and ≥3 grouped splits, and per-shot curves; Mantis' authors and TIC-FM both emphasize protocol sensitivity, so fixing the classifier head across backbones matters.

### Gaps
- No single source benchmarks all these baselines against TSFMs in the low-label regime; the PD paper will need to run it.
- aeon version at time of writing not shown on the docs page; pin the version in the paper.
