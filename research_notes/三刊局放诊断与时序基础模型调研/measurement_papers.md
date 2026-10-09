# Papers in *Measurement* (Elsevier/IMEKO, ISSN 0263-2241) on PD recognition, HVCB mechanical fault diagnosis, and few-shot/pretrained-model fault diagnosis (2021–2026)

Notes compiled 2026-10-09 for a planned paper, "Phase-anchored adaptation of pretrained time-series foundation models for few-shot partial discharge pattern recognition".

**How venues were checked.** A paper counts as a *Measurement* paper here only if its DOI begins `10.1016/j.measurement.` That prefix was confirmed through the Crossref REST API at `https://api.crossref.org/works/<DOI>`, which also supplied the volume, article number, author list, reference count and Crossref citation count. Abstract-level details (inputs, models, accuracies) come from search-engine extracts of each ScienceDirect abstract page or from citing papers, because direct fetches were blocked. ScienceDirect returned HTTP 403 and the Elsevier API asked for an API key. As a result, dataset sizes, data splits and baseline lists could not be read from the full texts. They are marked "not found" unless an abstract stated them.

## Q1. Which *Measurement* papers address PD recognition (GIS, cable, transformer, switchgear; UHF/HFCT/IEC 60270; PRPD/PRPS/time-domain pulse inputs), and what are their details?

### Takeaway
*Measurement* has published a steady and growing number of deep-learning PD recognition papers. Crossref returns more than 20 PD classification or recognition papers from 2021–2026 and about 10 more on PD denoising. Two parts of this set are closest to a few-shot paper:
- A Xi'an-area GIS group (Jing Yan, Yanxin Wang, Yingsan Geng, Jianhua Wang) contributes transfer, small-sample and zero-shot papers. The affiliation is my inference and was not verified.
- A cable-terminal group (Kai Liu, Guangning Wu) contributes 2026 papers on self-supervised learning and diffusion-based augmentation.

The search found no *Measurement* PD paper that uses a pretrained time-series foundation model. Most papers that state a number report accuracy of roughly 90–100% on their own laboratory data, and the train/test split protocol was rarely visible in the abstracts.

### Cited Findings

#### A. Most relevant: small-sample, transfer, zero-shot, self-supervised and augmentation PD papers
| # | Citation (Measurement vol., art. no., DOI) | Asset / sensor / input | Model | Data / split | Reported result | Source |
|---|---|---|---|---|---|---|
| 1 | Q. Jing, J. Yan, Y. Wang, J. Huang, H. Xiao, R. Ding, J. Wang, Y. Geng, "A novel small-sample diagnosis method for GIS partial discharge by transfer robust support matrix machine," *Measurement* 255 (2025) 118054, doi:10.1016/j.measurement.2025.118054 | GIS. PD samples kept as **matrices**. Sensor type not confirmed (abstract extracts do not say UHF) | Support matrix machine with a transfer-learning objective and a constraint term. Noise suppression splits each PD matrix into a low-rank clean part and a sparse noise part | "Two datasets" in one extract and "a field dataset" in another. Sample counts not found | **Sources conflict.** One extract gives 96.91% and 91.28% on two datasets, which are +3.16 and +7.75 points over the best baseline. Another gives 97.63% on a field dataset | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224125014137); [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2025.118054) |
| 2 | Y. Wang, J. Yan, Z. Wang, D. Zhao, R. He, J. Wang, Y. Geng, "Multi-source partial discharge diagnosis in gas-insulated switchgear via zero-shot learning," *Measurement* 217 (2023) 113033, doi:10.1016/j.measurement.2023.113033 | GIS. Authors built a multi-source PD experiment in GIS | Attention residual network extracts features. A semantic descriptor gives attributes, which a fully connected network maps into the visual space. A relation network scores similarity | Trained only on **single-source** PD, tested on multi-source PD | 90.25% | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224123005973) |
| 3 | J. Li, J. Tian, A. Banerjee, X. Zhai, S. Wang, "Gradient balanced selective mixture-of-experts for gas insulated switchgear partial discharge diagnosis," *Measurement* 256 (2025) 118139, doi:10.1016/j.measurement.2025.118139 | GIS. **Phase-wise fusion** of acoustic emission and UHF **phase-resolved pulse sequence (PRPS)** data | GB-SMoE: a multi-task selective mixture of experts with gradient-balanced task weighting. It does pattern recognition and severity assessment together | Not found | 94.97% classification; 91.31% severity | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224125014988) |
| 4 | K. Liu, S. Zhou, G. Nie, R. Yi, B. Cao, B. Gao, G. Wu, "Cable terminal partial discharge signal recognition method based on multi-scale self-supervision," *Measurement* 282 (2026) 121967, doi:10.1016/j.measurement.2026.121967 | Cable terminal | Multi-scale self-supervised learning. This is the closest PD analogue to pretraining in *Measurement* | Not found; search engines have not indexed the abstract | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.121967) |
| 5 | T. Zhang, K. Liu, G. Nie, G. Gao, G. Wu, "Application of an improved diffusion model-based data augmentation method in partial discharge pattern recognition for vehicle cable terminals," *Measurement* 268 (2026) 120672, doi:10.1016/j.measurement.2026.120672 | Vehicle (train) cable terminals | Diffusion-model data augmentation | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.120672) |
| 6 | Y. Liu, J. Peng, B. Wang, X. Song, "A transformer partial discharge detection method based on MambaGAN-driven data augmentation," *Measurement* 262 (2026) 120087, doi:10.1016/j.measurement.2025.120087 | Power transformer. PD sensor experimental platform. Stationary wavelet transform denoising, then 2-D image conversion | MambaGAN (CPSSMamba module) for augmentation | **800 samples: 4 types × 200** (tip, floating, air-gap, surface discharge) | Lower FID and higher IS and recognition accuracy than other augmentation methods. Accuracy value not found | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224125034463) |
| 7 | X. Jiang, C. Zhang, Z. Zhou, X. Liao, J. Fulneček, L. Yang, "Partial discharge data augmentation based on wavelet coefficients and variational autoencoder," *Measurement* 258 (2026) 119219, doi:10.1016/j.measurement.2025.119219 | Time-domain PD signals (wavelet coefficients) | VAE augmentation | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2025.119219) |

#### B. PD recognition: representation and architecture papers
| # | Citation | Input / asset | Model | Data / split | Result | Source |
|---|---|---|---|---|---|---|
| 8 | Y. Deng, Q. Xie, J. Chen, D. Tan, H. Liu, "KSMNet: Learn the core discriminant features and semantic relationships for partial discharge pattern recognition via Mamba," *Measurement* 246 (2025) 116683, doi:10.1016/j.measurement.2025.116683 | **PRPD** patterns | Fusion scanning perception network, an SS2D Mamba module (SSVM) and a Kolmogorov–Arnold head (TKAN) | Two PRPD datasets, **WHTPD** and **PDMDE** | Validation accuracy above 89% and training accuracy above 90%; also tested on noisy PD | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224125000429) |
| 9 | K. Sun, R. Li, L. Zhao, Z. Li, "Feature extraction based on time-series topological analysis for the partial discharge pattern recognition of high-voltage power cables," *Measurement* 217 (2023) 113009, doi:10.1016/j.measurement.2023.113009 | HV cable. **PD time series** reconstructed as a phase-space point cloud, with the time delay set by symbolic entropy plus PSO | Persistent-homology Betti curves fed to an optimized 1D-CNN | 4 typical PD patterns; counts not found | TDA features raise 1D-CNN overall accuracy by 11.25% | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224123005730) |
| 10 | Q. Jing, J. Yan, Y. Wang, R. He, L. Lu, "A novel differentiable neural network architecture automatic search method for GIS partial discharge pattern recognition," *Measurement* 195 (2022) 111154, doi:10.1016/j.measurement.2022.111154 | GIS | Differentiable architecture search over a layered, factorized CNN space with Gumbel-softmax relaxation | Not found | 97.625%. Claims noise robustness and tolerance of imbalanced data | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224122004122) |
| 11 | W.J.K. Raymond, C.W. Xin, L.W. Kin, H.A. Illias, "Noise invariant partial discharge classification based on convolutional neural network," *Measurement* 177 (2021) 109220, doi:10.1016/j.measurement.2021.109220 | Laboratory PD data, with noise overlaid at test time | CNN with **transfer learning** | **Modified 10-fold CV: trained on clean PD, tested on noise-overlapped PD** | Up to 16.90% higher accuracy under noise than traditional ML with manual features | [Semantic Scholar abstract (API)](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.measurement.2021.109220?fields=abstract) |
| 12 | R. Sahoo, S. Karmakar, "Comparative analysis of machine learning and deep learning techniques on classification of artificially created partial discharge signal," *Measurement* 235 (2024) 114947, doi:10.1016/j.measurement.2024.114947 | Artificially created corona, surface and internal PD. Denoised with a sym4 wavelet. CNN input is **CWT scalograms**; ML input is statistical features | CNN vs SVM / ANN / KNN / RF | Not found | CNN up to 100%; SVM 94.26%, ANN 94.26%, KNN 91%, RF 95.1% | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224124008327) |
| 13 | C. Zhang, J. Fulneček, L. Yang, Y. Zhang, J. Zheng, "Combining multi-level feature extraction algorithm with residual graph convolutional neural network for partial discharge detection," *Measurement* 242 (2025) 116151, doi:10.1016/j.measurement.2024.116151 | **Real measured data from medium-voltage overhead lines** (time-domain signals). Wavelet noise analysis with Bayesian optimization | 1D-CNN + ResGCN | Use of the VSB dataset not confirmed | 97.3% accuracy; 96.1% PD identification rate | [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224124020360) |
| 14 | X. Liang, B. Xu, W. Cao, F. Guo, "A PGS-LSTM-attention model for imbalanced partial discharge detection based on SVM," *Measurement* 256 (2025) 118223, doi:10.1016/j.measurement.2025.118223 | Imbalanced PD detection | LSTM-attention + SVM | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2025.118223) |
| 15 | A. Mansouri, K. Kessairi, I. Iannarelli, M. Hendel, A. Cavallini, "Two-stage autoencoder-based feature extraction with weighted fusion for partial discharge classification," *Measurement* 284 (2026) 122234, doi:10.1016/j.measurement.2026.122234 | PD classification | Two-stage autoencoder with weighted fusion | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.122234) |
| 16 | C. Wang, H. Shi, L. Cheng, L. Yang, "Metallic particle defect identification in impulse-voltage partial discharge signals based on background-aware residual learning," *Measurement* 292 (2026) 123179, doi:10.1016/j.measurement.2026.123179 | Impulse-voltage PD (metallic particle) | Background-aware residual learning | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.123179) |
| 17 | T. Gao, M. Sui, J. Yang, Z. Yang, L. Yao, Q. Guo, D. Kong, "A neural network-based partial discharge pattern recognition method integrated with Phasor Particle Swarm Optimization," *Measurement* 290 (2026) 122748, doi:10.1016/j.measurement.2026.122748 | PD patterns | NN + Phasor PSO | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.122748) |
| 18 | L. Pradeep, N. Haque, P. Preetha, "Discrimination of multiple partial discharge sources in oil impregnated pressboard insulation using HFCT sensor & ensembled learning classifiers," *Measurement* 254 (2025) 117920, doi:10.1016/j.measurement.2025.117920 | **HFCT**, oil-impregnated pressboard, multiple sources | Ensemble classifiers | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2025.117920) |
| 19 | L. Tharamal, S. Surlekar, P. Preetha, N. Haque, "A new method for incipient fault diagnosis of power transformers based on image processing of phase resolved partial discharge pattern of transformer oil," *Measurement* 258 (2026) 119536, doi:10.1016/j.measurement.2025.119536 | **PRPD** images, transformer oil | Image processing | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2025.119536) |
| 20 | L.F. Freitas-Gutierres et al. (12 authors), "Advancing substation inspection: The Hilbert–Huang transform approach for partial discharge recognition and assessment," *Measurement* 247 (2025) 116846, doi:10.1016/j.measurement.2025.116846 | Substation inspection | HHT features | Not found | Not found | [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2025.116846) |

#### C. PD measurement-science papers useful for a "phase-anchored" argument
- M. Florkowski, "Influence of harmonics on partial discharge measurements and interpretation of phase-resolved patterns," *Measurement* 196 (2022) 111198, doi:10.1016/j.measurement.2022.111198. It shows that the phase reference itself is a measurement concern, which supports a phase-anchoring design — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01)
- M. Florkowski, "Autonomous tracking of partial discharge pattern evolution based on optical flow," *Measurement* 179 (2021) 109513, doi:10.1016/j.measurement.2021.109513 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01)
- M. Florkowski, "Measurement of partial discharge echo and extraction of decay attributes," *Measurement* 238 (2024) 115338. Also M. Florkowski, M. Kuniewski, P. Mikrut, "Measurement and extraction of partial discharge power losses at high-voltage containing harmonics," *Measurement* 269 (2026) 120789 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01)
- S. Govindarajan, A. Morales, J.A. Ardila-Rey, N. Purushothaman, "A review on partial discharge diagnosis in cables: Theory, techniques, and trends," *Measurement* 216 (2023) 112882, doi:10.1016/j.measurement.2023.112882 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01)
- C. Mier, A. Rodrigo Mor, P. Vaessen, "A directional coupler for partial discharge measurements in gas-insulated substations," *Measurement* 225 (2024) 113996 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01)

#### D. PD denoising papers in *Measurement* (for the preprocessing or related-work paragraph)
All verified through the [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01):
- J. Huang, Y. Wang, J. Yan et al., multi-domain sparse-representation K-SVD denoising of GIS PD, 272 (2026) 121094
- C. Liu, W. Li, L. Yang, X. Xiao, J. Yu, PD noise suppression with convolutional and fully connected feedforward NNs, 281 (2026) 121894
- H. Zhou et al., dual-domain sparse collaboration denoising, 289 (2026) 122629
- N.H. Fauzan, Y.-C. Huang, C.-C. Kuo, multi-stage cascaded hybrid denoising, 288 (2026) 122518
- Z. Liu, X. Han, H. Ding, J. Li, noise suppression for composite pulse signals of PD, 289 (2026) 122660
- H. Jin, C. Zhang, H. Zhang, SVD + RIME-VMD for transformer PD, 259 (2026) 119670
- H. Jin et al., JADE blind source separation, 244 (2025) 116552
- S. Chaudhuri et al., TVD + autoencoder, 223 (2023) 113674
- A.A. Soltani, A. El-Hag, RBF-NN denoising, 172 (2021) 108970

#### E. Non-*Measurement* PD papers that came up (labeled by venue)
- **Measurement Science and Technology (IOP; DOI prefix 10.1088/1361-6501), not *Measurement*.** All from the [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=partial+discharge+few-shot+small+sample&filter=from-pub-date:2021-01-01):
  - Y. Wang, J. Yan, Z. Yang, J. Wang, Y. Geng, "A novel 1DCNN and domain adversarial transfer strategy for small sample GIS partial discharge pattern recognition," MST 32 (2021), doi:10.1088/1361-6501/ac27e8
  - Y. Wang et al., federated deep learning for GIS PD, MST 33, doi:10.1088/1361-6501/ac7a09
  - Y. Wang et al., multi-source domain adaptation for PD severity in GIS, MST 35, doi:10.1088/1361-6501/ad7488
  - J. Yan et al., domain-alignment multitask learning with digital twin, MST 35, doi:10.1088/1361-6501/ad3412
  - Y. Liu et al., cross-attention modal fusion for GIS PD, MST 36, doi:10.1088/1361-6501/adfb00
  - S. Govindarajan et al., deep-learning PD review, MST 37, doi:10.1088/1361-6501/ae6a0c
- **ISA Transactions.** Metric-based meta-learning for few-shot GIS PD: 93.17% at 4-way 5-shot — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0019057822004128)
- **Sensors (MDPI).** One-shot learning with a UHF sensor in GIS: 98.65% for 4 fault types plus noise — [PMC7582290](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7582290/)

### Inferences
- Six papers form the citable core for a few-shot PD paper in this journal:
  - #1 (transfer + small-sample)
  - #2 (zero-shot, data-scarcity framing)
  - #4 (self-supervised pretraining)
  - #5–#7 (augmentation as the competing small-data route)
  - #11 (transfer learning plus a noise-shift evaluation protocol)
- #3 (GB-SMoE), with its "phase-wise fusion" of PRPS data, is the closest *Measurement* precedent for using explicit phase structure. It should be discussed directly against a "phase-anchored" design.
- #9 (Betti curves from phase-space reconstruction) and #13 (time-domain PD with 1D-CNN/ResGCN) show that *Measurement* accepts time-series PD inputs as well as PRPD images. That suits CSV time-series data.
- Authors in the Yan / Wang / Geng / Jing group appear repeatedly in both *Measurement* and MST on GIS PD with little data. They are a likely reviewer pool, so citing #1, #2 and #10, plus the MST transfer and federated papers, is prudent. The reviewer-pool point is my inference.
- The search found no *Measurement* PD paper that uses MOMENT, Chronos, TimesFM, Moirai, Mantis, GPT4TS or Time-LLM, which leaves a novelty gap for the planned paper (title-level search only; see Gaps).

### Gaps
- Full texts could not be read, so most dataset sizes, specimen counts, split methods (random, by specimen or by session) and baseline lists are missing. MambaGAN (#6) is the exception, with 800 samples stated.
- The TRSMM accuracy figures conflict between two search extracts (96.91%/91.28% vs 97.63%). The abstract should be checked before citing.
- The abstracts of the 2026 papers #4, #5, #7 and #14–#17 have not been indexed by search engines. Their methods and results could only be inferred from their titles.
- The sensor type for #1 and #10 (UHF or not) was not confirmed.
- The search was over Crossref titles and bibliographic fields, not full text. PD papers whose titles avoid "partial discharge" (for example "insulation defect") may be missing.

## Q2. Which *Measurement* papers address HV circuit breaker mechanical fault diagnosis, especially few-shot?

### Takeaway
Only one *Measurement* paper in 2021–2026 is explicitly about few-shot HVCB mechanical fault diagnosis: Ye et al. 2022, a U-Net with a capsule network. Two other *Measurement* papers cover HVCB or operating-mechanism diagnosis by multi-sensor fusion, and a few related switching-device papers appear (OLTC, vacuum contactor). The strongest few-shot HVCB work is in other journals: IEEE TIM, ESWA, Control Engineering Practice and IET GTD.

### Cited Findings
- **X. Ye, J. Yan, Y. Wang, J. Wang, Y. Geng, "A novel U-Net and capsule network for few-shot high-voltage circuit breaker mechanical fault diagnosis," *Measurement* 199 (2022) 111527, doi:10.1016/j.measurement.2022.111527.** Crossref records 29 references and 40 citations. Capsule networks are placed in the U-Net's contracting and expanding paths to reduce pooling feature loss, and dynamic routing fuses higher-level capsules with lower layers. Citing papers describe high precision in few-shot settings. The accuracy value and samples per class were **not found** — [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2022.111527); [PMC citing summary](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419656/); [PeerJ reference list](https://peerj.com/articles/cs-2248/)
- **J. Zhang, Y. Wu, Z. Xu, Z. Din, H. Chen, "Fault diagnosis of high voltage circuit breaker based on multi-sensor information fusion with training weights," *Measurement* 192 (2022) 110894, doi:10.1016/j.measurement.2022.110894.** Crossref records 54 references and 61 citations. Three sensors: two mechanical-vibration and one actuator-current. Dempster–Shafer evidence fusion uses sensor weights learned by a BP network instead of empirical or entropy weights, to resolve conflicting evidence. Accuracy not found — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224122001786); [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2022.110894)
- **Z. Yang, K. Yuan, J. Tan, Y. Xing, Q. Huang, D. Cai, "A multisource feature fusion approach for intelligent diagnosis of circuit breaker operating mechanism states," *Measurement* 290 (2026) 122969, doi:10.1016/j.measurement.2026.122969.** Abstract not indexed — [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.122969)
- Related switching or tap-changing devices in *Measurement* — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=circuit+breaker&filter=from-pub-date:2021-01-01):
  - R. Qi, T. Zhao, B. Wei, B. Peng, Z. Chen, X. Wang, "Mechanical fault diagnosis of on-load tap changers using time–frequency vibration analysis and a lightweight YOLO model," *Measurement* 256 (2025) 118441 (Crossref: 69 references)
  - X. Zou et al., "Dynamic fault-relevant component analysis for fault diagnosis of vacuum contactor," *Measurement* 265 (2026) 120309
- **Not *Measurement*:**
  - Yan et al., Transformer-CNN with metric meta-learning for few-shot HVCB, **IEEE TIM** 2023, >95% accuracy — [PeerJ reference list](https://peerj.com/articles/cs-2248/)
  - Yang & Liao, zero-shot HVCB compound faults, **Expert Systems with Applications** 2024, 86.2% average accuracy — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417423036370)
  - Few-shot cross-domain HVCB, **Control Engineering Practice** 2025 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0967066125003399)
  - Newton–Raphson-optimised Transformer with meta-transfer learning, **IET GTD** 2026, 98.45% accuracy / 98.26% F1 at 5-way 5-shot — [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.70252?af=R)
  - Small-sample spring-mechanism diagnosis by stacking (220 kV breaker; MEMS pressure and travel, coil and motor current), **Sensors** 2026, 96.1% — [MDPI](https://www.mdpi.com/1424-8220/26/5/1485)

### Inferences
- In *Measurement*, HVCB few-shot work rests on Ye et al. 2022, the same Yan/Geng group as the GIS PD papers. A PD-focused paper can cite it as evidence that the journal accepts few-shot diagnosis of power-equipment time series.
- Zhang et al. 2022 is a good model of how a *Measurement* paper foregrounds the sensing setup (how many sensors, what each measures) before the learning part.

### Gaps
- No *Measurement* paper on HVCB **travel curves** or **coil current alone** with deep or few-shot learning appeared in the Crossref title searches.
- Accuracy and dataset size for Ye 2022 and Zhang 2022 were not found because the full texts were inaccessible.

## Q3. Which *Measurement* papers use few-shot, transfer, meta-learning, self-supervised pretraining, large pretrained models or time-series foundation models for fault diagnosis?

### Takeaway
*Measurement* publishes many few-shot, meta-learning and self-supervised fault-diagnosis papers, mostly on bearing and gearbox vibration. A smaller set targets power equipment: transformers, distribution systems and switch machines. LLM-based diagnosis and prognosis papers started appearing in 2026.

Title-level Crossref searches for MOMENT, Chronos, TimesFM, Moirai, Mantis, GPT4TS/OFA and Time-LLM found **no *Measurement* paper that applies a pretrained time-series foundation model to fault diagnosis** as of 2026-10-09. The closest TSFM-for-diagnosis evidence is an arXiv preprint outside the journal.

### Cited Findings
**Power and electrical equipment, in *Measurement*:**
- X. Wu, A. Jiang, S. Song, S. Zhang, "A fault feature enhancement model-agnostic meta-learning approach for incipient fault diagnosis in distribution systems," *Measurement* 279 (2026) 121790, doi:10.1016/j.measurement.2026.121790. Abstract not indexed — [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.121790)
- Z. Zhang, Y. Wang, J. Liu, "Diagnosis model of noise-type defects for dry-type transformer based on time–frequency–space tensors and improved prototypical network under small sample conditions," *Measurement* 221 (2023) 113450, doi:10.1016/j.measurement.2023.113450. Acoustic signals → parameter-optimized VMD + HHT time-frequency tensors. Virtual rotation of the sound source increases the sample count eightfold. Hyper-PCA reduces channels from 64 to 11. ResNet encoder in a prototypical network. Average precision **96.0% ± 2.1% with 15 samples per defect type** — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S026322412301014X)
- W. Wang, C. Xu, Y. Zhao, Q. Sun, M. Jiang, F. Zhang, "Power transformer cross-domain fault diagnosis method based on dynamic weight balancing of multidimensional indicators and deep adversarial networks," *Measurement* 258 (2026) 119425. Also W. Wang, F. Liu, X. Lv, F. Zhang, M. Jiang, Y. Jia, "Multi-source structural fusion and synchronous heterogeneous network for power transformers partial domain adaptation fault diagnosis," *Measurement* 258 (2026) 119319 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=power+transformer+fault+diagnosis+transfer+learning&filter=from-pub-date:2021-01-01)
- Y. He, D. He, Z. Lao, Z. Yao, H. Sun, C. He, Z. Yuan, "A class-center fine-tuning prototypical network for few-shot fault diagnosis of turnout switch machine driven by multi-source signals," *Measurement* 242 (2025) 115920 (Crossref: 23 citations) — [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2024.115920)
- S. Chen, H. Ge, H. Li, Y. Sun, X. Qian, "Hierarchical deep convolution neural networks based on transfer learning for transformer rectifier unit fault diagnosis," *Measurement* 167 (2021) 108257 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=power+transformer+fault+diagnosis+transfer+learning&filter=from-pub-date:2021-01-01)
- X. Chen, Z. Chen, "Cross-domain fault diagnosis of induction motor based on an improved unsupervised GAN and fine-tuning under limited labeled data," *Measurement* 249 (2025) 116988 — [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=pre-trained+model+fault+diagnosis+transfer&filter=from-pub-date:2021-01-01)

**Few-shot and meta-learning on general time-series signals, in *Measurement*.** These are highly cited and suitable as method baselines. Citation counts are from Crossref, 2026-10:
- T. Yang, T. Tang, J. Wang, C. Qiu, M. Chen, "A novel cross-domain fault diagnosis method based on model agnostic meta-learning," 199 (2022) 111564 (75 citations)
- Z. Liu, Z. Peng, "Few-shot bearing fault diagnosis by semi-supervised meta-learning with graph convolutional neural network under variable working conditions," 240 (2025) 115402 (63 citations)
- T. Tang et al., "A novel lightweight relation network for cross-domain few-shot fault diagnosis," 213 (2023) 112697 (36 citations)
- T. Tang et al., "An improved prototypical network with L2 prototype correction for few-shot cross-domain fault diagnosis," 217 (2023) 113065
- X. Zhang et al., "Feature distance-based deep prototype network for few-shot fault diagnosis under open-set domain adaptation scenario," 201 (2022) 111522
- 2026 papers:
  - Q. Bai et al., contrastive prototypical meta-learning, 292 (2026) 123150
  - C. Li et al., diffusion-model-driven feature mixing for domain shift in few-shot FD, 281 (2026) 121842
  - Z. Zhang et al., Liquid-transformer temporal self-supervised network for few-shot class-incremental FD of servo mechanisms, 272 (2026) 121078
  - W. Jing et al., MACo-Net multi-domain adaptive collaborative few-shot FD, 268 (2026) 120691
  - J. Li et al., physically consistent learnable filter for variable-speed few-shot diagnosis, 123321 (2026, in press)

Sources: [Crossref few-shot listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=few-shot+fault+diagnosis&filter=from-pub-date:2021-01-01); [Crossref meta-learning listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=meta-learning+fault+diagnosis&filter=from-pub-date:2021-01-01)

**Self-supervised pretraining, in *Measurement*:**
- Y. Liu, W. Wen, Y. Bai, Q. Meng, "Self-supervised feature extraction via time–frequency contrast for intelligent fault diagnosis of rotating machinery," 210 (2023) 112551 (37 citations)
- D. Kong et al., "Self-supervised knowledge mining from unlabeled data for bearing fault diagnosis under limited annotations," 220 (2023) 113387
- G. Chen, Q. Tang, physics-guided spatio-temporal graph contrastive SSL, 283 (2026) 121966

Source: [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=self-supervised+fault+diagnosis&filter=from-pub-date:2021-01-01)

**LLMs and large pretrained models, in *Measurement*:**
- J. Wang, G. Peng, Y. Chen, W. Zhang, W. Wu, T. Liu, "Auditable progressive multimodal bearing fault diagnosis with Large language model Screening," *Measurement* 292 (2026) 123183. Abstract not indexed — [Crossref](https://api.crossref.org/works/10.1016/j.measurement.2026.123183)
- X. Yu, Z. Ma, T. Liu, Z. Zhang, B. Tang, "Multi-head memory large language model for predicting degradation trends of gearbox under non-stationary excitation," *Measurement* 260 (2026) 119792. Wind-turbine gearbox degradation-trend prediction. A multi-head memory attention uses a memory matrix, and a normalization enhancement layer is added. The authors note that "existing large models exhibit limitations" under non-stationary excitation. This is **prognosis, not classification** — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0263224125031513)
- Vision foundation models adapted in *Measurement*, all industrial inspection and not time series:
  - X. He et al., "MSIDetector … adapted visual foundation model …," 242 (2025) 115753
  - L. Li et al., promptable visual foundation model for hot ring rolling, 290 (2026) 123021
  - Z. Wu et al., water level measurement with visual foundation models, 256 (2025) 118502

  Source: [Crossref listing](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=foundation+model+fault+diagnosis&filter=from-pub-date:2021-01-01)
- **No *Measurement* TSFM paper found.** Crossref title queries combining "time series foundation model", Chronos, MOMENT, TimesFM, GPT, "large model fine-tuning" and "pretrained transformer fault diagnosis" with ISSN 0263-2241 returned no TSFM fault-diagnosis title — [Crossref TSFM query](https://api.crossref.org/journals/0263-2241/works?query.bibliographic=time+series+foundation+model&filter=from-pub-date:2021-01-01)
- **Outside *Measurement* (arXiv preprint).** MOMENT and Mantis features with fine-tuning for motor condition monitoring. MOMENT beats conventional deep-learning baselines with 1% of the training data, and Mantis reaches about 90% at that ratio — [arXiv 2511.23177](https://arxiv.org/html/2511.23177). UniFault, a bearing fault-diagnosis foundation model for few-shot use, is also arXiv only — [arXiv 2504.01373](https://arxiv.org/pdf/2504.01373)

### Inferences
- If accepted, the planned paper would plausibly be among the first *Measurement* papers to adapt a pretrained time-series foundation model to fault or PD diagnosis. The novelty claim should still be phrased as "to the best of our knowledge", because only titles were searched.
- The few-shot vibration papers above give standard comparison baselines and episode-based evaluation conventions for reviewers in this venue: ProtoNet, relation network, MAML and semi-supervised meta-learning.
- The 2026 LLM papers show that the editors already accept "large model" framing, provided it is tied to a measured signal (vibration) and a concrete task.

### Gaps
- The search covered titles and bibliographic fields only. A paper could use MOMENT or Chronos inside its method without naming it in the title. Full-text search was not possible: ScienceDirect returned 403 and OpenAlex hit its rate limit.
- Abstracts of the 2026 LLM screening paper and the FFE-MAML distribution-system paper were not available.

## Q4. What does a typical *Measurement* fault-diagnosis paper look like structurally, and which 3–5 papers are the best structural templates?

### Takeaway
No full text could be opened, so section headings and figure or table counts could **not** be verified for any paper. Three things are known:
- **Measurable proxies.** Reference counts in the target papers run from 24 to 69, mostly 29–54.
- **Author-guide requirements.** Research articles have no fixed length. Highlights, a data-availability statement and CRediT statements are required, and review is double-anonymized.
- **The journal's ML policy.** It requires a measurement context, enough detail to replicate, and specific metrics.

The best templates for a few-shot, time-series PD paper are Jing 2025 (TRSMM), Wang 2023 (zero-shot GIS PD), Raymond 2021 (noise-invariant CNN), Li 2025 (GB-SMoE) and Sun 2023 (TDA cable PD).

### Cited Findings
- **Reference counts (Crossref), as a length proxy:**

  | Paper | References |
  |---|---|
  | Sun 2023 TDA | 24 |
  | Sahoo 2024 | 25 |
  | Ye 2022 HVCB | 29 |
  | Wang 2023 ZSL | 36 |
  | Jing 2025 TRSMM | 38 |
  | Zhang 2025 ResGCN | 38 |
  | Raymond 2021 | 40 |
  | Yang 2026 CB | 40 |
  | Liu & Peng 2025 | 42 |
  | Li 2025 GB-SMoE | 43 |
  | He 2025 turnout | 44 |
  | Zhang 2023 dry-type | 47 |
  | Liu 2026 MambaGAN | 53 |
  | Zhang 2022 HVCB fusion | 54 |
  | Deng 2025 KSMNet | 62 |
  | Qi 2025 OLTC | 69 |

  Sources: [Crossref works API](https://api.crossref.org/works/10.1016/j.measurement.2025.118054), queried per DOI.
- **Author-guide items** (third-party summary of the guide; verify on Elsevier's page):
  - Research articles have no fixed word or page limit
  - Highlights, data availability statement, CRediT contributions and ORCID iDs are required; a graphical abstract is encouraged
  - Review is double-anonymized
  - Article types: research article, review (usually commissioned), technical note

  Source: [Manusights submission guide](https://manusights.com/blog/measurement-submission-guide)
- **Policy on ML papers.** An ML or neural-network paper must:
  - advance measurement science, not just apply a tool
  - place the tools in a measurement-related context
  - give enough detail on tools, data and results for replication
  - use specific metrics

  "Failing to adhere to these guidelines will result in a paper desk-reject decision." Sources: [Elsevier shop journal page](https://shop.elsevier.com/journals/measurement/0263-2241); [Measurement policies page](https://www.sciencedirect.com/journal/measurement/about/policies/information-for-authors-measurement)
- **Experimental-platform evidence from abstracts:**
  - Wang 2023 built its own GIS multi-source PD experiment — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224123005973)
  - Liu 2026 used a PD sensor experimental platform with 4 × 200 samples — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224125034463)
  - Zhang 2022 states its 2 vibration + 1 current sensor configuration — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0263224122001786)
  - Raymond 2021 states its cross-validation protocol in the abstract — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.measurement.2021.109220?fields=abstract)
- **Recommended templates and why** (rationale grounded in the abstract findings in Q1 and Q2):
  1. **Jing et al. 2025, TRSMM (255:118054).** It has the same problem framing: small-sample GIS PD, a transfer-learning objective, explicit handling of field noise, and evaluation on two datasets or a field dataset. Use it to frame the introduction and the transfer setting.
  2. **Wang et al. 2023, zero-shot GIS PD (217:113033).** Its data-scarcity motivation ("collecting enough multi-source samples is unrealistic") matches the planned paper. It has a self-built laboratory platform and a modular method with three named sub-networks.
  3. **Raymond et al. 2021, noise-invariant PD CNN (177:109220).** Its evaluation protocol trains on clean data and tests on noise-shifted data with a modified 10-fold CV. It uses transfer learning and frames robustness in measurement terms. Use it for the robustness or shift experiment section.
  4. **Li et al. 2025, GB-SMoE (256:118139).** It uses explicit phase-wise fusion of PRPS data. A phase-anchored method must position itself against it, and its multi-task results section is a usable model.
  5. **Sun et al. 2023, TDA cable PD (217:113009).** It is a compact paper (24 references) that turns the raw PD time series into a new representation for a 1D-CNN, with a single headline gain (+11.25%). It is a good model for a lean time-series-representation paper.
  - For the few-shot evaluation section specifically, Zhang et al. 2023 (dry-type transformer, 15 samples per class, mean ± std reported) shows *Measurement*-style reporting of few-shot variance — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S026322412301014X)

### Inferences
- **Probable section skeleton** (inferred from the abstracts above and common *Measurement* practice; not verified from full texts):
  1. Introduction, with related work folded in or as Section 2
  2. Theory or method
  3. Experimental platform and data acquisition: test object, defect models, sensor, bandwidth, sampling, calibration
  4. Results and discussion: comparison with baselines, ablation, noise or shift robustness, visualization (t-SNE, confusion matrix)
  5. Conclusion
- Under the ML policy, the platform section is not optional. The planned paper should describe:
  - the IEC 60270 circuit or sensor chain, calibration in pC, sampling rate and record length
  - how the phase reference to the test voltage is obtained and its uncertainty; this is where "phase-anchored" becomes a measurement contribution, not only an ML trick
  - specimen counts and split by specimen or session
  - repeated few-shot episodes with mean ± std or confidence intervals
  - data and code availability
- Ref. Florkowski 2022 (harmonics distort phase-resolved patterns) can justify why the phase anchor must be robust to waveform distortion.

### Gaps
- Exact section headings, figure counts, table counts, number of baselines and number of ablations could not be verified for any template. ScienceDirect blocked access, and none of the target papers carries a CC licence in Crossref, so all are subscription-only.
- Manusights is a third-party source. Its "often well under 8,000 words" estimate is that site's opinion, not journal policy.

## Q5. Scope signals, review speed, acceptance rate, and APC / open-access options

### Takeaway
*Measurement* is a high-volume, roughly 19%-acceptance, hybrid IMEKO journal. First decisions come fast (about 5–7 days), a decision after review takes about 2 months, and acceptance takes about 4–5 months. Its scope **explicitly excludes** fault-diagnosis papers "with little or no elements of measurement science or technology," and it desk-rejects ML papers that lack a measurement context, replicability or specific metrics.

### Cited Findings
- **Elsevier Journal Insights (current snapshot).** I could not open the page myself (HTTP 403), so these figures come from a search-engine summary of it:

  | Metric | Value |
  |---|---|
  | Submission to first decision | 6 days |
  | Submission to decision after review | 54 days |
  | Submission to acceptance | 114 days |
  | Acceptance rate | 19% |
  | CiteScore | 11.5 |
  | Impact factor | 5.6 |
  | Print / online ISSN | 0263-2241 / 1873-412X |

  Source: [Measurement Insights page](https://www.sciencedirect.com/journal/measurement/about/insights)
- **An earlier Journal Insights snapshot** gave 7 / 57 / 126 days, IF 6.1 and CiteScore 10.2. **The IF figures conflict (5.6 vs 6.1)**, so check the current JCR — [journalinsights.elsevier.com review_speed](https://journalinsights.elsevier.com/journals/0263-2241/review_speed) (now redirects to the ScienceDirect journal page)
- **Peeref (third party; source of figures not stated).** Not an OA journal.

  | Metric | Value |
  |---|---|
  | First decision | 5 days |
  | Decision after review | 62 days |
  | Acceptance | 147 days |
  | Acceptance to online publication | 7 days |
  | Articles per year | 3,166 |
  | Gold OA share | 11.82% |
  | JCR quartile | Q1 (Engineering, Multidisciplinary; Instruments & Instrumentation) |

  Source: [Peeref](https://www.peeref.com/journals/5753/measurement)
- **LetPub** gives average peer review of about 3.8 months — [LetPub](https://www.letpub.com/journal-selector/journal/5753)
- **Scope (Elsevier):**
  - "novel achievements in all fields of measurement and instrumentation science and technology"
  - out of scope: papers on "image processing or fault diagnosis with little or no elements" of measurement science or technology
  - out of scope: military applications, and measurement results without new insight
  - ML papers are desk-rejected unless they meet the measurement-context, replicability and metrics conditions

  Source: [Elsevier shop journal page](https://shop.elsevier.com/journals/measurement/0263-2241). Per a search summary, the ML rule dates from a January 2021 policy update — [Measurement policies page](https://www.sciencedirect.com/journal/measurement/about/policies/information-for-authors-measurement)
- **Open access and APC.**
  - Hybrid. Gold OA APC "roughly USD 3,800" according to a third party, not verified on Elsevier's page — [Manusights](https://manusights.com/blog/measurement-submission-guide)
  - The Elsevier shop page lists no APC and points to OA sister titles (Measurement: Sensors, Measurement: Food, Measurement: Energy, Measurement: Digitalization) — [Elsevier shop](https://shop.elsevier.com/journals/measurement/0263-2241)
  - Measurement: Sensors (a different journal) charges up to USD 1,800 with a waiver policy — [DOAJ](https://www.doaj.org/toc/2665-9174)

### Inferences
- The scope exclusion is the main risk for a TSFM-for-PD paper. The manuscript should present phase anchoring as a measurement-domain contribution (synchronization to the test voltage, robustness to phase-reference error or harmonics, calibrated apparent charge as an input channel), not as a generic adaptation trick. It should also report uncertainty: variance across episodes and specimens.
- Publishing subscription-only avoids the APC. OA would cost roughly USD 3–4k unless an institutional agreement applies.

### Gaps
- I could not open the Elsevier Journal Insights page or the APC list directly (403). The current APC in USD and the exact Journal Insights reporting window are unverified.
- No data on desk-rejection share or on review speed for the fault-diagnosis subset specifically.
