# Measurement Science and Technology (MST, IOP, ISSN 0957-0233) — PD recognition, HVCB mechanical diagnosis, and few-shot / foundation-model fault diagnosis (2021–2026)

**How venue was verified.** Every paper listed under "Cited Findings" was retrieved through the Crossref REST API restricted to the MST ISSN (`/journals/0957-0233/works`, filter `from-pub-date:2021-01-01`), so all have DOI prefix `10.1088/1361-6501` and IOP as publisher. Example of the query pattern: [Crossref MST query "partial discharge"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01&rows=100). Per-paper metadata (authors, volume, article number, reference count) comes from `https://api.crossref.org/works/<DOI>`. Abstract-level facts come from the IOP abstract (deposited in Crossref and shown on the IOPscience page). Full text was readable for only one PD paper: Chang & Boyanapalli 2024, which is open access. Every other paper was read at abstract level only, because the full text is either paywalled or blocked by IOP's bot challenge. Wherever the abstract does not give dataset size, split or baselines, the entry says **"not found"**.

**Citation format.** MST uses article numbers. Cite as *Meas. Sci. Technol.* **vol** article-no (year). The first two digits of the article number encode the issue: e.g. 125118 → issue 12. By 2026 volume 37 runs to 38+ issues, e.g. [aea58b, Vol 37 Issue 38](https://api.crossref.org/works/10.1088/1361-6501/aea58b).

---

## Q1. Which MST papers address PD recognition (GIS, cable, transformer, switchgear; UHF/HFCT/IEC 60270; PRPD/PRPS/time-domain inputs)?

### Takeaway
MST has a small but coherent PD deep-learning corpus. A Crossref title/metadata sweep for 2021–2026 found about 10 PD recognition/diagnosis/assessment papers plus one 2026 topical review. Six of them come from a single author group (Yanxin Wang, Jing Yan, Jianhua Wang, Yingsan Geng; later with Dipti Srinivasan), and they focus on **GIS PD under small-sample / domain-shift / zero-shot conditions**. Only one MST PD paper is clearly measurement-centric and open access with full details: Chang & Boyanapalli 2024, cable joints, PRPD + CNN, specimen counts given. I found no MST PD paper that uses a pretrained time-series foundation model.

### Cited Findings

**Core PD recognition / diagnosis papers in MST (verified DOI prefix 10.1088/1361-6501):**

1. **Wang Y, Yan J, Yang Z, Wang J, Geng Y (2021). "A novel 1DCNN and domain adversarial transfer strategy for small sample GIS partial discharge pattern recognition." *Meas. Sci. Technol.* 32 125118. DOI 10.1088/1361-6501/ac27e8.** 30 references; 27 Crossref citations at retrieval — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ac27e8)
   - Model: a 1D CNN for unbalanced samples, plus domain adversarial transfer learning (DATL) with **two domain classifiers** that align decision boundaries. Moves a lab-trained model to **on-site GIS** with small samples, some of them unlabeled — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ac27e8)
   - Reported accuracy: **98.67% (source domain), >92% (target domain)**. Claims to beat the reliance of 2D CNNs on sample size — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ac27e8)
   - Input signal type (UHF vs other), dataset size, split and named baselines: **not found** in the abstract; the paper is closed access — [Unpaywall status via IOP page](https://iopscience.iop.org/article/10.1088/1361-6501/ac27e8)

2. **Wang Y, Yan J, Jing Q, Wang J, Geng Y (2022). "A novel federated deep learning framework for diagnosis of partial discharge in gas-insulated switchgear." *Meas. Sci. Technol.* 33 095112. DOI 10.1088/1361-6501/ac7a09.** 30 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ac7a09)
   - Model: federated learning with an improved FedAvg, plus a **subtractive-attention Siamese network** for unbalanced data. Reports **95.61%** accuracy and claims good performance with unbalanced and small samples — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ac7a09)
   - Dataset size, split and baselines: **not found**.

3. **Wang Y, Yan J, Yang Z, Wang Z, Wang J, Geng Y (2023). "Meta-autoencoder-based zero-shot learning for insulation defect diagnosis in gas-insulated switchgear." *Meas. Sci. Technol.* 34 065114. DOI 10.1088/1361-6501/acc1fc.** Hybrid open access; 30 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/acc1fc)
   - Model: CNN visual features, then a semantic autoencoder with triplet loss and unknown-class semantic constraints, then nearest-neighbour classification. Uses **episode training** to learn semantic prototypes — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/acc1fc)
   - Accuracy without any test-class training samples: **96.215% (single-source defects), 90.41% (multi-source defects)** — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/acc1fc)
   - Timeline: received 2 Jan 2023, accepted 7 Mar 2023, published 23 Mar 2023; the peer-review panel lists 2 revisions — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/acc1fc)
   - Sensor, dataset size and split: **not found** (the page body could not be read).

4. **Yan J, Wang Y, Zhang W, Wang J, Geng Y, Srinivasan D (2024). "Domain-alignment multitask learning network for partial discharge condition assessment with digital twin in gas-insulated switchgear." *Meas. Sci. Technol.* 35 065109. DOI 10.1088/1361-6501/ad3412.** 30 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ad3412)
   - Model: DAMTLN, a multitask network for joint PD diagnosis and localisation, with a digital-twin virtual model. Domain adaptation combines local MMD with an adversarial term — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad3412)
   - Reported results: **98.73%** diagnostic accuracy; localisation MAE **9.06 cm** — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad3412)

5. **Wang Y, Yan J, Zhang W, Geng Y, Wang J, Srinivasan D (2024). "Multi-source domain adaptation network for partial discharge severity assessment in gas-insulated switchgear." *Meas. Sci. Technol.* 35 125105. DOI 10.1088/1361-6501/ad7488.** 28 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ad7488)
   - Model: MSDAN. The feature extractor captures PD features and long-term dependencies. Multi-source domain adaptation across defect types is combined with adaptive weighted classification. Reports **95.38%** for on-site GIS PD severity assessment — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad7488)

6. **Chang C-K, Boyanapalli B K (2024). "Study of partial discharges measurement cycles effect on defect recognition for underground cable joints." *Meas. Sci. Technol.* 35 065406. DOI 10.1088/1361-6501/ad366c.** Open access; full text read — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
   - Specimens: **14 cable joints** (Type A ×6, Type B ×6, Type C ×2). The joints are 25 kV / 200 A pre-molded straight-through joints on 3–5 m XLPE cable. Defects: A = 10 mm gap between cable insulation and inner semiconductor; B = 3 mm-diameter, 4 mm-deep void; C = triangular protrusion (15 mm base, 20 mm height) — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
   - Measurement chain: RLC measuring impedance (200 kHz bandwidth, 300 kHz centre frequency), an 8000:1 capacitive divider, and a **20 MHz ADC**. Test voltage was 24–32 kV for A/B and 25–55 kV for C. The voltage protocol was 2 days on, 1 day off, stepped by 5 kV, with recordings every 4 min — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
   - Input: **PRPD (n-q-φ) images at 88×66 px** built from 40/80/120/200/1200 power cycles. Sample counts: 10,500 at 40 cycles (4786/2968/2746); 2,099 at 200 cycles (957/593/549); 349 at 1200 cycles — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
   - Model: CNN with 4 conv layers, 2 max-pool layers, ReLU, Adam, 20 epochs. **100%** total recognition accuracy at 200 cycles. The paper also argues for a confidence threshold to suppress false defect-type calls — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c)
   - Train/test split method: **not found**. The text says training/testing data were "kept consistent" across cycle settings but gives no ratio and no specimen-wise split. Named DL baselines: **none**; the comparison is across measurement durations — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)

7. **Liu Y, Lai Z, Hua X, Liu H-Y, Li W, Li H, Zhang G-J, Chen X (2025). "A novel modal fusion method using cross-attention for partial discharge pattern recognition in gas insulated switchgear." *Meas. Sci. Technol.* 36 086139. DOI 10.1088/1361-6501/adfb00.** 32 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/adfb00)
   - Input: **PRPD spectra (DenseNet) plus time-domain waveforms (1D-CNN → GCN cascade)**, fused by multi-head cross-attention (TPCAN) — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/adfb00)
   - Baselines and gains: +2.37% vs CoAtNet, +5.01% vs ResNet, +10.27% vs 1-DCNN, +13.91% vs LSTM. Also includes an ablation of the cross-attention module and a sensitivity analysis on the number of attention heads. Absolute accuracy is not given in the abstract — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/adfb00)
   - Data are not public ("available on request"). Received 30 Mar 2025, revised 9 Aug 2025, accepted 13 Aug 2025, published 26 Aug 2025 — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/adfb00)

8. **Yan J, Wang Y, Lin Y, Geng Y, Wang J, Xiao H, Ding R (2025). "A novel domain adaptive network guided by expert experience and active learning for field gas insulated switchgear insulation defect diagnosis." *Meas. Sci. Technol.* 36 115101. DOI 10.1088/1361-6501/ae185f.** 54 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ae185f)
   - Setting: lab-to-field domain gap plus **novel classes** in the target domain. Method (ALDAN): autoencoder-based PD signal reconstruction with prior-feature prediction, and source-free open-set domain adaptation driven by active learning. Reports **93.26% with only 6% labelled samples** on real-world GIS datasets — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae185f)

9. **Tian H, Song R, Du J, Wang J, Liu H, Zhang W, Song Y, Zhang Z, Chen W (2025 online / vol. 37 2026). "Optical fiber sensing and partial discharge fault diagnosis for power cable health monitoring." *Meas. Sci. Technol.* 37 015106. DOI 10.1088/1361-6501/ae26a8.** 17 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ae26a8)
   - Sensor: fibre-optic Michelson interferometer detecting PD acoustic signals. Pipeline: denoising, time–frequency images, DL classifier. Accuracy, dataset and split: **not found** in the abstract — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae26a8)

10. **Chen Z, Qian Y, Pan C, Lei Y, Li J, Sheng G (2026). "Multi-modal sensing and pulse sequence analysis for single-source and dual-source partial discharge diagnosis in transformers under complex operating conditions." *Meas. Sci. Technol.* 37 326107. DOI 10.1088/1361-6501/ae8e99.** Hybrid OA; 41 references — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ae8e99)
    - Platform: lab rig with **five typical PD defect models of oil-immersed transformers**, using synchronised **Optical + UHF + HFCT** measurements. An SDLTR method suppresses narrowband interference in HFCT under undersampling — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae8e99)
    - Input: **pulse sequence analysis (PSA)** time-series features; a tri-modal PSA fusion uses expert-weighted decisions. Under class imbalance, PSA6 reaches **95.47% accuracy / 95.17% Macro-F1**. Dual-source mixtures are identified in all 10 combinations, with zero false decisions in single-source verification — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae8e99)
    - Received 14 May 2026, revised 30 Jun 2026, accepted 22 Jul 2026, published 10 Aug 2026. Data available on reasonable request — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae8e99)

11. **Govindarajan S, Portilla-Gómez J, Ardila-Rey J, Subramanian V, Ramakrishnan A H (2026). "Deep learning for partial discharge diagnosis in electrical assets: fundamentals, applications, and future perspectives." *Meas. Sci. Technol.* 37 212001 (topical review). DOI 10.1088/1361-6501/ae6a0c.** **174 references** — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ae6a0c)
    - Frames DL-for-PD research as "fragmented" across pre-processing and architectures. Discusses differences and validation strategies **between laboratory and field testing**, and proposes a framework for future work — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae6a0c)

**PD measurement-side MST papers.** These are sensing and localisation papers, useful for measurement framing; titles are from Crossref:
- Distributed optical-fibre acoustic sensor for PD detection (2024), DOI 10.1088/1361-6501/ad480b; transformer PD ultrasonic sensor optimisation (2023), DOI 10.1088/1361-6501/ace46a; dual-sensor ultrasonic PD localisation in transformers (2025), DOI 10.1088/1361-6501/ae1991; multi-source PD localisation by FCM-LOF clustering (2022), DOI 10.1088/1361-6501/ac9494; 3D PD localisation by PSO plus voice-activity detection (2026), DOI 10.1088/1361-6501/ae7741; fibre MZI PD detection in XLPE cable accessories (2026), DOI 10.1088/1361-6501/ae65c0 — [Crossref MST query](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=partial+discharge&filter=from-pub-date:2021-01-01&rows=100)

### Inferences
- The MST PD-recognition literature is thin (about 10 papers in 6 years) and concentrated in one group's GIS work. A PD paper on few-shot TSFM adaptation would face little direct overlap in MST, but it will almost certainly be reviewed against this group's series. Cite ac27e8, acc1fc, ae185f as the small-sample/zero-shot/few-label lineage, and ad3412, ad7488 as the domain-adaptation lineage.
- Input-representation precedent in MST: PRPD images (ad366c), PRPD + waveform fusion (adfb00), pulse-sequence analysis (ae8e99), time–frequency images (ae26a8). A **phase-anchored raw time-series** representation fed to a TSFM is not represented. It sits between PRPD (phase-folded) and PSA (pulse-to-pulse), and that positioning can be argued explicitly.
- Chang & Boyanapalli show MST reviewers accept a paper whose *measurement parameter* (acquisition duration in power cycles) is the main independent variable. A "how many cycles / how many shots are needed" analysis fits MST's measurement emphasis.
- None of the MST PD abstracts reports a **specimen-wise (or session-wise) train/test split**. Even ad366c, which lists only 2–6 specimens per class, does not state one. A leakage-safe split by specimen would be a differentiator worth stating.

### Gaps
- For the closed papers (ac27e8, ac7a09, ad3412, ad7488, adfb00, ae185f, ae26a8) I could not find sensor type (UHF/HFCT/IEC 60270), sample counts, split protocol or baseline lists. IOPscience serves only front matter to unauthenticated fetches, and the PDF route returned a JavaScript challenge.
- The OA papers acc1fc and ae8e99 could not be downloaded either (same bot challenge), so their section structure and figure/table counts are **not found**.
- The Crossref search matches titles and bibliographic metadata, not full text. A PD paper with an unusual title, e.g. one that never says "partial discharge", could be missed.

---

## Q2. Which MST papers address HV circuit breaker mechanical fault diagnosis, especially few-shot settings?

### Takeaway
MST has a clear HVCB cluster from one group (Qiuyu Yang, Yuyi Lin, Jiangjun Ruan). It covers **zero-shot** compound-fault diagnosis from vibration (2024 ×2) and an **acoustic** method tested in few-shot and cross-manufacturer scenarios (2025). The other CB papers are signal-processing + SVM, life/RUL prediction, or sensing papers. I found **no MST paper using coil current or travel curves** as the primary input, and none titled "few-shot" for CBs.

### Cited Findings
- **Yang Q, Lin Y, Ruan J (2024). "A zero-shot fault attribute transfer learning method for compound fault diagnosis of power circuit breakers." *Meas. Sci. Technol.* 35 056111. DOI 10.1088/1361-6501/ad2667.** DSR-AL: a depthwise separable residual CNN learns from **3D time–frequency images of CB vibration**, and fault attribute learners transfer knowledge to unseen compound faults. Validated by orthogonal experiments on **real industrial switchgear**. Accuracy is not stated in the abstract — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad2667); 39 references, 11 citations — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ad2667)
- **Yang Q, Liu Y, Lin Y, Li J, Ruan J (2024). "Zero-shot fault diagnosis of high-voltage circuit breakers: fusion of phase space reconstruction and attribute embedding methods." *Meas. Sci. Technol.* 35 116113. DOI 10.1088/1361-6501/ad6898.** AEZSD: phase-space reconstruction of vibration signals, with fault attributes derived from electromechanical signal characteristics and an attribute-embedding network. Diagnoses unseen faults from historical fault data. No accuracy in the abstract — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad6898)
- **Yang Q, Cai Y, Ruan J, Xue X, Xie J, Lin Y (2025). "Acoustic AMFbank and DPC-MEnet: an enhanced diagnostic approach for power circuit breakers." *Meas. Sci. Technol.* 36 116125. DOI 10.1088/1361-6501/ae1cde.** Uses an acoustic adaptive Mel filterbank and a dual-path convolutional mix-feature enhancement network. Claims improved accuracy "under different sampling rates, **in few-shot scenarios**, and across CBs from distinct manufacturers". Numbers not in the abstract — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae1cde)
- **Yan R, Zhuang W, Yu N (2024). "Research on circuit breaker operating mechanism feature extraction method combining ICEEMDAN-MRSVD denoising and VMD-PSE." *Meas. Sci. Technol.* 35 106123. DOI 10.1088/1361-6501/ad5f4e.** Vibration denoising, VMD power-spectrum-entropy features, SVM; **98.61%** — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad5f4e)
- **Zhao H, Qian Y, Li X (2025). "Remaining useful life prognosis of vacuum circuit breaker mechanical performance via vibration signal feature fusion." *Meas. Sci. Technol.* 36 076120. DOI 10.1088/1361-6501/adec0c.** Uses short-time-energy features for the trigger unit, transmission and spring mechanism, then FPCA with Bayesian updating and tensor fusion. Framed as **small-sample** RUL; MAE/MSE/RMSE all below 0.1 — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/adec0c)
- **Sun S, Wen Z, Zhang W, Wang J, Gao H (2022). "On-line mechanical life prediction method for a conventional circuit breaker based on multi-parameter PSO-SVR using vibration detection." *Meas. Sci. Technol.* 33 095110. DOI 10.1088/1361-6501/ac727f.** Correlates mechanism action-time parameters with vibration events — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ac727f)
- **Jin L, Yao S, Zhang J, Zhang Z (2026). "Research on intelligent sensing method for tulip contact status of circuit breakers based on lightweight multi-class SVM." *Meas. Sci. Technol.* 37 056110. DOI 10.1088/1361-6501/ae3908.** Uses 10 kV VCB contact-pressure time series and a self-built dataset of **408 samples**. Compared against RF and KNN, and validated over 375 threshold/width combinations — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae3908)
- Related switchgear mechanical (GIS, acoustic) papers:
  - **Ji H, Liu H, Wang J, Yuan G, Yang J, Yang S (2023), *Meas. Sci. Technol.* 35 015008, DOI 10.1088/1361-6501/acfbf0** (OA). Auditory-brainstem-response saliency features with a 2D CNN; **96.1%** on a 110 kV three-phase GIS fault-simulation rig, with anti-noise tests — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/acfbf0)
  - **Wang Z, Zhang Z, Shao Y, Qian K, Liu H, Hu B, Schuller B W (2024), *Meas. Sci. Technol.* 35 076121, DOI 10.1088/1361-6501/ad3d78** (OA). **Pre-trained** audio-spectrogram-transformer autoencoder plus a frequency regressor, with multi-stage training; validated on real GIS acoustic data — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad3d78)
- Other CB titles found but not examined: "Non-invasive online detection of power circuit breaker timing and asynchronism via phase-space spectral radius analysis" (2025), DOI 10.1088/1361-6501/adb6c8; "Intelligent state assessment of molded case circuit breaker based on multi-source sensor data fusion and machine learning" (2026), DOI 10.1088/1361-6501/ae6932 — [Crossref MST query "circuit breaker"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=circuit+breaker&filter=from-pub-date:2021-01-01&rows=60)
- Non-MST context, for contrast: a few-shot HVCB paper using Transformer–CNN plus metric meta-learning appeared in IEEE (venue not MST) — [IEEE Xplore](https://ieeexplore.ieee.org/document/10233109/). A cross-domain few-shot HVCB paper appeared in Control Engineering Practice (Elsevier, not MST) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0967066125003399).

### Inferences
- In MST, the low-data CB story is told through **zero-shot attribute transfer** (Yang/Ruan group) rather than N-way K-shot meta-learning. A PD few-shot paper can cite ad2667/ad6898 as evidence that MST publishes data-scarcity work on power switching equipment.
- Coil-current and travel-curve CB diagnosis seems absent from MST titles. That literature appears to sit in IEEE/Elsevier venues; this is inferred from the search results, not verified exhaustively.

### Gaps
- Accuracy, dataset size and split for the Yang/Ruan CB papers: **not found** (the abstracts give no numbers; full text is closed).
- I could not confirm whether any MST CB paper uses an explicit N-way K-shot protocol. The abstract of ae1cde mentions only "few-shot scenarios".

---

## Q3. Which MST papers use few-shot / transfer / meta-learning / self-supervised / large pretrained models for time-series fault diagnosis? Has MST published anything with MOMENT, Chronos, TimesFM, Moirai, Mantis, Time-LLM or similar TSFMs?

### Takeaway
Few-shot fault diagnosis is a **very crowded** MST topic. Dozens of 2022–2026 papers exist, mostly on bearings and gearboxes, using prototypical/Siamese/meta-transfer/contrastive SSL. In 2025–2026 MST began publishing **LLM / vision-language / tabular-foundation-model** fault-diagnosis papers. A Crossref metadata search returned **zero MST hits for "Chronos", "TimesFM" and "Moirai"**, and no title mentions MOMENT, Mantis or Time-LLM. Using a pretrained *time-series* foundation model for PD therefore appears unclaimed in MST as of Oct 2026, judging by titles and metadata.

### Cited Findings

**Foundation-model / LLM-based fault diagnosis in MST (closest analogues to a TSFM-adaptation paper):**
- **Wu Y, Zhao X, Cui Y, et al. (2026). "LLM-VSFD: cross-modal learning for few-shot vibrating screen fault diagnosis." *Meas. Sci. Technol.* 37 046008. DOI 10.1088/1361-6501/ae3978.** 62 references. Two stages: (1) contrastive + classification alignment of time-series vibration into the LLM semantic space; (2) a hybrid prompt module plus **LoRA** fine-tuning. **5-shot: 84.74%** on a real vibrating-screen dataset (about +30 points over the next-best baseline); 80.59% on CWRU and 86.45% on JNU — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae3978)
- **Gu H, Zhang D, Yi S, Wang C (2026). "Prompt transfer for few-shot fault diagnosis in IIoT: a federated learning framework with tabular foundation models." *Meas. Sci. Technol.* 37 326113. DOI 10.1088/1361-6501/ae9571** (OA). In-context learning of a **frozen tabular foundation model**, with zero gradient updates. **5-shot: 94.0%** on the SEU bearing data vs FedAvg 20.1%; communication rounds down 95% — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae9571)
- **Fan J, Zhao F, Gong Y, Zhang R (2026). "A vision–language-guided multimodal framework with dynamic prototype memory for small-sample fault diagnosis." *Meas. Sci. Technol.* 37 346109. DOI 10.1088/1361-6501/ae9585** (OA). Explicitly cites "large-scale pretrained foundation models" as transferable priors and converts time series to images for a pretrained V–L model. Tested on CWRU and Paderborn under multiple small-sample ratios and cross-condition transfers — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae9585)
- **Pan Y, Guo Y, Huang S, Wu T, Tao Y (2026). "Spectrum analysis and perception multimodal LLM for robust fault diagnosis under data scarcity." *Meas. Sci. Technol.* 37 126115. DOI 10.1088/1361-6501/ae51b4** (OA). A fine-tuned MLLM with a spectrum-aware visual encoder and a spectral vector encoder; compared with ViT and traditional methods — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae51b4)
- **Ma L, Wang Z, Guan S, Zhi S, Wang T (2026). "CoFAE-LLM framework…" *Meas. Sci. Technol.* 37 186107. DOI 10.1088/1361-6501/ae612d** (OA). Multi-view Dirichlet evidence is mapped through soft prompts into an LLM. Reports 99.27/98.64/98.11% on CWRU/SEU/JNU, with WDCNN as the strongest baseline (+1.02/+1.22 points) — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae612d)
- **Zhang K, Zhao F, Wu X, Dong C, Yao Y (2026). "Explainable fault diagnosis with large vision language models." *Meas. Sci. Technol.* 37 386204. DOI 10.1088/1361-6501/aea58b** (OA). CWT images plus expert text are used to fine-tune an LVLM with "asymmetric QLoRA", and a reasoning-chain prompt is applied at inference — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/aea58b)
- Other LLM titles in MST: RAF-LLM, retrieval-augmented LLM for RUL (2026), DOI 10.1088/1361-6501/ae989d; LLM-informed RUL for bearings (2026), DOI 10.1088/1361-6501/ae8879; wind-turbine blade damage with LLM + Mamba (2025), DOI 10.1088/1361-6501/ae2287 — [Crossref MST query "Time-LLM"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=Time-LLM&filter=from-pub-date:2021-01-01&rows=60)
- **TSFM name search:** Crossref MST queries for "Chronos", "TimesFM" and "Moirai" (2021+) each returned **total-results = 0**. The query "Chronos TimesFM MOMENT" returned only unrelated "moment" papers (moment of inertia, central moment discrepancy) — [Crossref "TimesFM"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=TimesFM&filter=from-pub-date:2021-01-01); [Crossref "Chronos"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=Chronos&filter=from-pub-date:2021-01-01); [Crossref "Moirai"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=Moirai&filter=from-pub-date:2021-01-01)

**Transfer / fine-tuning / few-shot / SSL papers in MST (selected, most citable as related work):**
- **Tang T, Wu J, Chen M (2022). "Lightweight model-based two-step fine-tuning for fault diagnosis with limited data." 33 125112. DOI 10.1088/1361-6501/ac856d.** Tunes "less task-specific weights" in two steps, with a distance loss for sparser features. This is the closest MST precedent for **partial fine-tuning of a pretrained backbone** — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ac856d)
- **Zhong J, Gu K, Jiang H, Liang W, Zhong S (2024). "A fine-tuning prototypical network for few-shot cross-domain fault diagnosis." 35 116124. DOI 10.1088/1361-6501/ad67f5.** Splits the target support set into a pseudo-support and a pseudo-query set to fine-tune; validated on three rotating-equipment datasets — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad67f5)
- **Wu K, Nie Y, Wu J, Wang Y (2023). "Prior knowledge-based self-supervised learning for intelligent bearing fault diagnosis with few fault samples." 34 105104. DOI 10.1088/1361-6501/acddd9.** Combines meta-learned priors, SSL and GCN fusion; transfers from artificial to real damage — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/acddd9)
- **Zheng X, Feng Z, Lei Z, Chen L (2024). "Few-shot learning fault diagnosis of rolling bearings based on siamese network." Vol 35, issue 9; article number not found. DOI 10.1088/1361-6501/ad57d9.** Baselines: SVM, DCNN, WDCNN, CNN-BiGRU, on two bearing datasets — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad57d9). The authors' initials come from a search snippet and are unverified; check them on the IOP page before citing.
- Other few-shot titles: "Self-supervised contrastive learning with time-frequency consistency for few-shot bearing fault diagnosis" (2025), DOI 10.1088/1361-6501/add9b4; "An enhanced meta-learning network with sensitivity penalty for cross-domain few-shot fault diagnosis" (2024), DOI 10.1088/1361-6501/ad5039; "A meta-learning method for few-shot bearing fault diagnosis under variable working conditions" (2024), DOI 10.1088/1361-6501/ad28e7; "Cross-condition few-shot bearing fault diagnosis via learnable diagonal metric and transformer representation" (2026), DOI 10.1088/1361-6501/ae6ac5; "A hierarchical transformer-based adaptive metric and joint-learning network for few-shot rolling bearing fault diagnosis" (2023), DOI 10.1088/1361-6501/ad11e9 — [Crossref MST query "self-supervised pretraining fault diagnosis few-shot"](https://api.crossref.org/journals/0957-0233/works?query.bibliographic=self-supervised+pretraining+fault+diagnosis+few-shot&filter=from-pub-date:2021-01-01&rows=60)
- Power or electrical-equipment few-shot work in MST (non-PD):
  - **Jian Y, An A, Yu S, Chang T (2026). "Few-shot learning photovoltaic fault diagnosis based on Tri-MSTCN." 37 236201, DOI 10.1088/1361-6501/ae7139.** Accuracy rises from 96.67% (45 samples) to 98.89% (135 samples); compared with GRU, CBG and TCN — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae7139)
  - **Qiao L, Li W (2026). "Sample-weighted prototypical network with EMD distance: a few-shot fault diagnosis approach for wind turbine generators." 37 266109, DOI 10.1088/1361-6501/ae7fa6.** N-way K-shot episodes, a CNN + Transformer encoder, and Earth mover's distance; data are simulated with OpenFAST — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae7fa6)
  - "FS-TFD-MEN: an asynchronous motor few-shot fault diagnosis framework based on integrated metrics" (2026), DOI 10.1088/1361-6501/ae5fbd — [Crossref](https://api.crossref.org/works/10.1088/1361-6501/ae5fbd)

### Inferences
- The MST few-shot space is saturated with bearing benchmarks: CWRU, PU/Paderborn, JNU and SEU recur. A PD paper built on **real lab PD CSV data** with a TSFM backbone would stand apart on both data domain and backbone class. Reviewers will still expect the standard few-shot apparatus: N-way K-shot episodes, several K values, repeated episodes, and prototypical/Siamese/meta-learning baselines.
- MST's 2026 LLM papers set a reviewer-familiar precedent for **parameter-efficient adaptation (LoRA/QLoRA, prompts, in-context learning)** of large pretrained models on sensor signals. Citing LLM-VSFD (ae3978) and FedICL (ae9571) supports "pretrained-model adaptation is an accepted MST theme". The TSFM angle is still novel because those papers use LLMs, VLMs or tabular FMs, not time-series FMs.
- FedICL's frozen tabular-FM in-context baseline is a natural extra baseline idea: frozen embeddings plus a light classifier, or in-context classification.

### Gaps
- The name search covers titles and bibliographic metadata only. An MST paper might use MOMENT or Chronos as a *baseline* inside the full text without naming it in the title or abstract, and that cannot be ruled out.
- "Mantis" and "Time-LLM" do not match cleanly in Crossref: the Time-LLM query matched generic "LLM" titles. No MST paper by that name was found.

---

## Q4. What does a typical MST fault-diagnosis paper look like structurally, and which 3–5 papers are the best templates?

### Takeaway
From the one fully read MST PD paper (ad366c) plus metadata across the set, a typical MST fault-diagnosis full paper has **~10–15 journal pages, 5–7 numbered sections, about 8–15 figures, 3–6 tables and 25–55 references**. Reviews run much longer (174 refs). MST pages put a dedicated **experimental-procedure / measurement-setup section right after the Introduction**, with apparatus parameters, specimen construction and data-collection protocol, before any modelling.

### Cited Findings
- **ad366c (Chang & Boyanapalli 2024), full structure:**
  - Size: 11 pages; ~6,100 words of extracted text including references; **9 figures, 4 tables, 21 references** — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
  - Section headings: 1 Introduction; **2 PD experimental procedure** (2.1 setup with RLC detector, divider, 20 MHz ADC; 2.2 Fabrication of artificial defects; 2.3 PD data collection from defect samples); 3 PRPD patterns and analysis with different PD measurement cycles (3.1 Phase resolved patterns; 3.2 PD measurement PRPD patterns at measurement cycles); 4 Defect recognition using CNNs at different PD measurement cycles; 5 Results of PRPD pattern recognition (5.1 Training…; 5.2 Testing…; 5.3 Training and testing accuracies comparison…); 6 Conclusion; Data availability statement — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
  - About 2 of the 11 pages (Section 2, figures 1–2, table 1) go to the measurement setup and specimens — [IOP PDF](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf)
- **Reference counts as a length proxy:**

  | Paper | References |
  |---|---|
  | ac27e8 | 30 |
  | ac7a09 | 30 |
  | acc1fc | 30 |
  | ad3412 | 30 |
  | ad7488 | 28 |
  | adfb00 | 32 |
  | ae185f | 54 |
  | ae8e99 | 41 |
  | ad2667 | 39 |
  | ae1cde | 32 |
  | ae3978 (LLM-VSFD) | 62 |
  | ae9571 | 43 |
  | aea58b | 57 |
  | ae6a0c (review) | 174 |

  Sources: [Crossref ac27e8](https://api.crossref.org/works/10.1088/1361-6501/ac27e8), [Crossref ae185f](https://api.crossref.org/works/10.1088/1361-6501/ae185f), [Crossref ae3978](https://api.crossref.org/works/10.1088/1361-6501/ae3978), [Crossref ae6a0c](https://api.crossref.org/works/10.1088/1361-6501/ae6a0c)
- Baseline and ablation pattern visible in abstracts:
  - TPCAN compares against **4 named baselines** (CoAtNet, ResNet, 1-DCNN, LSTM), with a module ablation and an attention-head sensitivity study — [IOP adfb00](https://iopscience.iop.org/article/10.1088/1361-6501/adfb00)
  - The Siamese bearing paper uses **4 baselines** (SVM, DCNN, WDCNN, CNN-BiGRU) — [IOP ad57d9](https://iopscience.iop.org/article/10.1088/1361-6501/ad57d9)
  - AWCDN (2025) claims to beat **seven** DL models on two bearing datasets — [IOP ade0e2](https://iopscience.iop.org/article/10.1088/1361-6501/ade0e2)
  - CoFAE-LLM tests on 3 public datasets plus ablations — [IOP ae612d](https://iopscience.iop.org/article/10.1088/1361-6501/ae612d)
  - Tri-MSTCN uses **3 baselines** (GRU, CBG, TCN) plus a sample-size sweep (45 to 135) — [IOP ae7139](https://iopscience.iop.org/article/10.1088/1361-6501/ae7139)

**Recommended structural templates (3–5), and why:**
1. **Chang & Boyanapalli 2024 (ad366c)**, cable-joint PRPD + CNN. Open access with full details. It is the best template for the **"PD experimental procedure"** section: specimen fabrication with dimensions, detector bandwidth, divider ratio, ADC rate, voltage protocol, a per-class sample-count table, and an experiment built around a *measurement parameter*. Weakness to avoid copying: no specimen-wise split and no named DL baselines — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c)
2. **Chen et al. 2026 (ae8e99)**, transformer PD with Optical/UHF/HFCT and pulse-sequence analysis. It is open access and the most recent. It is the closest MST match to **pulse/time-series PD input from a lab platform with 5 defect models**, and it reports **Macro-F1 under class imbalance** and multi-sensor synchronisation. Use it for the measurement-platform and imbalance-metric framing — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae8e99)
3. **Liu et al. 2025 (adfb00)**, TPCAN PRPD + waveform fusion in GIS. Use it for the **experiments section layout**: 4 strong baselines including a modern hybrid (CoAtNet), a module ablation, and a hyperparameter sensitivity analysis (attention heads ≈ adapter rank / phase-anchor count in the user's paper). Note that it is closed access — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/adfb00)
4. **Wang et al. 2023 (acc1fc)** or **Yan et al. 2025 (ae185f)**, low-data GIS insulation-defect diagnosis. Use these for **framing data scarcity and lab-to-field transfer in PD** within MST. Episode training (acc1fc) and "93.26% with only 6% labels" (ae185f) show how MST PD papers headline label efficiency — [IOP acc1fc](https://iopscience.iop.org/article/10.1088/1361-6501/acc1fc); [IOP ae185f](https://iopscience.iop.org/article/10.1088/1361-6501/ae185f)
5. **Wu et al. 2026 (ae3978, LLM-VSFD)**. Use it for **adapting a large pretrained model to few-shot sensor time series** in MST: a two-stage align-then-LoRA pipeline, a 5-shot headline number, and cross-dataset generalisation. It is the closest MST methodological analogue to TSFM adaptation — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae3978)

### Inferences
- A target outline that matches MST norms:
  1. Introduction, with the novelty/significance framing MST requires
  2. PD measurement system and dataset: specimens, IEC 60270 / HFCT chain, sampling, phase reference, per-class counts table, uncertainty of the measurement chain
  3. Method: TSFM backbone, phase anchoring, adaptation
  4. Experimental protocol: specimen-wise split, N-way K-shot episodes, repeated runs
  5. Results with 4–7 baselines, ablations and sensitivity
  6. Discussion, including measurement-parameter effects such as cycles per sample and SNR
  7. Conclusion
- Aim for about 10–14 pages, 10–14 figures, 4–6 tables and 35–50 references, sitting between the 2021–2024 GIS papers (~30 refs) and the 2026 LLM papers (57–62 refs).

### Gaps
- Figure/table counts and page counts for every paper except ad366c are **not found**, because PDF access was blocked. The numbers above for the "typical" paper extrapolate from one full text plus reference counts; they are not measured across the set.
- Word counts for MST full papers are not stated on any MST page I could access.

---

## Q5. MST scope expectations and review/publication timeline

### Takeaway
MST requires an **advance in measurement science or technique**, not routine application of ML, plus explicit **consideration of uncertainty, precision and/or accuracy**, and a novelty/significance statement under 100 words. ML/AI is in scope as an "advanced measurement tool", and fault diagnosis/condition monitoring is a listed subject area. Observed submission-to-acceptance times for the PD papers checked are about **2–4.5 months**, with online publication 1–3 weeks after acceptance.

### Cited Findings
- Scope: MST "publishes articles on new measurement techniques and associated instrumentation". Papers describing experiments "must represent an advance in measurement science or measurement technique". "All submitted articles should contain consideration of the uncertainty, precision and/or accuracy". A statement under 100 words on novelty, significance and broader relevance is required — [IOP Publishing Support: About MST](https://publishingsupport.iopscience.iop.org/?p=3564)
- Subject areas include fault diagnosis and condition-based maintenance under control/automation. "Advanced measurement tools" covers "application of machine learning, artificial intelligence, neural networks, digital twinning" to measurement science — [IOP Publishing Support: About MST](https://publishingsupport.iopscience.iop.org/?p=3564)
- Article types: full papers, technical design notes (normally ≤2,500 words, about 3 pages), topical reviews, focus collections. Peer review is single- or double-anonymous at the author's choice. The journal promises "rapid first decision", but no numeric median is published on that page — [IOP Publishing Support: About MST](https://publishingsupport.iopscience.iop.org/?p=3564)
- Observed timelines:

  | Paper | Received | Accepted | Online |
  |---|---|---|---|
  | acc1fc | 2 Jan 2023 | 7 Mar 2023 (~9 weeks; 2 revision rounds) | 23 Mar 2023 |
  | ad366c | 11 Jan 2024 | 21 Mar 2024 (~10 weeks) | 28 Mar 2024 |
  | adfb00 | 30 Mar 2025 | 13 Aug 2025 (~4.5 months) | 26 Aug 2025 |
  | ae8e99 | 14 May 2026 | 22 Jul 2026 (~10 weeks) | 10 Aug 2026 |

  Sources: [IOP acc1fc](https://iopscience.iop.org/article/10.1088/1361-6501/acc1fc); [IOP PDF ad366c](https://iopscience.iop.org/article/10.1088/1361-6501/ad366c/pdf); [IOP adfb00](https://iopscience.iop.org/article/10.1088/1361-6501/adfb00); [IOP ae8e99](https://iopscience.iop.org/article/10.1088/1361-6501/ae8e99)
- Impact factor: 3.7 (2026 release, 2025 data) and 3.4 for the prior year, per a third-party aggregator; not verified against Clarivate — [JournalMetrics](https://www.journalmetrics.org/journal/measurement-science-and-technology). Wikipedia gives 2.7 for 2023 — [Wikipedia](https://en.wikipedia.org/wiki/Measurement_Science_and_Technology)
- An apparent special-issue signal: FedICL (ae9571) states it "directly addresses the special issue themes of incomplete data and federated learning" — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae9571)

### Inferences
- To pass MST's scope test, the PD-TSFM paper should foreground measurement aspects:
  - calibration and description of the PD acquisition chain (IEC 60270 charge calibration, HFCT/UHF bandwidth, sampling rate, phase-reference synchronisation)
  - the effect of acquisition parameters (cycles per sample, SNR, trigger threshold) on few-shot accuracy
  - **uncertainty reporting**: mean ± std or 95% CI over repeated episodes and seeds, plus specimen-wise splits
- "Phase-anchored" can be pitched as a measurement-informed representation: the power-frequency phase reference *is* a measured quantity. That positions it as an advance in measurement technique rather than pure ML.
- Expect about 2–5 months to acceptance, based on four data points only.

### Gaps
- No official IOP median for "submission to first decision" or "submission to acceptance" for MST was found. The timelines above are a sample of four papers.
- An MST author-guidelines page with explicit length limits for full papers was not retrieved; only the technical-design-note limit (2,500 words) is stated.
- The current MST special-issue call (incomplete data / federated learning, inferred from ae9571) was not located.
