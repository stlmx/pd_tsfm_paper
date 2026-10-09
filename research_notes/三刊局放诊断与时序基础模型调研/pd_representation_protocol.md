# PD Signal Representations, Phase-Resolved Physics, Prior Phase-Aware Deep Models, Leakage-Free Evaluation Protocols, and Public Datasets (plus HV Circuit-Breaker Event-Anchored Signals)

Scope note: research done 2026-10-09. Several publisher pages (MDPI, ScienceDirect, Kaggle, IEC/ANSI previews) returned HTTP 403 to the fetch tool, so some items below come from search-result summaries or from PDFs that were downloaded and text-extracted. Each item says where it came from. Anything not verified is in "Gaps". No DOI below was guessed. A DOI is listed only when it appeared on a source page or in a reference list that was read.

---

## Q1. Standard PD data representations (raw pulse, TRPD, PRPD, PRPS, PSA, statistical operators): what physical information each carries, plus standards and reviews

### Takeaway
PD data has two layers. The first is pulse-level: the raw waveform, its time-frequency content, and TRPD, which carry information about the sensor, coupling path and propagation. The second is ensemble-level: phase-resolved representations (PRPD φ-q-n; PRPS, a cycle-by-cycle phase-resolved sequence; PSA, which correlates consecutive pulses), which carry the physics of the defect. The classic Gulski-school operators (skewness, kurtosis, peaks and +/- half-cycle cross-correlation of Hqmax(φ), Hqn(φ) and Hn(φ)) are hand-made summaries of the phase-resolved layer. A Measurement paper shows that these fingerprints are sensitive to the phase-bin width, so the phase-window size is a real design and robustness variable.

### Cited Findings
**Standards**
- IEC 60270 now exists as **Edition 4.0 (2025-06)**, retitled "High-voltage test techniques – Charge-based measurement of partial discharges". Its contents list separate subsections for wide-band PD instruments with an active integrator (4.3.5) and narrow-band PD instruments (4.3.6). This comes from a search-result summary of the IEC preview. A direct fetch returned 403, and the numeric limits are not visible in the preview. — [ANSI/IEC preview of IEC 60270 ed4.0](https://webstore.ansi.org/preview-pages/IEC/preview_iec60270%7Bed4.0%7Db.pdf)
- A patent that paraphrases the older IEC 60270 gives the recommended wide-band values as lower cut-off f1 = 30–100 kHz, upper cut-off f2 < 500 kHz and bandwidth Δf = f2 − f1 = 100–400 kHz, measured at 6 dB below mid-band. The often-quoted "Δf ≤ 900 kHz / f2 ≤ 1 MHz" figures could not be verified from any retrieved text. — [US8990035B2 (Google Patents)](https://patents.google.com/patent/US8990035)
- IEC TS 62478:2016 ("High voltage test techniques – Measurement of partial discharges by electromagnetic and acoustic methods") applies to electromagnetic (HF/VHF/UHF) and acoustic PD measurements in electrical apparatus insulation. It covers sensors with different frequency ranges and sensitivities, PD location, and calibration or sensitivity checks. — [Standard Norge listing of IEC TS 62478:2016](https://online.standard.no/en/iec-ts-62478-2016-ed1); [BSI listing](https://knowledge.bsigroup.com/products/high-voltage-test-techniques-measurement-of-partial-discharges-by-electromagnetic-and-acoustic-methods)
- Band definitions used in PD practice are HF 3–30 MHz and VHF 30–300 MHz. The conventional electrical method is described in IEC 60270. — [Energies review, Univ. Stuttgart repository (energies-16-01395)](https://elib.uni-stuttgart.de/bitstream/11682/12795/1/energies-16-01395-v2.pdf). CIGRE WG D1.37 defines "VHF = 30 to 300 MHz, UHF = 300 MHz to 3 GHz" (footnote, cable section). — [CIGRE WG D1.37, *Guidelines for PD detection using conventional (IEC 60270) and unconventional methods*, 2016](https://cigre.cz/dokumenty_komise/d1/WG%20D1.37_TB_Final.pdf)
- For UHF PD in transformers, most publications report a spectrum of 200 MHz–1 GHz, and a practical "common" band of about 400–900 MHz, though the band and peak power "strongly vary from case to case". — [Energies review (Stuttgart repository)](https://elib.uni-stuttgart.de/bitstream/11682/12795/1/energies-16-01395-v2.pdf)

**What phase-resolved patterns carry (authoritative guideline)**
- CIGRE WG D1.37 (2016; members include E. Gulski as convenor, S. Tenbohlen and S. Kornhuber) states the following. "The phase-resolved PD pattern is reflecting the physical phenomena (statistical behavior) of the specific PD-sources." Starting-electron availability depends on the source, the insulation material and the source's position relative to the electrodes. "Five typical PD-patterns can be defined." The same source measured by IEC 60270, UHF or acoustic methods should in theory give similar PRPD patterns once propagation times and attenuation are normalised. In practice, noise, superimposed sources and changing amplitudes mean "the five typical PD patterns appear in many variations". — [CIGRE WG D1.37 guideline, §4.1](https://cigre.cz/dokumenty_komise/d1/WG%20D1.37_TB_Final.pdf)
- The same guideline lists three reasons why PRPD helps identify a PD source. It helps identify the type of PD source. "Individual PD patterns are not influenced by the signal transfer function of the extended insulating system." It can separate superimposed defects by their different statistical behaviour. It adds that pulse shape and frequency content (§4.1.1) give "fundamental information about the signal source". — [CIGRE WG D1.37 guideline](https://cigre.cz/dokumenty_komise/d1/WG%20D1.37_TB_Final.pdf)

**TRPD / PRPD / PRPS**
- TRPD (time-resolved), PRPD (phase-resolved) and PRPS (phase-resolved pulse sequence) are the three common PD "image" types. PRPD shows the relation between discharge phase, discharge magnitude and discharge count. — [Symmetry 2022, 14(11), 2464, GIS PD recognition by multi-feature fusion of PRPD image, doi:10.3390/sym14112464](https://doi.org/10.3390/sym14112464)
- PRPS as used with UHF sensors is a 3D graph (phase × cycle index × amplitude). One recent example builds PRPS from **50 power-frequency cycles sampled every 5°** and classifies suspended-electrode, surface and metal-tip discharges with a simple CNN plus a quadratic SVM. This is from a search summary of the paper. — [Energies 2024, 17(11), 2443, doi:10.3390/en17112443](https://doi.org/10.3390/en17112443)
- UHF monitoring devices commonly output both PRPD and PRPS spectra. — [ResearchGate: PD pattern recognition in power transformers based on CNN (2017)](https://www.researchgate.net/publication/320096992_Partial_Discharge_Pattern_Recognition_in_Power_Transformers_Based_on_Convolutional_Neural_Networks)
- A deep-learning PD survey reports that PRPD is typically accumulated over about 50 cycles. One CNN study used time-domain waveform images spanning five consecutive cycles, and the survey suggests that raw pulse sequences or spectrogram-like inputs could beat summary patterns. This is from a search summary, and the authors were not verified. — [Energies 2019, 12(13), 2485, "PD Classification Using Deep Learning Methods—Survey of Recent Progress"](https://www.mdpi.com/1996-1073/12/13/2485)

**Statistical operators (Gulski school)**
- Phase-resolved distributions are built from the mean pulse height Hqn(φ) and the pulse count Hn(φ). Each is split into positive and negative half-cycle distributions, which gives four distributions. Their shapes are described by skewness (asymmetry relative to a normal distribution) and kurtosis (sharpness relative to a normal distribution). The same operators are applied to the maximum-charge distribution Hqmax(φ). A cross-correlation factor measures the difference in shape between the (+) and (−) half-cycle sets, and the number of peaks separates single-peak from multi-peak distributions. This is from a search summary of a scanned TU Delft document; the full text could not be extracted. — [TU Delft repository document (Gulski)](https://repository.tudelft.nl/file/File_ac86b0e1-b6d5-4ecc-bef9-a1a2c15e28ca?preview=1)
- Classic references, as printed in the reference list of Mas'ud et al. (text-extracted):
  - E. Gulski & F. Kreuger, "Computer aided analysis of discharge patterns", vol. 23 (1990) pp. 1569–1575. The list prints the venue as "Journal of Applied Physics", but volume and pages match *J. Phys. D: Appl. Phys.*, so verify.
  - E. Gulski & A. Krivda, "Neural networks as a tool for recognition of partial discharges", *IEEE Trans. Electr. Insul.* 28(6) (1993) 984–1001.
  - A. Krivda, "Automated recognition of partial discharges", *IEEE TDEI* 2(5) (1995) 792–821.
  - W.J.K. Raymond, H.A. Illias, A.H.A. Bakar, H. Mokhlis, "Partial discharge classifications: review of recent progress", *Measurement* 68 (2015) 164–181.

  — [Mas'ud, Stewart, McMeekin, accepted manuscript on Strathprints](https://strathprints.strath.ac.uk/58465/1/Mas_ud_etal_MJIMC_2016_An_investigative_study_into_the_sensitivity_of_different_partial.pdf)
- Gulski and Krivda set 95% mean confidence intervals on statistical features for artificial two-electrode defect classes. Those tolerances assumed fixed phase and amplitude resolutions of the φ-q-n patterns. — [Mas'ud et al., Strathprints](https://strathprints.strath.ac.uk/58465/1/Mas_ud_etal_MJIMC_2016_An_investigative_study_into_the_sensitivity_of_different_partial.pdf)
- **Phase-resolution sensitivity (Measurement, 2016, per the Strathprints filename "MJIMC_2016"; DOI not retrieved).** Mas'ud, Stewart and McMeekin captured φ-q-n patterns at **1° phase resolution (PR) and 100 amplitude bins (AB)**. They re-binned them to **3°, 6°, 9°, 12° and 15° PR** and to **50 and 25 AB**. Defects were surface discharge in air, a single void in PET, corona in air and surface discharge in oil, measured per IEC 60270. The statistical fingerprints differed across PR and AB sizes, "most significant in the Hn(q)+ and Hqn(φ)− plots but less significant in the Hn(φ) plots". The authors warn that "care has to be taken not to simply train SNN or ENN with any PD φ-q-n resolution data and test with another φ-q-n resolution data". Because instruments use different φ-q-n settings, "certain φ-q-n PR and AB sizes be maintained for consistency of recognition". Positive corona in air was described as having "low repetition rate and higher amplitude". — [Mas'ud et al., Strathprints PDF](https://strathprints.strath.ac.uk/58465/1/Mas_ud_etal_MJIMC_2016_An_investigative_study_into_the_sensitivity_of_different_partial.pdf)

**Pulse sequence analysis (PSA)**
- PSA was introduced by Patsch and Hoof in 1995. It uses consecutive pulses to derive differences in discharge magnitude and time, plotted as x-y PSA diagrams. Both 2-pulse and 3-pulse forms exist in the literature. — [TU Delft repository thesis (via search summary)](https://repository.tudelft.nl/file/File_0e693293-34b4-4741-a3b0-f73624b68938?preview=1)
- "Sequence correlated parameters such as the voltage differences of the applied voltage or time differences between consecutive discharges are far more decisive parameters." A systematic shift in the phase angles of occurrence produces an apparent "statistic scatter" of phase angles unless the correlation between consecutive discharges is taken into account. — [J. Phys. D, doi:10.1088/0022-3727/35/1/306](https://www.doi.org/10.1088/0022-3727/35/1/306)
- In power transformers, PSA patterns of different PD sources form different clusters, but multiple sources or the test-voltage amplitude strongly influence the clustering. — [Pfeffer, Tenbohlen, Kornhuber, ISH 2011](https://www.ieh.uni-stuttgart.de/dokumente/publikationen/2011_ISH_Pfeffer_Pulse-Sequence_Analysis.pdf)
- Under DC stress phase information is missing, so PSA is used instead of phase-resolved analysis to identify defects. — [SINTEF publication on HVDC GIS (via search summary)](https://www.sintef.no/en/publications/publication/01a07be51ee1-c4f1a45f-c234-41ac-974a-6e6500d7ca4c/)

### Inferences
- **Representations by physical content.** (a) The raw pulse waveform and spectrum (TRPD) encode the sensor, coupling and propagation path more than the defect, because the CIGRE guideline says PRPD, unlike the waveform, is not affected by the transfer function. (b) PRPD (φ-q-n) encodes the defect's field and inception physics, but accumulation discards the cycle order. (c) PRPS keeps the cycle order, so it holds intermittency and slow evolution such as particle hopping and floating-electrode bursts. (d) PSA keeps pulse-to-pulse memory effects. A "phase-anchored" interface (one patch per cycle or phase window, plus a phase positional encoding) effectively turns raw or PRPS data into a learnable PRPS/PSA hybrid. That is a defensible physical motivation.
- The Mas'ud et al. result gives a ready, citable justification for (i) choosing the phase-window width deliberately (common choices include 1°, 3.6° = 100 bins, 5°, and 6°) and (ii) adding a **cross-phase-resolution robustness test** (train at one bin width, test at another) as an instrument-setting shift axis. Reviewers at *Measurement* will recognise this paper.
- Because IEC 60270 was revised in 2025 (Ed. 4.0), cite the edition you actually used, and avoid quoting bandwidth numbers from memory.

### Gaps
- Exact IEC 60270 (Ed. 3.0/2000+A1:2015 and Ed. 4.0/2025) bandwidth limits and the definitions of the phase angle φi and time ti were not retrieved because the full text is paywalled. Check against the standard itself.
- Whether IEC TS 62478 itself defines the HF/VHF/UHF band limits (rather than secondary sources) was not verified.
- The CIGRE WG D1.37 PDF does not print its Technical Brochure number. It is widely cited as TB 662, but this was not verified here.
- Original Gulski operator formulas, such as the exact definitions of Q, cc, mcc and Pe, were not extracted because the TU Delft file is a scanned image.
- The full author list of the Energies 2019 deep-learning PD survey was not verified.

---

## Q2. How PD defect types differ in phase distribution: the physical basis for phase alignment

### Takeaway
PD occurs only when the local field exceeds the inception level and a starting electron is available. Both depend on the instantaneous applied voltage, its polarity, and the space or surface charge left by earlier discharges. That is why each defect class occupies characteristic phase regions and shows half-cycle asymmetry when referenced to the power-frequency cycle.
- Corona and protrusion activity is polarity-asymmetric (negative half-cycle first).
- Voids are roughly symmetric in both half-cycles ("rabbit ears").
- Surface discharge appears as "two hills".
- Floating electrodes give high-count, high-amplitude clusters.
- Free particles are only weakly phase-correlated when hopping.

Published phase ranges disagree across sensors and setups, so phase rules are not universal. This is an argument for learning phase-conditioned features rather than hard-coding them.

### Cited Findings
- Two conditions are needed for PD. The field strength must exceed the withstand level near the defect, and free electrons must be available to start and sustain the discharge. The second condition leads to "the stochastic, non-regular behavior of PD signals". — [CIGRE WG D1.37 guideline, §2.1](https://cigre.cz/dokumenty_komise/d1/WG%20D1.37_TB_Final.pdf)
- Typical GIS sources, per CIGRE WG D1.37:
  - **Moving or "hopping" metallic particles** (millimetre range). These move under the field, are "relatively easy to detect" by conventional and unconventional methods, and are "considered a relatively high risk because of their unpredictability".
  - **Floating potential.** The component forms a capacitive divider that sparks over "each time the potential difference across the gap exceeds the withstand strength", then recharges. It usually gives "high amplitude and high pulse count signals".
  - **Protrusions.** These give PD with "the well-known characteristics of corona discharge" and are usually found at factory test.
  - **Voids, cracks and delamination in solid insulation.** A missing start electron causes long inception delays. Small voids (<1 mm) give very low amplitude (<<10 pC) with high variation in activity over time.
  - **Surface discharge from contamination.** Signals are relatively high in magnitude but highly variable.

  — [CIGRE WG D1.37 guideline, §2.1](https://cigre.cz/dokumenty_komise/d1/WG%20D1.37_TB_Final.pdf)
- **Measured phase distributions in gas-insulated equipment** (Kong et al., *Front. Energy Res.* 10:937599, 2022, doi:10.3389/fenrg.2022.937599):
  - **Floating defect** (PDIV 36.17 kV). UHF pulses fall at 0–90° and 180–270°, with the positive half-cycle dominant (max count 79). HFCT shows 0–90° and 270–360° at about 1,200 pC.
  - **Void.** UHF detects it at 8.82 kV; the pattern "looks like two rabbit ears", concentrated at the peaks of both half-cycles, with the positive side stronger. HFCT first detects it at 15.24 kV (max 182 pC, positive-dominant).
  - **Surface defect.** UHF detects it at 45.7 kV with a pattern like "two hills", roughly symmetric. HFCT detects it at 50.25 kV, concentrated at 0–90° and 180–270°.
  - Acoustic emission detected only the floating defect.

  — [Kong et al. 2022, Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.937599/full)
- **Corona polarity asymmetry.** Negative corona typically incepts on the negative half-cycle before positive corona, with Trichel pulses concentrated around the negative peak. Positive corona appears at higher voltage and signals higher flashover risk. Trichel pulses occur only in electronegative gases. This is from search summaries of a Western University thesis and ESA 2010 proceedings, not read in full. — [Western Univ. ETD 4167](https://ir.lib.uwo.ca/etd/4167); [ESA 2010, Sattari](https://electrostatics.org/wp-content/uploads/2023/08/ESA2010_K4_Sattari.pdf). Positive corona in air has a "low repetition rate and higher amplitude", which is why Mas'ud et al. separated positive and negative corona φ-q-n patterns. — [Mas'ud et al.](https://strathprints.strath.ac.uk/58465/1/Mas_ud_etal_MJIMC_2016_An_investigative_study_into_the_sensitivity_of_different_partial.pdf)
- A vendor blog places negative corona at about 250°–290°. This is non-peer-reviewed; use only as a heuristic. — [Hertzinno blog](https://hertzinno.com/blog-detail/how-to-interpret-prpd-patterns-for-corona,-surface,-and-floating-potential-discharges). Another blog says voids cluster on the rising slopes (about 0°–90° and 180°–270°, ahead of the peaks), which is not fully consistent with the "peaks" description in Kong et al. — [insulationtesting.com](https://insulationtesting.com/partial-discharge-prpd-patterns/). Blogs also disagree on whether corona sits at zero-crossings or at peaks. — [HV Hipot blog](https://blog.hvhipot.com/2026/05/15/what-are-partial-discharge-patterns-and-how-do-they-affect-high-voltage-systems/)
- **Memory effects.** In a dielectric-bounded cavity, free-electron supply, discharge development and surface-charge decay "bring about memory effects and become the main reasons for stochastic behavior of PDs". Modern cavity PD models largely follow Niemeyer's early-1990s approach. — [Review: Numerical modeling of PDs in a solid dielectric-bounded cavity (CORE)](https://core.ac.uk/works/58286568); [Univ. Southampton thesis](https://eprints.soton.ac.uk/420943/1/Final_Thesis.pdf)
- **Free particles in GIS.** In an acoustic study, static particles gave weak signals with strong phase correlation, while freely jumping particles gave strong signals with weak phase correlation. Flight time depends on particle shape and size. — [XJTU journal, Characteristics of acoustic signal from free moving metallic particles in GIS](https://zkxb.xjtu.edu.cn/en/article/75013447/). Separately, UHF measurements combined with high-speed video show that different particle movement modes produce distinct UHF pulse patterns (repetition rate and amplitude). PSA diagrams were used to infer particle motion. — [Wenger et al., Univ. Stuttgart IEH, CIGRE Paris 2020](https://www.ieh.uni-stuttgart.de/dokumente/publikationen/2020_08_Wenger_Partial_Discharge_Analysis_in_Gas-Insulated_HVDC-Systems.pdf)
- **Generators.** Hudon and Belec reproduced common generator discharge sources in the laboratory, recorded each with a phase-resolved acquisition system, extracted the dominant PRPD features per source, built a reference database for source recognition, and compared it with about a decade of field data. — *IEEE TDEI* 12(2), April 2005, [TU Delft hosted copy](https://repository.tudelft.nl/file/File_5bc81790-d411-4b3e-b21c-ff4869024dbf)

### Inferences
- **Physical argument for phase anchoring.** PD class is decided by when in the voltage cycle discharges occur (inception phase, polarity, half-cycle asymmetry) and by how activity is correlated across cycles (memory effects, particle hopping). A patching scheme that ignores the cycle boundary, such as fixed-length windows not aligned to the cycle, mixes phase positions across patches. The model then has to re-infer phase from context.
- **Phase-reference fragility must be addressed.** Kong et al. report different phase ranges for the same floating defect on UHF (0–90°/180–270°) and HFCT (0–90°/270–360°). Phase location therefore depends on the sensor and phase reference (coupling delay, which phase conductor is used as reference). Design implications:
  - Use a cyclic encoding of φ, such as sin/cos, which keeps 0°/360° continuity.
  - Evaluate robustness to a global phase offset, e.g. ±10–30°, and possibly apply random phase-shift augmentation.
  - Report results with and without a synchronised voltage reference.
- Encoding the relationship between φ and φ+180° (half-cycle symmetry) mirrors the Gulski cross-correlation operator. That gives a physically interpretable inductive bias.

### Gaps
- A single authoritative, peer-reviewed table mapping each defect class (void, surface, corona, floating, free particle) to phase ranges under AC was not found. CIGRE TB 444/525/654-type GIS documents and IEC TS 60034-27-1/-2 (rotating machines) likely contain such reference patterns but were not retrieved.
- Niemeyer's original model papers (phase-position predictions) were not retrieved, so no DOI is given.
- Hudon and Belec 2005: the DOI and the specific PRPD signatures per source were not extracted.

---

## Q3. Prior phase-aware / cycle-synchronous / Transformer models for PD (positioning and novelty risk)

### Takeaway
Many works feed PRPD/PRPS images to CNN, ViT or Swin models, or feed PRPD sequences to LSTMs or self-attention. GIS has used this route since 2018 (LSTM, Energies) and 2020 (self-attention, Energies). A 2025 EPSR paper explicitly says ViT "hard patch partitioning may disrupt the phase continuity" of PRPD spectra. That is very close to the motivation for phase-anchored patching and must be cited and differentiated. No paper was found that:
- (i) patches raw PD signals by power-frequency cycle or phase window for a pretrained time-series foundation model (TSFM), or
- (ii) uses an explicit phase-angle positional encoding.

Treat this as "not found in our searches", not as proven novelty.

### Cited Findings
**Sequence models on phase-resolved data (GIS)**
- Nguyen M.-T., Nguyen V.-H., Yun S.-J., Kim Y.-H., "Recurrent Neural Network for Partial Discharge Diagnosis in Gas-Insulated Switchgear", *Energies* 11(5):1202 (2018), doi:10.3390/en11051202. An LSTM takes PRPD as input and learns temporal features directly, reaching 96.74% accuracy versus SVM baselines. — [doi.org](https://dx.doi.org/10.3390/en11051202); [IDEAS](https://ideas.repec.org/a/gam/jeners/v11y2018i5p1202-d145364.html)
- Tuyet-Doan V.-N., Nguyen T.-T., Nguyen M.-T., Lee J.-H., Kim Y.-H., "Self-Attention Network for Partial-Discharge Diagnosis in Gas-Insulated Switchgear", *Energies* 13(8):2102 (2020), doi:10.3390/en13082102. A self-attention block processes parts of the PRPD signal in parallel. It is motivated by the LSTM's lack of parallel computation and its inability to weigh the relevance of all inputs, and it is compared with LSTM+self-attention. — [MDPI](https://www.mdpi.com/1996-1073/13/8/2102); [DOAJ](https://doaj.org/article/42cf9c7d8112447fa91cc0b5bdb71e72)
- A Chinese-language framework combines LSTM with multi-head attention for GIS PD detection. — [J. Henan Polytechnic Univ., doi:10.16186/j.cnki.1673-9787.2023020042](https://chinacaj.net/JGXB/doi/10.16186/j.cnki.1673-9787.2023020042)
- A *Mechanics & Industry* 2026 paper (J. Wang et al., doi:10.1051/meca/2026001) argues that PD patterns "are essentially time-series data that vary periodically with voltage phase", so an LSTM can capture the dynamics of their phase distribution and amplitude evolution. This is from a search summary; the page returned 403. — [Mechanics & Industry](https://www.mechanics-industry.org/10.1051/meca/2026001)

**Transformers on PRPD/PRPS images**
- Liu Y., Li W., Hua X., Liu H., Li H., Zhang G.J., "A Swin transformer enhanced depthwise convolution with dynamic channel reweighting for partial discharge diagnosis in gas insulated switchgear" (CSEST), *Electric Power Systems Research* 248:111993 (2025), doi:10.1016/j.epsr.2025.111993. The abstract states that ViT "hard patch partitioning may disrupt the phase continuity in spectrum, compromising feature integrity". This motivates shifted-window attention. — [XJTU Scholar page](https://scholar.xjtu.edu.cn/en/publications/a-swin-transformer-enhanced-depthwise-convolution-with-dynamic-ch/)
- "A Study on PD Fault Identification in GIS Based on Swin Transformer-AFPN-LSTM Architecture", *Information* 16(2):110 (2025), Qinghai University. It combines PRPD phase data with TRPD timing data converted to MTF images, and reports improvements of 14.09–21.23% over its comparison models. From a search summary; DOI not seen. — [MDPI Information](https://www.mdpi.com/2078-2489/16/2/110)
- PMSNet, a Transformer encoder with spatial-interaction attention on PRPD maps, *Sensors* 2024. — [PMC11174456](https://pmc.ncbi.nlm.nih.gov/articles/PMC11174456/)
- Vision Transformer for PD pattern recognition in power transformers, *J. Phys.: Conf. Ser.* 2903 (2024) 012010. — [ADS](https://ui.adsabs.harvard.edu/abs/2024JPhCS2903a2010H/abstract)
- Swin Transformer for power-cable PD recognition, *Processes* 13(3):852 (2025). — [MDPI Processes](https://www.mdpi.com/2227-9717/13/3/852)
- 3D PRPS (50 cycles × 5° bins) with simple CNN + QSVM. — [Energies 2024, 17(11), 2443](https://doi.org/10.3390/en17112443)
- Cable PD with a diffusion denoiser plus morphological attention. Its sinusoidal positional encoding is for the diffusion timestep, not for phase. A Swin baseline reached 93.25% but struggled with noise. — [Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-25197-9)

**Raw 20 ms cycle signals (VSB)**
- Michau G., Hsu C.-C., Fink O., "Interpretable Detection of Partial Discharge in Power Lines with Deep Learning", *Sensors* 21(6):2154 (2021), doi:10.3390/s21062154. Key points:
  - Each phase of a 50 Hz period is processed independently by the same 1D CNN and combined at line level.
  - Pulses are extracted and "ordered by decreasing amplitudes and not by their relationships in the signal", so phase position is discarded, although the authors "expect some periodicity in the occurrence of the pulses with respect to the utility frequency".
  - The Pulse Activation Map explains which pulse drove the decision, not where in the cycle it occurred.

  — [PMC8003486](https://pmc.ncbi.nlm.nih.gov/articles/PMC8003486/)
- Tehrani P., Levorato M., arXiv:2009.06825 (2020). On VSB, a high-pass filter "flattens the voltage signal and remove[s] the signal phase". Peak extraction then reduces 800,000 samples to hundreds, followed by LSTM-attention plus a 1D-CNN on frequency features. — [arXiv 2009.06825](https://arxiv.org/pdf/2009.06825) (text extracted)

**Few-shot / self-supervised / domain adaptation for PD (neighbouring claims)**
- Metric-based meta-learning for few-shot GIS PD diagnosis: hybrid self-attention CNN plus episodic training, 93.17% in 4-way 5-shot. — [XJTU Scholar](https://scholar.xjtu.edu.cn/en/publications/novel-metric-based-meta-learning-model-for-few-shot-diagnosis-of-/)
- Fourier-enhanced prototype contrastive learning for GIS PD diagnosis and localisation, reporting >95% on real GIS data. — [XJTU Scholar](https://scholar.xjtu.edu.cn/zh/publications/fourier-enhanced-prototype-contrastive-learning-for-partial-disch/)
- Siamese few-shot GIS diagnosis, Chinese title. — [XJTU Scholar](https://scholar.xjtu.edu.cn/en/publications/%E5%9F%BA%E4%BA%8E%E5%A4%9A%E7%BA%A7%E4%BA%8C%E9%98%B6%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%AD%AA%E7%94%9F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%B0%8F%E6%A0%B7%E6%9C%AC-gis-%E5%B1%80%E9%83%A8%E6%94%BE%E7%94%B5%E8%AF%8A%E6%96%AD%E6%96%B9%E6%B3%95/)
- Self-supervised representation learning for GIS PD (multi-task diagnosis, localisation and severity), reportedly comparable to supervised learning. — [XJTU Scholar](https://scholar.xjtu.edu.cn/en/publications/self-supervised-representation-learning-for-gis-partial-discharge/). A *High Voltage Engineering* paper reports a 9.34% average gain from self-supervised pretraining in low-label settings. This is from a search summary, and the mapping to this exact link is uncertain. — [HVE doi:10.13336/j.1003-6520.hve.20231406](https://epjournal.csee.org.cn/gdyjs/article/doi/10.13336/j.1003-6520.hve.20231406)
- Wang Y., Yan J., Yang Z., Wang J., Geng Y., "Deep Domain-Invariant LSTM Network for PD Localization in GIS", *IEEE TPWRD* 38(4):2810–2820 (2023), doi:10.1109/TPWRD.2023.3262761. Transfers from digital-twin simulation data to real GIS using MMD plus adversarial adaptation. — [XJTU Scholar](https://scholar.xjtu.edu.cn/zh/publications/deep-domain-invariant-long-short-term-memory-network-for-partial-/)
- A subdomain adaptation capsule network transfers lab to field GIS data, reporting 93.75% on field data. — [Sensors 2023, PMC10217171](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10217171/)

**TSFM context**
- No work applying time-series foundation models (e.g. TimesFM, MOMENT, Chronos, TTM) to PD was found. General TSFM studies warn that zero-shot generalisation "is tightly coupled to the distribution seen during pretraining". — [arXiv 2510.00742](https://arxiv.org/pdf/2510.00742v3). Network-traffic work shows TSFMs can still be strong few-shot learners at resolutions outside their pretraining data. — [Computer Networks (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1389128625003627)
- Background: Transformers have no inherent sense of order, so positional encoding is required. — [Survey, arXiv 2502.12370](https://arxiv.org/pdf/2502.12370)

### Inferences
- **Novelty is narrow.** Phase-resolved deep models (LSTM, attention, ViT/Swin on PRPD/PRPS) and few-shot GIS PD models are crowded areas. The defensible contribution is the interface: cycle-synchronous or phase-window patching of raw or PRPS data plus a cyclic phase positional encoding, plugged into a pretrained TSFM. It should be backed by an ablation showing why anchoring matters (aligned vs. misaligned patches, phase-PE vs. index-PE, and phase-offset robustness).
- CSEST's claim about phase continuity is stated without experimental evidence in its abstract. A controlled ablation that tests it directly would be a clean contribution and a natural "related work" hook.
- The two VSB papers above discard the phase position. An evaluation on VSB that restores it via phase anchoring is a direct, citable contrast.

### Gaps
- It was not verified whether Nguyen 2018 or Tuyet-Doan 2020 use one LSTM or attention step per power cycle, or what positional encoding the self-attention model used (MDPI full text returned 403).
- No paper with an explicit "phase positional embedding" or a "cycle-synchronous patch" for PD was found. Searches were limited to English-language web search, and Chinese journals (高电压技术, 电网技术, 中国电机工程学报) were not systematically searched.
- No PD + TSFM paper was found. A dedicated arXiv/Google Scholar search, e.g. "MOMENT partial discharge" or "foundation model PRPD", is still advisable before claiming novelty.

---

## Q4. Data leakage and evaluation protocols (grouped splits, cross-specimen/sensor/voltage, noise robustness)

### Takeaway
Leakage from random segment or window splits is well documented in fault diagnosis. On CWRU, near-perfect accuracy under segment splits collapses, sometimes below a majority-class baseline, when records or fault instances are held out. Peer-reviewed sources (Kapoor & Narayanan 2023; Hendriks et al. 2022; Matania et al. 2024; Vieira et al. 2026) recommend splitting by physical specimen, bearing or record. No PD-specific study that quantifies this inflation was found. PD papers do acknowledge shortcut learning of setup-specific features, sensitivity to operating voltage, resolution or instrument shift, and lab-to-field gaps. Noise-robustness practice uses WGN, DSI and pulse interference (synthetic, at SNR levels) or real recorded noise, typically training on clean data and testing on contaminated data.

### Cited Findings
**General ML leakage**
- Kapoor S., Narayanan A., "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4(9):100804 (2023), doi:10.1016/j.patter.2023.100804. — as cited in [Matania et al., PHME 2024](https://papers.phmsociety.org/index.php/phme/article/download/4125/2462)
- Whether a partitioning strategy counts as leakage depends "fundamentally on the claims made about model generalizability". — [Artificial Intelligence Review 2025, "Don't push the button! Exploring data leakage risks in ML and transfer learning"](https://link.springer.com/article/10.1007/s10462-025-11326-3)

**Fault diagnosis (bearing) leakage critiques**
- Hendriks J., Dumond P., Knox D.A., "Towards better benchmarking using the CWRU bearing fault dataset", *MSSP* 169:108732 (2022), doi:10.1016/j.ymssp.2021.108732. — reference read in [Matania et al.](https://papers.phmsociety.org/index.php/phme/article/download/4125/2462)
- Matania O., Cohen R., Bechhoefer E., Bortman J., "Test-Training Leakage in Evaluation of Machine Learning Algorithms for Condition-Based Maintenance", *Proc. 8th European Conf. PHM Society* 2024 (p. 825 onward; CC BY 3.0). Findings (text-extracted):
  - Two leakage types are defined. (1) Segments of the same record are split randomly; this is the worse case, since models learn record-specific signatures such as the test-rig transfer function, after SKF evidence (Liefstingh et al. 2021). (2) Records sharing the same fault shape are split randomly.
  - With KNN and random forest, the segment split gives accuracy "close to 90%" on CWRU. The record split drops significantly. Under the correct fault-shape split (leave-one-out), CWRU results were "even lower than the accuracy of the degenerated algorithm, which predicts the training mode constantly".
  - On Paderborn (PU), even the record split remained optimistic.

  — [PHME 2024 PDF](https://papers.phmsociety.org/index.php/phme/article/download/4125/2462)
- Vieira J.P., Bauler V.A., Rosa R.K., Silva D., "Towards a more realistic evaluation of machine learning models for bearing fault diagnosis", *MSSP* 258:114640 (2026), doi:10.1016/j.ymssp.2026.114640 (arXiv:2509.22267). Segment-wise and condition-wise splits "introduce spurious correlations that inflate performance metrics". The authors propose bearing-wise (non-overlapping physical bearings) splits, multi-label reformulation and prevalence-independent ROC metrics. The number of unique training bearings is a decisive factor for generalisation. Datasets: CWRU, Paderborn, UORED-VAFCLS, HUST. — [arXiv abs](https://arxiv.org/abs/2509.22267); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0888327026007971)
- A non-peer-reviewed audit reports random-forest macro-F1 of 1.00 under random window splits on CWRU versus 0.73 (0.64–0.74 over 10 seeds) on unseen bearings. — [GitHub bearing-audit](https://github.com/parvparakhiya/bearing-audit)
- Basic models already reach very high accuracy on CWRU and XJTU-SY, so these benchmarks saturate. — [Zhao et al., open-source benchmark, arXiv 2003.03315](https://arxiv.org/pdf/2003.03315)

**Grouping analogues outside PD**
- Ship acoustics: a random split lets a single ship or temporally adjacent background appear in both train and test. The fix is grouping by vessel identity and recording day. — [UniqueShip, arXiv 2609.13659](https://arxiv.org/pdf/2609.13659)
- Eye tracking: subject-specific features leak when a subject is in both sets, so splits should be subject-exclusive. — [arXiv 2311.16381](https://arxiv.org/pdf/2311.16381)

**PD-specific evaluation issues**
- Shortcut learning: deep PD models "are more likely to learn shallow statistical regularities related to specific experimental settings rather than the underlying physical mechanisms of the discharge". — [Advanced Signal Processing Methods for PD Analysis: A Review (PMC12694323)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12694323/)
- A recent AWA-CNN PD study split train/val/test (60:20:20) *before* augmentation and augmented within each subset to avoid near-duplicate leakage. This handles image-level leakage only, not specimen-level leakage. — [arXiv 2605.21352](https://arxiv.org/pdf/2605.21352)
- Operating-voltage shift: a model trained at the repetitive inception voltage and tested at lower voltages showed that performance "may vary". Longer PRPD windows made the model more robust to voltage-level variation. This is from a search summary, and the attribution to this paper is likely but not verified. — [Applied Sciences 2023, 13(19), 10570, doi:10.3390/app131910570](https://doi.org/10.3390/app131910570)
- Resolution/instrument shift: train/test mismatch in φ-q-n phase or amplitude resolution leads to unreliable predictions. — [Mas'ud et al.](https://strathprints.strath.ac.uk/58465/1/Mas_ud_etal_MJIMC_2016_An_investigative_study_into_the_sensitivity_of_different_partial.pdf)
- Lab-to-field shift: on-site accuracy drops are attributed to overlapping multi-source features and inconsistent sample distributions. Subdomain adaptation reached 93.75% on field data. — [Sensors 2023, SACN, PMC10217171](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10217171/)
- Open-set: high-accuracy GIS and cable-joint classifiers "remain restricted to fixed defect categories". Ageing and fault evolution produce new signatures that must be rejected. — [Robust open-set PD diagnosis, Ain Shams Eng. J. 2025 (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2090447925005039)
- VSB: Michau et al. note that prior work reporting on the full training set "might therefore be overfitted", and they recompute metrics on their own re-split (6,972 train / 1,740 test). That paper does not state whether splits were grouped by measurement ID (the three phases of one line). — [PMC8003486](https://pmc.ncbi.nlm.nih.gov/articles/PMC8003486/)

**Noise-robustness protocols**
- Raymond W.J.K., Illias H.A., Abu Bakar A.H., "Classification of Partial Discharge Measured under Different Levels of Noise Contamination", *PLoS ONE* 12(1):e0170111 (2017), doi:10.1371/journal.pone.0170111.
  - Setup: five 11 kV XLPE cable joints, one artificial defect type each, measured at 9 kV with an Omicron MPD600, 100 one-minute measurements per joint.
  - Noise: real ground-interference noise recorded during lightning (up to about 250 pC, random phase) was added for 5–60 s, because amplitude could not be controlled. Classifiers were trained on clean data and tested on contaminated data.
  - Results: ANN with statistical features fell from 91.8% to 42.0%, ANFIS from 91.4% to 21.4%, and PCA+SVM from 93.6% to 75.6% (the most robust).

  — [PMC5234804](https://pmc.ncbi.nlm.nih.gov/articles/PMC5234804/)
- The three most common PD noise types are white Gaussian noise (WGN), discrete spectral interference (DSI) and stochastic pulse-shaped interference. — [Energies (Oct 2021), DOAJ](https://doaj.org/article/936531d6124a4288bdbbe87bc3b83d4b). DSI is deterministic, often sinusoidal, and comes from power-line carrier and radio systems. — [Energies 14(23):7967, doi:10.3390/en14237967](https://0-doi-org.brum.beds.ac.uk/10.3390/en14237967)
- A synthetic protocol: PD pulses corrupted by WGN plus DSI at SNRs from about +9.78 dB to −10.34 dB. This is a denoising and localisation study, not a classifier. — [IJACSA 16(12), Paper 57 (2025)](https://thesai.org/Downloads/Volume16No12/Paper_57-Adaptive_Denoising_of_Partial_Discharge_Using_Absolute_Difference_Optimization.pdf)
- Covered-conductor (VSB-type) signals are affected by DSI, repetitive pulse interference, random pulse interference and ambient noise. — [Tehrani & Levorato, arXiv 2009.06825](https://arxiv.org/pdf/2009.06825)
- In UHF monitoring, external signals (corona, mobile phones, radar) can trigger false alarms. CIGRE case studies document external EMI recorded by UHF systems. — [CIGRE WG D1.37 guideline, §2.5](https://cigre.cz/dokumenty_komise/d1/WG%20D1.37_TB_Final.pdf)

### Inferences
- **Proposed protocol for reviewers (derived from the above):**
  1. **Grouping key = physical specimen** (test cell or defect instance) **and session** (assembly, day, voltage-ramp run). Never split windows, cycles or PRPD frames of one acquisition across train and test. For VSB, group by `id_measurement`, so the three phases of a line stay together.
  2. **Leave-one-specimen-out** or grouped k-fold, with ≥2–3 independent specimens per defect class. Raymond et al. 2017 used one joint per defect type, which perfectly confounds class with specimen. That is the textbook design flaw to avoid, and citing it is a good motivation.
  3. **Shift axes:**
     - cross-voltage: train near PDIV, test at higher or lower voltage (Applied Sciences 2023);
     - cross-sensor or coupling: UHF ↔ HFCT, where phase ranges differ (Kong et al.);
     - cross-resolution or instrument setting (Mas'ud et al.);
     - phase-offset robustness (specific to phase anchoring);
     - lab-to-field when available.
  4. **Noise:** train clean and test under (a) WGN at fixed SNRs, e.g. +10 to −10 dB, (b) DSI sinusoids, and (c) recorded field noise or pulse interference. Report degradation curves.
  5. **Few-shot specifics:** draw support and query sets from different specimens or sessions. Report mean ± CI over many episodes and seeds.
  6. Augment only within the training fold (AWA-CNN practice). Choose thresholds and hyperparameters on a grouped validation fold, not on the test set; Michau et al.'s Np selection is a cautionary example.
  7. Include a majority-class or degenerate baseline and a simple Gulski-feature + RF/SVM baseline. Matania et al. show that a leakage-free split can push models below the degenerate baseline.
- Report the "leaky" random-split number alongside the grouped number. This quantifies the inflation on PD data and fills a gap: no PD-specific quantification was found.

### Gaps
- No peer-reviewed PD paper that quantifies accuracy inflation from random vs. specimen-wise splits was found. This appears to be an open, citable gap.
- No standardised PD noise-robustness benchmark (fixed WGN + DSI + pulse SNR grid applied to a classifier) was found.
- The Applied Sciences 2023 attribution (voltage shift) was not verified in full text.

---

## Q5. Public PD datasets usable as auxiliary or external test sets

### Takeaway
Only one large, widely used public raw PD dataset with natural cycle structure was found: **VSB Power Line Fault Detection** (Kaggle, 2018). It contains one 50 Hz cycle per signal at about 40 MS/s, with 3 phases, labelled PD vs. no PD on covered overhead conductors. Defect-type-labelled sets are small, recent and often subscription-only:
- Mendeley PRPD images of generator PD (CC BY 4.0);
- IEEE DataPort HFCT raw time-series of controlled corona, surface and internal sources (UFCG Brazil, 125 MS/s, 35 ms windows, with a voltage channel; subscription);
- IEEE DataPort LV motor PD with a 50 Hz reference (subscription).

No public GIS UHF PRPD/PRPS defect-labelled dataset was found.

### Cited Findings
**VSB Power Line Fault Detection (Kaggle, VSB – Technical University of Ostrava, ENET Centre)**
- Each signal has 800,000 samples over 20 ms, one full 50 Hz cycle, and all three phases are measured simultaneously. Train has 8,712 signals (2,904 measurements); test has 20,337 signals (6,779 measurements). The meter measures up to 40 MHz (40 MHz sampling). Labels: 1 = PD, 0 = normal. — [Tehrani & Levorato, arXiv 2009.06825](https://arxiv.org/pdf/2009.06825)
- **Conflict on positive count.** Tehrani & Levorato state 525 of 8,712 positives and label this "less than 0.06%"; 525/8,712 is actually about 6.0%, so that percentage is an arithmetic slip. Michau et al. report 575 damaged of 8,712 and a test size of "20,037", which conflicts with 6,779 × 3 = 20,337. Verify on the Kaggle data page. — [arXiv 2009.06825](https://arxiv.org/pdf/2009.06825); [Michau et al., PMC8003486](https://pmc.ncbi.nlm.nih.gov/articles/PMC8003486/)
- The evaluation metric is the Matthews correlation coefficient (MCC). The test set has no public labels; it was scored on the Kaggle leaderboard. — [Michau et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8003486/); [arXiv 2009.06825](https://arxiv.org/pdf/2009.06825)
- The sensor measures "the voltage signal of the stray electrical field" along insulated (covered) overhead conductors. — [Dong, Sun, Wang, arXiv 1905.01588](https://arxiv.org/abs/1905.01588)
- The VSB smart-grid lab page links the data and asks users to cite related papers by Misák et al. This is from a search summary; the page redirected. — [VSB ENET Smart Grid Laboratory](https://cenet.vsb.cz/en/departments/smart-grid-laboratory/)

**Mendeley Data: generator PRPD images**
- Zorrilla Henao J.D., Tenorio Tamayo H.A., Jiménez Segura J.A., Diaz H., Paz A. (Universidad del Valle), "Images of resolved phase patterns of partial discharges in electric generators", v8, 7 Dec 2023, doi:10.17632/xz4xhrc4yr.8, **CC BY 4.0**. Classes are corona, surface ("superficial") and internal PD. The images are PNG PRPD patterns produced by the "MPD And MI 1.6.7.1" software and are intended for AI training and evaluation. A companion data article appears in *Data in Brief*. — [Mendeley Data](https://data.mendeley.com/datasets/xz4xhrc4yr/8); [ScienceDirect data article](https://www.sciencedirect.com/science/article/pii/S2352340923010223)

**IEEE DataPort: UFCG controlled-source HFCT databases**
- "Database of different sources of controlled partial discharges", Melo J.V.J.S., de Lira G.R.S., da Costa E.G., Vilar P.B., Leite Neto A.F., doi:10.21227/a3r5-zs61 (June 2025).
  - Sources: corona (two needle-plane configurations, 5 cm gap), surface (simulated rain on a suspension insulator), internal in solid (oil-filled cell with phenolic plates) and internal in liquid (needles 3 mm apart in oil).
  - Acquisition: HFCT about 1–80 MHz, 125 MSa/s, 35 ms per file. CH1 is PD and CH2 is applied voltage. Files are .mat and the archive is 8.39 GB.
  - The abstract says 250 acquisitions, but the per-source counts sum to 350 (inconsistent on the page). A UV camera confirmed onset.
  - **Subscription required**; no licence stated.

  — [IEEE DataPort](https://ieee-dataport.org/documents/database-different-sources-controlled-partial-discharges)
- "Database of multiple sources of controlled partial discharges", Melo et al. (UFCG, ISA Energia), doi:10.21227/0ttp-8582 (May 2026).
  - Sources: internal PD from a potential transformer with a manufacturing defect, corona (needle-plane) and simultaneous corona plus internal, intended for **source separation**.
  - Acquisition: HFCT 1–80 MHz, 125 MSa/s, 35 ms windows. CH1 is voltage and CH3 is HFCT. Files are .mat; the simultaneous folder is 1.04 GB.
  - **Subscription required**; no licence stated.

  — [IEEE DataPort](https://ieee-dataport.org/documents/database-multiple-sources-controlled-partial-discharges)

**Other IEEE DataPort and Zenodo sets**
- "Partial Discharge detection", Pinardi D., Toscani A., Immovilli F. (Univ. Parma / UNIMORE), doi:10.21227/57zj-4y32 (June 2024). LV induction-motor proof of concept. Each series has 9 measurements at 1.0–2.0 kV, with channels for PD and a **50 Hz reference**. Sampling is 20 MHz on the oscilloscope (about 0.1 s records) and 48 kHz on the DAQ (16 s). Files are .wav in a .7z archive (59 MB). **Subscription required**. — [IEEE DataPort](https://ieee-dataport.org/documents/partial-discharge-detection)
- PD-Loc (doi:10.21227/yy7g-2p79): a 32-sensor acoustic array (.wav) with chirps, WGN and PD signals, intended for localisation, not pattern recognition. — [IEEE DataPort](https://ieee-dataport.org/open-access/partial-discharge-localisation-pd-loc-dataset)
- A synthetic PD dataset for underground cables (simulated PD trends for reliability and outage-cost analysis), not suitable for pattern recognition. — [Zenodo 21240484](https://zenodo.org/records/21240484)

### Inferences
- **VSB** fits a cycle-anchored design well: each record is exactly one cycle, so phase windows correspond to sample ranges. Caveats:
  - Labels are binary (PD vs. no PD), not defect types.
  - Whether records start at a fixed phase reference is not stated in Michau et al. Tehrani et al. high-pass filter away the 50 Hz component, so the phase must be estimated from the low-frequency component.
  - Group by `id_measurement`.
  
  Use it as a **pretraining or auxiliary-domain set and an external detection benchmark**, not for defect-type few-shot evaluation.
- The **UFCG DataPort** sets include an applied-voltage channel, which allows true phase referencing. The 35 ms windows hold about 2.1 cycles at 60 Hz; the 60 Hz Brazilian grid frequency is general knowledge, not stated on the pages. They are the best candidate external test set for defect-type classification (corona / surface / internal), if subscription access and redistribution terms allow. Each source appears to come from one or few test arrangements, so the specimen grouping must be checked.
- The **Mendeley** set is image-only (PRPD PNG), so it suits image baselines or PRPD-feature baselines, not a raw time-series TSFM interface.

### Gaps
- The Kaggle VSB licence or competition data-use rules, the file sizes and the exact positive count (525 vs 575) were not verified, because the Kaggle page could not be fetched.
- The image counts per class and the measurement provenance (number of generators or sessions) of the Mendeley dataset were not obtained.
- No public GIS UHF PRPS/PRPD dataset with defect labels was found (Mendeley, Zenodo, Kaggle and DataPort were searched by web search only, not by in-portal search).
- Licences for the IEEE DataPort "standard" datasets are not stated on their pages. Subscription access may restrict redistribution and use as a shared benchmark.

---

## Q6. (Second paper) Event-anchored representations for HV circuit-breaker mechanical signals and public breaker datasets

### Takeaway
The trip or close coil current profile has a physically defined event structure:
- plunger movement → current rise;
- plunger strikes the latch → change in slope;
- plunger hits the buffer → a deep "V";
- current rises to its rated plateau;
- the auxiliary contact opens → current falls to zero.

Its characteristic times (latch, buffer, main contact, auxiliary contact and end times) are the established features. They are the breaker analogue of the PD phase anchor: patches or positions anchored to these mechanical events rather than to clock time. Vibration-based monitoring uses similar event times derived from the coil current. The "t1–t5" notation itself was not found in a retrievable English source. Public data is minimal: one Zenodo set with 83 trip-coil current measurements (CC BY 4.0) and one subscription DataPort MV vacuum-breaker set with vibration and sound.

### Cited Findings
- Stephen B., Strachan S.M., McArthur S.D.J., McDonald J.R., Hamilton K., "Design of a trip current monitoring system for circuit breaker condition assessment", *IET Generation, Transmission & Distribution* (2007), Strathprints accepted version, text-extracted; DOI not retrieved. The physics of the profile:
  - The trip signal energises the coil and the plunger moves. The induced current "rises until gravity abates both the plunger velocity and the current induced".
  - The plunger then connects with the triggering latch, causing "a step change in the reduction rate of the current".
  - After unlatching, the plunger hits a buffer, producing "a deep 'V' shape in the current profile".
  - The current then rises to a peak set by the coil rating until the supply is disconnected when the mechanism opens the contacts, after which it falls to zero.
  - Extracted features (Kelman Profile handheld): **Latch time, Buffer time, Main Contact time (MCON), Auxiliary Contact time (ACON), End time**.
  - Diagnostic rules: a latch-time delay suggests obstructed or misaligned plunger travel; a buffer-time delay suggests absent latch lubrication; an ACON delay suggests defective contacts, and prolonged coil energisation risks burnout.

  — [Strathprints PDF](https://strathprints.strath.ac.uk/11747/1/Stephen_etal_IET_GTD_2007_Design_of_trip_current_monitoring_system_for_circuit.pdf)
- Strachan et al. (IEEE TPWRD, 2007) used data mining of trip-coil current signatures with thresholds for features "representing each stage of a circuit breaker's operation". Kezunovic et al., "Automated Monitoring and Analysis of Circuit Breaker Operation", *IEEE TPWRD* (July 2005), is cited for automated analysis. — [Strathprints 11547](https://strathprints.strath.ac.uk/11547/); reference list in [Stephen et al.](https://strathprints.strath.ac.uk/11747/1/Stephen_etal_IET_GTD_2007_Design_of_trip_current_monitoring_system_for_circuit.pdf)
- A trip-diagnostic patent describes a coil-current profile with three transition points defining four time segments, from the initial rise above nominal to the return to nominal. — [USPTO 9864008](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9864008)
- A vibration-based monitoring patent labels closing events tc0, tc1, tc21, tc22 and opening events to0, to1, to21, to22. tc0 is known, tc1 is a rough motion start, and tc21/tc22 are derived from the current curve. — [USPTO 12007442](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12007442)
- CIGRE A3 (2026), "Data-driven condition monitoring of AC-supplied trip coils of circuit breakers": profile analysis identifies mechanical events (e.g. "anchor hitting end position, power cut-off") that are prominent in DC measurements but disguised in AC sinusoidal waveforms. — [e-cigre A3-11992-2026](https://www.e-cigre.org/publications/detail/a3-11992-2026-data-driven-condition-monitoring-of-ac-supplied-trip-coils-of-circuit-breakers.html)
- Related methods:
  - Curvature-scale-space corner points on opening and closing coil current. — [Zhejiang Electric Power, June 2024 (DOAJ)](https://doaj.org/article/7c3bec3e0cdb4d74bb2c6d5ab51c4df2)
  - Improved DTW (Sakoe-Chiba window) on coil current. — [IFMCA 2016, Atlantis Press](https://www.atlantis-press.com/article/25871834.pdf)
  - Trip/close coil current used to detect the mode and cause of incipient failures. — [Sharif Univ. repository](https://repository.sharif.edu/resource/408224/-/&from=search&&query=khalouei--e&field=authorOther&count=20&execute=true)
- Vibration-based HVCB diagnosis uses lab-introduced faults (spring failure, clearance error, cam eccentricity, over-dead-point), reporting up to 94% accuracy, with no data release mentioned. — [CSEE Electric Power journal](https://epjournal.csee.org.cn/article/id/75A95A55123137CFE054D89D67F5A4E2). An unsupervised XAI study on an HVCB uses an experimental dataset with healthy and artificially introduced faults; public availability was not verified. — [arXiv 2507.19168](https://arxiv.org/html/2507.19168v1)

**Public datasets**
- Jordon G. (NTNU), "Dataset – The impact of high-voltage circuit breaker condition on power system reliability indices", Zenodo, 22 Aug 2024, doi:10.5281/zenodo.13358840, **CC BY 4.0**. Contains **83 trip-coil current measurements** plus an outage and life-history database of 464 HVCBs in the Icelandic grid. Files: "Condition data.xls" (53.0 MB) and "Survival data.xls" (54.8 kB). Sampling rate and labels are not stated. — [Zenodo](https://zenodo.org/records/13358840)
- Guo N. (Georgia Tech), IEEE DataPort doi:10.21227/3spy-de39: current, voltage, vibration and sound during load switching of an EATON MV vacuum circuit breaker, for non-invasive arc-duration measurement. It is a "standard" (subscription) dataset, and coil current is not listed. — [IEEE DataPort](https://ieee-dataport.org/documents/non-invasive-circuit-breaker-arc-duration-measurement-method-improved-robustness-based)
- An ABB low-voltage breaker anomaly thesis uses angular travel curves from endurance tests; the data appear proprietary. — [Politecnico di Milano thesis](https://www.politesi.polimi.it/retrieve/4fb03b45-6e73-47b4-bdfa-605fb45d55b2/thesis_final_report.pdf)

### Inferences
- An "event-anchored" interface for breakers would segment or patch the coil current, travel and vibration signals relative to detected mechanical events (latch, buffer, contact separation, auxiliary-contact cut-off), not by clock time. It could encode "time since event k" as a positional signal, analogous to phase-PE. The Stephen et al. features give the physically meaningful anchors and the expected fault-to-anchor mapping.
- Leakage grouping for breakers should use the **breaker unit and the fault-installation session**. Lab fault-injection studies typically use one breaker with sequential faults, the same confound as in the PD specimen case.
- Public data are too small for a benchmark paper (83 coil currents). A second paper would likely need in-house data, or should frame the Zenodo set as an external sanity check.

### Gaps
- No retrievable English source defining "t1–t5" for coil current was found. This notation is common in Chinese literature, and the source for it still needs to be identified.
- No public dataset combining coil current, travel curve and vibration with fault labels was found.
- The DOIs of Stephen et al. (IET GTD 2007), Strachan et al. (IEEE TPWRD 2007) and Kezunovic et al. (IEEE TPWRD 2005) were not retrieved.
- The Zenodo HVCB dataset's sampling rate, labelling and per-breaker structure were not stated on the record page.
