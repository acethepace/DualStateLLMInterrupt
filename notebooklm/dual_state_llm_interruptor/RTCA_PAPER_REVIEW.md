# Comprehensive Review & Audit: RTCA Workshop Submission (`rtca.tex`)

**Target Paper**: *Dual-State KV Forking: Low-Overhead Semantic Interruption for Full-Duplex LLM Agents*  
**Target Venue**: NeurIPS 2026 Workshop on Real-Time Conversational Agents (RTCA)  
**Track**: 4-Page Double-Blind Workshop Track (Main text $\le 4$ pages, plus unlimited references and appendix)  
**Date**: September 2026  
**Review Synthesis**: Consolidated from Academic Peer Review (NeurIPS RTCA Specialist), ML Systems Architecture Specialist, and Citation & Literature Auditor.

---

## 1. Overall Assessment & Meta-Review Recommendation

| Metric | Score | Evaluation Summary |
| :--- | :---: | :--- |
| **Overall Recommendation** | **Accept (Oral / Spotlight Contender)** | **7.5 / 10** — High conceptual novelty, outstanding systems impact, and physical hardware validation. |
| **Scientific Contribution** | **4.2 / 5.0** | Bridges the critical divide between passive input streaming and active semantic floor control for LLM agents. |
| **Technical & Systems Rigor** | **3.8 / 5.0** | Hardware-measured latencies on RTX 4070 Ti are compelling; requires correction of copy-paste data anomalies in Tables 3 & 4. |
| **Workshop Fit & Page Limit** | **5.0 / 5.0** | Main text adheres strictly to the 4-page workshop limit (Sections 1–5 end on Page 4; References begin on Page 5). |
| **Bibliographic Integrity** | **2.5 / 5.0** | **Action required**: Contains one hallucinated venue/page entry (`rodriguez2021quizbowl`), one author list conflation (`defossez2024moshi`), one natbib key mismatch (`lin2022duplex`), and multiple preprints that must be updated to their formal peer-reviewed proceedings. |

---

## 2. Academic Peer Review & Scientific Analysis

### 2.1 Strengths
1. **Clear Conceptual Framing**: The paper clearly defines the problem: input streaming warms the cache for post-utterance TTFT but leaves the agent passive. Dual-State provides the missing active semantic floor control.
2. **Strict Adherence to Workshop Constraints**: The main paper is exactly 4 pages, dense with high-value technical content, an informative TikZ architectural diagram (Figure 1), and two core result tables (Tables 1 & 2).
3. **Physical Hardware Validation**: All benchmarks are run on real hardware (NVIDIA RTX 4070 Ti, 4-bit quantization) with proper CUDA event timing. Shallow-cloning / pointer-aliasing overhead is physically measured at **0.01–0.06 ms**.
4. **Comprehensive Benchmark Coverage**: Evaluated across static human transcripts (FLEXI, 400 cases), dynamic multi-agent interaction (Full-Duplex-Bench, 400 cases), and end-to-end multi-attempt floor control on adversarial progressive trivia (HANDRAISER, 200 scenarios).
5. **Airtight Token Accounting**: Demonstrates an empirical **78.6% net token reduction** and curtails token overage from $+1424\%$ to $+227\%$, explaining context re-prefill elimination as the direct mechanical driver.

### 2.2 Critical Data Anomalies & Discrepancies Requiring Immediate Correction

#### Anomaly A: Copy-Paste Error in Table 4 (Llama 3.1 8B Instruct, Appendix)
* **The Bug**: In Table 4 (line 353), under **Independent (Logit Gating)** on **FLEXI**, the reported accuracy is **91.00%**, but the printed confusion matrix is:
  $$\text{TP } 198 / \text{TN } 197 / \text{FP } 3 / \text{FN } 2 \quad (198 + 197 = 395 / 400 = \mathbf{98.75\%})$$
  Similarly, for **FDB** (line 354), accuracy is reported as **95.00%**, but the printed confusion matrix is:
  $$\text{TP } 199 / \text{TN } 200 / \text{FP } 0 / \text{FN } 1 \quad (199 + 200 = 399 / 400 = \mathbf{99.75\%})$$
* **Root Cause**: These two confusion matrices were accidentally copied verbatim from Qwen 3 4B (Table 5, lines 372–373).
* **Official Data from Log Files** (`results/unsloth_Meta-Llama-3.1-8B-Instruct/`):
  * **FLEXI Logit Gating**: Accuracy = **91.00%**, $\mathbf{TP = 166, TN = 198, FP = 2, FN = 34}$ ($166 + 198 = 364 / 400 = 91\%$).
  * **FDB Logit Gating**: Accuracy = **95.00%**, $\mathbf{TP = 180, TN = 200, FP = 0, FN = 20}$ ($180 + 200 = 380 / 400 = 95\%$).

#### Anomaly B: Blank Confusion Matrices in Table 3 (Gemma 4 12B Instruct, Appendix)
* **The Bug**: In Table 3 (lines 334–335), under **Independent (Logit Gating)**, the `\scriptsize` sub-blocks were left completely empty (`\makecell{98.75\% \\ \scriptsize }` and `\makecell{90.75\% \\ \scriptsize}`).
* **Official Data from Log Files** (`results/unsloth_gemma-4-12b-it/`):
  * **FLEXI Logit Gating**: Accuracy = **98.75%**, $\mathbf{TP = 200, TN = 195, FP = 5, FN = 0}$.
  * **FDB Logit Gating**: Accuracy = **90.75%**, $\mathbf{TP = 200, TN = 163, FP = 37, FN = 0}$.

#### Anomaly C: Table 3 Width Formatting (`Overfull \hbox`)
* In the LaTeX compilation log, Table 3 generates `Overfull \hbox (29.62155pt too wide)` because of fixed column definitions (`p{4.1cm}cccc`). Wrapping in `\resizebox{\columnwidth}{!}{...}` or adjusting `\tabcolsep` will resolve this margin protrusion.

#### Anomaly D: Model Naming Consistency (Gemma 4 12B vs. Gemma 3 Technical Report)
* Line 124 states: *"Gemma 4 12B~\citep{gemmateam2025gemma3}"*. The cited paper is the Gemma 3 technical report (arXiv:2503.19786, March 2025). The Gemma 4 technical report was released in 2026 (arXiv:2607.02770). The citation should be updated to `gemmateam2026gemma4` to match the model evaluated.

#### Anomaly E: Contextual Clarifications in Text
1. **Qwen Consensus Time in Table 2**: Independent Decider has 3.0786 s vs. Dual-State 4.0542 s. The text should clarify that earlier non-fatal buzzes with continuation allow the agent to hear more clues in the multi-attempt loop, achieving higher resolution accuracy (+1.0%) and game score at the expense of longer dialogue duration.
2. **Llama Time-to-Halt on FDB**: On Llama FDB, Independent Logit Gating is 117.67 ms and Dual-State is 115.73 ms (a 1.6% cut), because both 1-token decodes operate at the physical hardware memory bandwidth floor on an 8B model at $L \approx 200$. Clarify that the 28%–32% latency reduction over logit gating applies to Qwen and Gemma, and long contexts ($L=1024$).

---

## 3. ML Systems Architecture Review

### 3.1 Pointer Aliasing vs. Shallow Tensor Cloning (0.01–0.06 ms)
* **Clarification**: In PyTorch prototype execution (`run_noop_ttft_experiment.py`), an eager deep clone of the small $L \le 1024$ KV cache took $0.01\text{--}0.06\,\text{ms}$ simply because the total tensor byte volume is small.
* **Production Serving Semantics**: In real production serving engines (vLLM, SGLang), this operation is executed as **true pointer aliasing / block table reference sharing** with zero device memory copying ($0.00\,\text{ms}$ device transfer). The paper should explicitly disambiguate the PyTorch prototype from the target engine abstraction.

### 3.2 Serving Engine Integration & Copy-on-Write (CoW)
* **Partial-Block CoW Dynamic**: When speaker tokens arrive in chunks of $k = 5$ and engine page size is 16 or 32, the assessment boundary lands inside a partially occupied tail block.
* **Systems Precision**: Preceding saturated blocks are zero-copy pointer-aliased, while Copy-on-Write is applied strictly to the partially filled tail page ($< B_{\text{size}}$ tokens). Highlighting this sub-block CoW mechanism demonstrates deep systems competence to MLSys/OSDI reviewers.

### 3.3 Memory Footprint Framing (49.5% Reduction)
* Against a **parallel stateful decider** (which maintains a second active streaming session), Dual-State cuts persistent KV memory by 49.5%.
* Against a **stateless decider** (which allocates context dynamically on each chunk), Dual-State eliminates the transient $O(L)$ memory spike and allocation churn during each assessment forward pass.

### 3.4 Rate Inversion & Adaptive Chunk Coalescing
* When Assessor forward latency exceeds the physical chunk arrival interval ($T_{\text{eval}} > k \cdot T_{\text{token}}$, e.g. Gemma 180 ms forward pass vs. 75 ms arrival window), the serving scheduler naturally implements **adaptive chunk coalescing**: tokens arriving while an assessor pass is active are ingested together at the subsequent boundary, preventing pipeline backlog.

---

## 4. Thorough Reference Audit (All 22 References)

Every single citation in `neurips_template/rtca.tex` (lines 198–315) was audited against official publisher metadata and arXiv registries.

| # | BibTeX Key | Current Paper Details in `rtca.tex` | True Publication Details & Required Corrections | Status |
|---|:---|:---|:---|:---:|
| 1 | `agrawal2023sarathi` | arXiv preprint arXiv:2308.16369, 2023 | **Published in USENIX OSDI 2024** as *"Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"*, pages 1127–1143. Update from preprint to OSDI '24. | **Update to OSDI '24** |
| 2 | `defossez2024moshi` | Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Eugene Kharitonov, Jade Copet, Yossi Adi, and Edouard Grave. arXiv:2410.00037, 2024 | **Critical Author Conflation Error**: Kharitonov, Copet, and Adi are from Meta/dGSLM, not Moshi. **True Authors**: Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. | **Critical Fix** |
| 3 | `gemmateam2025gemma3` | Gemma Team. Gemma 3 technical report. arXiv:2503.19786, 2025 | Author must be braced as `{{Gemma Team}}` to prevent natbib author parsing errors. Update to Gemma 4 technical report (`gemmateam2026gemma4`, arXiv:2607.02770) to match the 12B model evaluated. | **Update & Protect** |
| 4 | `gim2024promptcache` | In Gim et al. In MLSys, 2024 | Published in *Proceedings of Machine Learning and Systems (MLSys 2024)*, Vol. 6, pages 341–353. Add volume and page numbers. | **Update Metadata** |
| 5 | `grattafiori2024llama3` | Aaron Grattafiori et al. arXiv:2407.21783, 2024 | Add proper natbib two-part author structure `[Grattafiori et~al.(2024)Grattafiori et~al.]` to prevent natbib crash. Lead author is Abhimanyu Dubey. | **Fix Natbib Label** |
| 6 | `kwon2023vllm` | Woosuk Kwon et al. In SOSP, 2023 | Published in *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23)*, pages 611–626. Add ACM and pages. | **Add Page Numbers** |
| 7 | `leviathan2023speculative`| Yaniv Leviathan et al. In ICML, 2023 | Published in *Proceedings of the 40th International Conference on Machine Learning (ICML '23)*, PMLR 202:19274–19286. Add PMLR volume/pages. | **Add PMLR Metadata** |
| 8 | `li2023camel` | Guohao Li, Hasan Hammoud et al. In NeurIPS, 2023 | Published in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, pages 51991–52008. Second author full name: Hasan Abed Al Kader Hammoud. | **Add Volume & Pages** |
| 9 | `lin2025fdbench` | Guan-Ting Lin et al. In Proc. IEEE ASRU, 2025. arXiv:2503.04721 | Formally accepted at *IEEE Automatic Speech Recognition and Understanding Workshop (ASRU 2025)*. Capitalize title: `{{Full-Duplex-Bench}}`. | **Fix Title Braces** |
| 10 | `lin2022duplex` | `[Lin et~al.(2022)Lin, Wang, Li, Zhang, Chen, and Si]` Ting-En Lin et al. In KDD, 2022 | **Critical Natbib Bug**: Optional label lists phantom authors (`Wang`, `Zhang`, `Chen`) that do not match printed authors (`Lin, Wu, Huang, Si, Sun, and Li`). Published in *KDD '22*, pages 3438–3447. | **Critical Fix** |
| 11 | `nguyen2023dgslm` | Tu Anh Nguyen et al. TACL, 11:250–266, 2023 | Verified correct in *TACL* Vol 11, pages 250–266. Add French diacritic on `Beno{\^\i}t Sagot`. | **Minor Accent Fix** |
| 12 | `rodriguez2021quizbowl` | Pedro Rodriguez et al. *Computational Linguistics*, 47(3):537–593, 2021 | **CRITICAL HALLUCINATION**: This paper was **never** published in *Computational Linguistics*; those exact page numbers belong to Nguyen et al. (2016). Rodriguez et al. is an **arXiv preprint: arXiv:1904.04792 (2019)**. Must be fixed immediately. | **Critical Fix** |
| 13 | `sacks1974turntaking` | Harvey Sacks et al. Language, 50(4):696–735, 1974 | Verified 100% correct. Landmark turn-taking paper. | **Verified Correct** |
| 14 | `skantze2021review` | Gabriel Skantze. Computer Speech & Language, 67:101178, 2021 | Verified 100% correct. Review paper on turn-taking. | **Verified Correct** |
| 15 | `wallace2019trick` | Eric Wallace et al. TACL, 7:387–401, 2019 | Verified 100% correct. Adversarial Quizbowl benchmark paper. | **Verified Correct** |
| 16 | `wu2023autogen` | Qingyun Wu et al. arXiv:2308.08155, 2023 | arXiv preprint (August 2023); Best Paper Award at ICLR 2024 Workshop on LLM Agents. Capitalize `{Auto{G}en}` and `{LLM}`. | **Fix Title Braces** |
| 17 | `xiao2023streamingllm` | Guangxuan Xiao et al. In ICLR, 2024 | Published in *Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)*. BibTeX key has 2023, but year is 2024. | **Verified Correct** |
| 18 | `yang2025qwen3` | An Yang et al. Qwen3 technical report. arXiv:2505.09388, 2025 | arXiv:2505.09388 (May 2025). Add proper natbib two-part author structure `[Yang et~al.(2025)Yang et~al.]`. | **Fix Natbib Label** |
| 19 | `yngve1970` | Victor H. Yngve. In Chicago Linguistic Society, pages 567–578, 1970 | Verified 100% correct. Foundational paper that introduced "back channel". | **Verified Correct** |
| 20 | `yu2022orca` | Gyeong-In Yu et al. In OSDI, 2022 | Published in *Proceedings of the 16th USENIX OSDI '22*, pages 521–538. Add page numbers. | **Add Page Numbers** |
| 21 | `zhang2024duplexmodels`| Xinrong Zhang et al. In EMNLP, 2024 | Published in *Proceedings of EMNLP 2024*, pages 11543–11557. Add formal proceedings title and page numbers. | **Add Page Numbers** |
| 22 | `zheng2023sglang` | Lianmin Zheng et al. In NeurIPS, 2024 | Published in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. Verified 100% correct. | **Verified Correct** |

---

## 5. Curated List of High-Impact NEWER References (Late 2024–2026)

The following 8 papers represent top-tier work published between late 2024 and 2026 directly relevant to the paper's themes. **Presented for user review prior to inclusion in `rtca.tex`**:

### Category A: Full-Duplex Spoken Dialogue & Omni Models
1. **LLaMA-Omni (ICLR 2025)**
   - *Citation*: Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. *"LLaMA-Omni: Seamless Speech Interaction with Large Language Models"*. In *International Conference on Learning Representations (ICLR)*, 2025. arXiv:2409.06666.
   - *Rationale*: Open-source speech-to-speech architecture achieving ultra-low 236 ms latency. Directly complements the related work discussion on spoken full-duplex agents in Section 2 (alongside Moshi and dGSLM).
2. **Mini-Omni2 (2024/2025)**
   - *Citation*: Zhifei Xie and Changqiao Wu. *"Mini-Omni2: Towards Open-Source GPT-4o with Vision, Speech and Duplex Capabilities"*. *arXiv preprint arXiv:2410.11190*, 2024.
   - *Rationale*: Open-source duplex omni-model capable of simultaneous listening and speaking with command-based interruption mechanisms.
3. **Full-Duplex-Bench-v2 (2025)**
   - *Citation*: Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Alexander H. Liu, and Hung-yi Lee. *"Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner"*. *arXiv preprint arXiv:2507.03157*, 2025.
   - *Rationale*: The multi-turn evolution of Full-Duplex-Bench (which is already a flagship benchmark in `rtca.tex`), using an automated AI examiner to evaluate staged conversational goals in real time.

### Category B: Real-Time Interruption Detection & Semantic Barge-In
4. **SID-Bench & Average Penalty Time (2026)**
   - *Citation*: Kangxiang Xia, Bingshen Mu, Xian Shi, Jin Xu, and Lei Xie. *"Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model"*. *arXiv preprint arXiv:2603.24144*, 2026.
   - *Rationale*: Establishes **SID-Bench** and the **Average Penalty Time (APT)** metric to formalize the mathematical trade-off between interruption speed and false alarms on backchannels. Citing this provides direct theoretical anchoring for Dual-State's Time-to-Halt evaluation.

### Category C: KV-Cache Sharing, Prefix Caching & Serving Engines
5. **Mooncake (USENIX FAST 2025 — Best Paper Award)**
   - *Citation*: Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. *"Mooncake: Trading More Storage for Less Computation — A KVCache-Centric Architecture for Serving LLM Chatbot"*. In *Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25)*, pages 1–16, 2025.
   - *Rationale*: Won the Best Paper Award at FAST 2025. Powers Kimi's production architecture by decoupling prefill and decode around a disaggregated KV cache pool. Provides direct validation for Dual-State's trunk-leaf memory management in Appendix D.
6. **CacheBlend (EuroSys 2025 — Best Paper Award)**
   - *Citation*: Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. *"CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion"*. In *Proceedings of the 20th European Conference on Computer Systems (EuroSys '25)*, 2025.
   - *Rationale*: Won the Best Paper Award at EuroSys 2025. Reuses precomputed KV caches for non-contiguous text chunks, recomputing only 10–15% of cross-document attention to achieve a 2.2–3.3x TTFT cut. Strongly supports Dual-State's zero-copy KV reuse argument.
7. **DistServe (USENIX OSDI 2024)**
   - *Citation*: Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. *"DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving"*. In *Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI '24)*, pages 193–209, 2024.
   - *Rationale*: Landmark systems paper establishing prefill-decode disaggregation to eliminate interference between compute-bound prefills and memory-bound decodes.
8. **CacheGen (ACM SIGCOMM 2024)**
   - *Citation*: Yuhan Liu, Hanchen Li, Kuntai Du, Jiayi Yao, Amy Huang, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, and Junchen Jiang. *"CacheGen: Fast Context Loading for Language Model Serving"*. In *Proceedings of the ACM SIGCOMM 2024 Conference (SIGCOMM '24)*, pages 796–814, 2024.
   - *Rationale*: Demonstrates streaming KV-cache encoding/compression for rapid transmission and streaming context ingestion.

---

## 6. Actionable Implementation Plan

1. **Clean up Table 4 (Llama 3.1 8B)**: Replace the copy-pasted Qwen confusion matrices with the genuine Llama logit gating values (`TP 166 / TN 198 / FP 2 / FN 34` on FLEXI, `TP 180 / TN 200 / FP 0 / FN 20` on FDB).
2. **Populate Table 3 (Gemma 4 12B)**: Fill in the blank confusion matrix cells for Independent Logit Gating (`TP 200 / TN 195 / FP 5 / FN 0` on FLEXI, `TP 200 / TN 163 / FP 37 / FN 0` on FDB), and wrap with `\resizebox` to fix the 29.6pt `Overfull \hbox`.
3. **Apply the Corrected Bibliography**: Update the 22 references in `rtca.tex` using the cleaned BibTeX entries to fix the Rodriguez hallucination, the Moshi author conflation, the Lin natbib key error, and the Sarathi/SGLang/StreamingLLM published proceedings.
4. **Clarify Systems Terminology in Text**: Explicitly clarify pointer aliasing vs. prototype cloning in Section 3 and note tail-block CoW in Section 5 / Appendix F.
5. **Awaiting User Selection on Newer References**: User to select which of the 8 candidate references (e.g. Mooncake FAST '25, CacheBlend EuroSys '25, LLaMA-Omni ICLR '25, SID-Bench '26) to incorporate into `rtca.tex`.
