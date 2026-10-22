# Research basis

Every problem statement in the app is tied to a source. `aeronexus/research.py` checks each claim's numbers and wording
against a short excerpt of the source, and `tests/test_research_claims.py` fails CI if any claim drifts.
Industry (non-peer-reviewed) figures are labelled as such and should be read as directional.

### CLM-DRAW-1 · quote

**Claim:** On real construction drawings, state-of-the-art multimodal LLMs show a substantial gap versus experts, largest at higher reasoning depths.  
**Excerpt:** "substantial gap between model and expert performance, particularly at higher reasoning depths"  
**Source:** [DrawingVQA: Multi-Depth Visual-Textual Reasoning on Construction Drawings (arXiv 2607.15418)](https://www.alphaxiv.org/abs/2607.15418) — preprint, 2026  
**Used for:** Why drawings are not sent straight to a general VLM: rules/NER first, VLM fallback, human review on low confidence.  
**Check:** passed

### CLM-DRAW-2 · quote

**Claim:** A domain-specialized model trained on MechVQA beats the strongest closed-source baseline by 7.57 points.  
**Excerpt:** "MechVL outperforms the strongest closed-source baseline by 7.57 percentage points"  
**Source:** [MechVQA: Benchmarking and Enhancing Multimodal LLMs on Mechanical Drawing Understanding (arXiv 2605.30794)](https://www.emergentmind.com/papers/2605.30794) — peer-reviewed (ICML 2026), 2026  
**Used for:** Motivates the LoRA fine-tune in notebooks/02_vlm_lora.ipynb.  
**Check:** passed

### CLM-DRAW-3 · datasets

**Claim:** MechVQA contains 3.3K drawings with 21K question-answer pairs and is published under Apache-2.0.  
**Excerpt:** "3.3k high-density pictures with 21K question-answer pairs; License: apache-2.0"  
**Source:** [MechVQA dataset card (Hugging Face XiaofengAlg/MechVQA)](https://huggingface.co/datasets/XiaofengAlg/MechVQA) — dataset card, 2026  
**Used for:** Primary training/eval set for the drawing VLM.  
**Check:** passed

### CLM-DRAW-4 · evals

**Claim:** Token-overlap metrics under-report VLM capability on technical descriptions by 2-6x relative to semantic similarity.  
**Excerpt:** "token-overlap metrics under-report model capability on technical descriptions by 2-6x"  
**Source:** [Enginuity: Vision-Language Understanding of Engineering Diagrams (arXiv 2606.03410)](https://www.emergentmind.com/papers/2606.03410) — preprint, 2026  
**Used for:** Free-text fields are scored with a calibrated judge, not BLEU/ROUGE.  
**Check:** passed

### CLM-JUDGE-1 · evals

**Claim:** Across 21 judges, raw exact-match agreement overstates chance-corrected kappa by 33-41 points on MT-Bench.  
**Excerpt:** "21 judges ... kappa deflation ... is universal (33--41 pp on MT-Bench)"  
**Source:** [Reliability without Validity: Evaluation of LLM-as-a-Judge Models (arXiv 2606.19544)](https://arxiv.org/abs/2606.19544) — preprint, 2026  
**Used for:** Judge reliability is reported as Cohen's kappa with bootstrap CI; the kappa-deflation number is shown in the UI.  
**Check:** passed

### CLM-JUDGE-2 · evals

**Claim:** High agreement among LLM judges is not evidence of agreement with humans (inter-LLM r about 0.35 vs LLM-human 0.27-0.32).  
**Excerpt:** "Inter-LLM agreement exceeds LLM--human agreement (r 0.35 versus 0.27--0.32)"  
**Source:** [The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment (arXiv 2606.03043)](https://www.emergentmind.com/papers/2606.03043) — preprint, 2026  
**Used for:** The judge is calibrated against human labels, never against other judges.  
**Check:** passed

### CLM-JUDGE-3 · evals

**Claim:** Judge-based estimates should be corrected using the judge's sensitivity and specificity measured on a human-labelled calibration set.  
**Excerpt:** "Using this calibration dataset, we estimate the judge's sensitivity ... specificity"  
**Source:** [How to Correctly Report LLM-as-a-Judge Evaluations (ICML 2026 slides)](https://icml.cc/media/icml-2026/Slides/63607.pdf) — peer-reviewed (ICML 2026), 2026  
**Used for:** metrics.corrected_pass_rate (Rogan-Gladen) in the judge harness.  
**Check:** passed

### CLM-RAG-1 · rag

**Claim:** In retrieval-augmented QA, only 27-41% of citation pairs were supported at the sentence level.  
**Excerpt:** "Only 27-41% of citation pairs are supported at the sentence level"  
**Source:** [VERICITE: Sentence-Level Citation Faithfulness in Retrieval-Augmented Medical QA](https://aclanthology.org/people/yixian-ma/) — peer-reviewed (BioNLP 2026), 2026  
**Used for:** Every generated sentence is checked against the passage it cites (grounding.verify).  
**Check:** passed

### CLM-RAG-2 · rag

**Claim:** Removing the NLI verification step barely changed answer correctness but severely degraded citation quality.  
**Excerpt:** "Removing the NLI verification step ... caused severe deterioration in citation quality"  
**Source:** [VeriCite: Reliable Citations in RAG via Rigorous Verification (arXiv 2510.11394)](https://arxiv.org/html/2510.11394v1) — preprint, 2025  
**Used for:** Optional DeBERTa NLI verifier (AERONEXUS_USE_NLI=1); lexical verifier is the fast default.  
**Check:** passed

### CLM-ECM-1 · change

**Claim:** Engineer-to-order companies still report cost overruns caused by engineering changes.  
**Excerpt:** "ETO companies still report cost overruns owing to ECs"  
**Source:** [Factors affecting EC implementation performance in ETO: Norwegian shipbuilding case studies](https://www.sintef.no/en/publications/publication/1845083) — peer-reviewed, 2020  
**Used for:** Problem statement for the change-impact agent.  
**Check:** passed

### CLM-ECM-2 · change

**Claim:** Literature and case studies show ETO firms struggle with change impact analysis, data management and coordination.  
**Excerpt:** "challenges with ECM, particularly with regard to impact analysis, data management, coordination"  
**Source:** [Engineering Change Management in the Engineer-to-Order Production Environment (NTNU)](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2978358) — thesis, 2021  
**Used for:** Why the agent links drawing diffs to ERP orders and CRM accounts automatically.  
**Check:** passed

### CLM-MRO-1 · mro

**Claim:** FAA compliance letters to major carriers show common errors in completing maintenance entries and handling documents.  
**Excerpt:** "common errors with the completion of maintenance entries, handling of maintenance documents"  
**Source:** [Maintenance Errors Common to Major U.S. Certificated Air Carriers (Oklahoma State)](https://shareok.org/handle/11244/330178) — peer-reviewed, 2021  
**Used for:** Problem statement for the AD compliance checker.  
**Check:** passed

### CLM-MRO-2 · mro

**Claim:** Maintenance logbooks are full of domain terms, abbreviations and non-standard spelling; MaintNet's aviation set has 6,169 entries.  
**Excerpt:** "abbreviations, and non-standard spelling ... Aviation Maintenance 6,169"  
**Source:** [MaintNet: Open-Source Library for Predictive Maintenance Language Resources](https://aclanthology.org/2020.coling-demos.2) — peer-reviewed (COLING 2020), 2020  
**Used for:** Abbreviation normalisation + fuzzy NER; MaintNet used for NER training.  
**Check:** passed

### CLM-AP-1 · ap

**Claim:** About 25% of invoices were flagged for exceptions in an industry AP benchmark.  
**Excerpt:** "About 25% of invoices were flagged for exceptions last year"  
**Source:** [Ardent Partners AP Metrics that Matter 2021 (summary)](https://www.resourcefulfinancepro.com/articles/ap-benchmarks-2021/) — industry report, 2021  
**Used for:** Problem statement; industry figure, not peer-reviewed - treat as directional.  
**Check:** passed

### CLM-AP-2 · ap

**Claim:** APQC reports a median of 12,000 invoices per AP FTE per year (3,748 companies).  
**Excerpt:** "median of 12,000 invoices processed per accounts payable FTE ... 3,748 companies"  
**Source:** [AP Automation ROI Benchmarks (Tipalti, citing APQC)](https://tipalti.com/blog/ap-automation-roi-benchmark/) — industry report, 2026  
**Used for:** Throughput baseline for the AP module's ROI estimate.  
**Check:** passed

### CLM-DATA-1 · datasets

**Claim:** DocILE has 6,680 annotated business documents, 100,000 synthetic and about 932,467 unlabeled documents.  
**Excerpt:** "annotated 6,680 ... synthetic 100,000 ... unlabeled 932,467 (Table 2)"  
**Source:** [DocILE Benchmark for Document Information Localization and Extraction (arXiv 2302.05658)](https://arxiv.org/pdf/2302.05658) — peer-reviewed (ICDAR 2023), 2023  
**Used for:** Invoice layout pre-training (non-commercial license: research/portfolio only).  
**Check:** passed

### CLM-DATA-2 · datasets

**Claim:** FATURA comprises 10,000 annotated invoices with 50 distinct layouts.  
**Excerpt:** "Comprising 10,000 invoices with 50 distinct layouts"  
**Source:** [FATURA: A Multi-Layout Invoice Image Dataset (arXiv 2311.11856)](https://arxiv.org/abs/2311.11856v1) — preprint, 2023  
**Used for:** Invoice field extraction training for the AP module.  
**Check:** passed

### CLM-PDM-1 · pdm

**Claim:** C-MAPSS FD001 has 100 training and 100 test engine trajectories under one operating condition.  
**Excerpt:** "FD001 Train trjectories: 100 Test trajectories: 100 Conditions: ONE"  
**Source:** [C-MAPSS readme.txt (NASA PCoE data, GitHub mirror used by the fetch script)](https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/readme.txt) — dataset documentation (US Government work), 2008  
**Used for:** RUL model training/evaluation; files pinned by SHA-256 in aeronexus/public_data.py.  
**Check:** passed

### CLM-SAFE-1 · safety

**Claim:** ASRS narratives are not verified or validated by NASA, so they cannot be treated as confirmed facts.  
**Excerpt:** "NASA ASRS does not verify or validate reports"  
**Source:** [ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html) — dataset page (NASA), 2026  
**Used for:** ASRS hits are labelled 'unverified report' in retrieval and never counted as confirmed events.  
**Check:** passed

### CLM-SAFE-2 · safety

**Claim:** OpenSky Network grants only a limited, terminable license to its data.  
**Excerpt:** "limited, non-exclusive, non-transferable, non-assignable, and terminable license"  
**Source:** [OpenSky Network General Terms of Use & Data License Agreement](https://zenodo.org/records/6325961/files/LICENSE.txt?download=1) — data license, 2026  
**Used for:** Trajectory analytics kept in notebook 07; no OpenSky data shipped in the repo.  
**Check:** passed

### CLM-SAFE-3 · safety

**Claim:** The FAA publishes Service Difficulty Reports as one CSV file of all reports per year.  
**Excerpt:** "CSV file of all reports for that year"  
**Source:** [Download Service Difficulty Reports (SDR) by Year](https://www.faa.gov/av-info/download_SDR) — dataset page (FAA), 2026  
**Used for:** Manual export path: CSVs dropped into data/public/faa_sdr are indexed for retrieval as unverified reports.  
**Check:** passed
