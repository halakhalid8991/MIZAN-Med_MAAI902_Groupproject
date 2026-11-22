# MIZAN-Med

**Evidence grounded health claim verification with calibrated abstention and a cross lingual Arabic evaluation**

MIZAN-Med checks a short health claim, typed or spoken in English or Arabic, against research abstracts from PubMed. It shows the exact sentences behind its answer and the study design they come from, and it gives a verdict only when the evidence and the model's calibrated confidence are strong enough. Otherwise it reports that there is not enough evidence. The name comes from the Arabic word *mizan* (ميزان), meaning balance or scale.

> **Research prototype only. This is not medical advice.** Verdicts and translations can be wrong. Do not enter personal health information.

Group project for **MAAI 902: Natural Language Processing and LLMs**, University of Wollongong in Dubai, 2026.

---

## How it works

```
Claim (English or Arabic, typed or spoken)
   │
   ├─ Spoken claim → text with Whisper base (the user reviews and edits the transcript)
   │
   ├─ Arabic → English with NLLB-200 (the user can correct the English)
   │
   ▼
1. Retrieve   PubMed search (up to 25 abstracts), ranked by BM25 + BGE-base (reciprocal rank fusion)
2. Verify     Each abstract is split into sentences; a fine tuned DeBERTa-v3-large NLI model
              labels the top 3 sentences of the top 3 abstracts as SUPPORT, REFUTE or NEUTRAL
3. Abstain    Temperature scaled confidence + evidence rules tuned on SciFact training claims
4. Answer     Evidence leans supportive · Evidence leans against · Mixed evidence · Not enough evidence
5. Explain    Extractive explanation by default; an LLM (Qwen3-4B, run locally) explanation is shown
              only if its citations are valid and every sentence is entailed by the cited evidence
```

## Speech input

The website also accepts spoken claims. Open **🎙️ Speak your claim · تسجيل الادعاء**, choose Auto detect, English or Arabic, record up to 30 seconds, stop, and press **Transcribe recording**. The multilingual `openai/whisper-base` model transcribes the recording on the CPU, so it does not compete with the verifier for GPU memory. The transcript is added to the claim box and does not start the check automatically, so it can be corrected first.

* The first transcription downloads the Whisper model and can take a few minutes.
* Recordings are processed on the server hosting the demo and are not saved to project files; Gradio may keep temporary audio files during a session.
* Another multilingual Whisper model can be used by setting `MIZAN_SPEECH_MODEL`. Do not use an English only model for Arabic.
* This is a usability feature. It was tested informally and is **not** part of the reported evaluation.

## Main results

All numbers are produced by the notebook; nothing is hard coded.

| Research question | Result |
|---|---|
| **RQ1 Retrieval** (SciFact dev) | Dense BGE-base found a relevant abstract in the top 10 for **96.2%** of claims (nDCG@10 0.878); equal weight fusion with BM25 lowered it to 0.854 |
| **RQ2 Verification** (HealthVer test, 1,823 pairs) | DeBERTa-v3-large NLI reached **0.817 ± 0.003** macro F1 over 3 seeds, against 0.647 for TF-IDF and 0.561 for zero shot Qwen3-4B |
| **RQ3 Abstention** (SciFact dev, 300 held out claims) | Tuned evidence rules answered **45.3%** of claims at **73.5%** accuracy; without rules 56.7% at 66.5%; the earlier hand set rules answered none |
| **RQ4 Explanations** | All LLM explanations cited valid PubMed IDs, but only **16.8%** were entailed by the evidence they cited |
| **RQ5 Arabic** (same 1,823 pairs) | Translate test with NLLB-200 lost **1.2** points of macro F1 (not significant), OPUS-MT 2.4, and a multilingual verifier reading Arabic directly 4.0 |

A native Arabic speaker reviewed the 61 claims whose prediction changed after translation (seed 42 of the repeat run): 68.9% kept their meaning in translation, 24.6% shifted in meaning and 6.6% lost a negation.

**Reproducibility check.** The full notebook was run a second time on another machine. Everything that does not involve GPU training came out identical; the verifier scored 0.821 instead of 0.817, and every main conclusion held.

## Repository contents

| File or folder | Contents |
|---|---|
| `MIZAN_Med_v0_6_2.ipynb` | Main executed run: data audit, baselines, training, evaluation, ablations and the website. The main results in the paper and report come from this run |
| `MIZAN_Med_Hala_final.ipynb` | Repeat run of the same notebook on a second machine (verifier macro F1 0.821), with two extra cells: the single pair calibration figures and confusion matrix, and the native speaker review of the 61 changed Arabic predictions. Its website module also includes the speech input |
| `mizan_core.py`, `mizan_research.py`, `mizan_llm.py`, `mizan_website.py` | Pipeline modules (the notebook also writes these files itself when run). `mizan_website.py` contains the Gradio website, including the speech input |
| `docs/` | Final paper, Assessment 3 report and presentation |

The dataset bundle (`MIZAN_Med_Datasets.zip`, with `MANIFEST.json` holding the SHA-256 checksums, sources and licences) and the human review files (`MIZAN_Med_Review_Files.zip`) are kept in the project Google Drive folder, not in this repository:

**[MIZAN-Med Google Drive folder](https://drive.google.com/drive/folders/1ScUzF-NM8kCJgKGTWjLX0eu47k-jAVHx?usp=drive_link)**

They are shared for academic assessment only (see [Datasets](#datasets)).

## Running the notebook

### Google Colab (recommended)
1. Open `MIZAN_Med_v0_6_2.ipynb` in Colab and choose a GPU runtime (an NVIDIA L4 was used; a T4 also works, but more slowly).
2. Download `MIZAN_Med_Datasets.zip` and `MIZAN_Med_Review_Files.zip` from the [project Google Drive folder](https://drive.google.com/drive/folders/1ScUzF-NM8kCJgKGTWjLX0eu47k-jAVHx?usp=drive_link) and put them in **My Drive/MIZAN-Med** (or upload them when the notebook asks). Keep them zipped.
3. Run the cells from the top. After the first install cell, restart the session (**Runtime → Restart session**) and run all cells again.
4. In **Experiment controls**, set `EXPERIMENT_MODE = 'SMOKE'` for a quick test or `'FULL'` for the reported results. SMOKE outputs are never valid results.
5. Run **Start the website** last. It prints a temporary `gradio.live` link, valid for 72 hours. Open the link (not `localhost`) and allow microphone access to use the speech input.

To open only the website after a FULL run, run the setup cells (environment, install, workspace, the four `%%writefile` cells and the module import), skip the experiments, and run **Start the website**. The trained verifier is loaded from Drive.

Finished training runs are saved to Drive and reused, so an interrupted run can be resumed by running the cells again.

### Own computer
Install PyTorch with CUDA first, then run the notebook in Jupyter. Set the project folder before starting, for example:

```bash
export MIZAN_PROJECT_DIR="/path/to/MIZAN-Med"     # Windows: set MIZAN_PROJECT_DIR=C:\path\to\MIZAN-Med
```

### Main libraries
PyTorch, Hugging Face `transformers` (≥4.51, <5), `datasets` 3.6.0, `sentence-transformers` 3, `rank_bm25`, scikit-learn, SciPy, `sacrebleu`, `gradio` 4.44.1 (Whisper speech input through the `transformers` pipeline and `scipy` for resampling). Exact versions are pinned in the install cell.

## Datasets

The datasets were downloaded from their official sources and are checked against SHA-256 checksums before loading. They are shared through the [project Google Drive folder](https://drive.google.com/drive/folders/1ScUzF-NM8kCJgKGTWjLX0eu47k-jAVHx?usp=drive_link) only so that the work can be assessed and reproduced for the MAAI 902 subject. They remain the property of their authors, must be cited, and must not be reused or redistributed outside the terms of each licence below.

| Dataset | Role | Licence | Source |
|---|---|---|---|
| HealthVer | Train and test the verifier; Arabic test set | No licence stated; research use with citation | https://github.com/sarrouti/HealthVer |
| SciFact | Retrieval, rule tuning, end to end test | CC BY 4.0 (claims), ODC-By 1.0 (abstracts) | https://github.com/allenai/scifact |
| PUBHEALTH | Claim only shortcut test | MIT | https://github.com/neemakot/Health-Fact-Checking |
| QCRI COVID-19 Disinformation (Arabic) | Arabic text normalisation test | CC BY-NC-SA 4.0 | https://huggingface.co/datasets/QCRI/COVID-19-disinformation |

AraCOVID19-MFH and PubMedQA were screened but excluded (tweet IDs only; question answering rather than claim verification).

## Models used

| Component | Model |
|---|---|
| Verifier | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, fine tuned on HealthVer |
| Multilingual verifier | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, fine tuned on HealthVer |
| Retrieval | BM25 + `BAAI/bge-base-en-v1.5` |
| Translation | NLLB-200 and OPUS-MT |
| LLM | `Qwen/Qwen3-4B-Instruct-2507`, run locally |
| Independent checker | DeBERTa-v3-base NLI cross encoder |
| Speech input | `openai/whisper-base` (multilingual), run on the CPU |

## Limitations

* HealthVer covers COVID-19 only, so other health topics are untested.
* The Arabic claims are machine translated Modern Standard Arabic, not real or dialectal rumours.
* The evidence rules were tuned on SciFact, not on live PubMed, and no user study has been carried out.
* LLM explanations were checked automatically, not by human reviewers.
* The speech input was tested only informally; its accuracy on Arabic, and especially Gulf dialect, has not been measured.
* NLLB-200 and the QCRI data are licensed for non commercial use only, so the system as built is for research use only.

## Team

| Member | Main role |
|---|---|
| Hala Khalid Sarhan | Idea, data, initial code, Arabic evaluation, paper, and presentation |
| Nourhanne Abdelaziz | Idea, initial code and literature review and report |
| Mohamed Sakr | Idea, data, initial code, speech to text feature and demonstration |
| Harley Douti-Mama | Code development, fine tuning, experiments and demonstration |
| Sanzhar Seisenov | Presentation support |

Instructor: Dr. Pallavi Ranjan.

## Use of generative AI

Generative AI tools, including Claude (Anthropic), assisted with code development, debugging and editing. All experiments were run by the team, and all results were checked against the notebook outputs.
