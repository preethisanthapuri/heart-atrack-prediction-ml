# Publication readiness assessment

**Assessment date:** 2026-10-06  
**Project:** `heart-atrack-prediction-ml`  
**Decision:** The completed work is suitable for internal review as an exploratory benchmark manuscript. It is **not ready for submission as a heart-attack prediction or clinical-risk paper**. Author and venue checks plus scientific provenance work remain before any submission.

This assessment checks the manuscript and project artifacts against the evidence currently in the workspace. It does not certify the accuracy of external references, guarantee IEEE acceptance, or substitute for a domain expert review.

## Readiness by area

| Area | Status | Evidence and action |
|---|---|---|
| Reproducible project artifacts | Present, with local-data caveats | Modular source, run scripts, saved result catalog, and a recorded 38/38 test run exist. Re-run the documented pipeline from a clean environment before submission and archive exact dependency versions and commands. |
| Data provenance and rights | **Blocking** | The 303-row file resembles a familiar benchmark but provenance is unverified. Record the original source, version, license/permission, transformations, and checksum; cite the source file. Do not infer source from column similarity. |
| Endpoint and title | **Blocking** | The field `output` has codes 0/1 but no verified codebook, event definition, or prediction horizon. The manuscript must not be submitted as heart-attack prediction or future cardiovascular risk. Verify semantics or retitle and bound the paper to an unlabeled-source benchmark. |
| Primary analysis | Present, requires author audit | Seven baselines, fold-local preprocessing, tuning, feature selection, calibration, uncertainty, SHAP, external, ablation, and subgroup artifacts exist. The final choice is Stage B, with a documented accuracy/F1 tradeoff. Reconcile every table and figure against its source CSV once author-facing materials are frozen. |
| Holdout integrity | **Material limitation** | The 61-row holdout was screened for accuracy in Phase 5. It must not be represented as untouched. The final Stage B was selected from training-only OOF results, but this does not undo the earlier holdout screening. A new independent cohort or a prospectively locked protocol is needed for a clean confirmatory estimate. |
| External validation | **Not established for final model** | Historical UCI Hungary was scored with the Phase 6 baseline, not final Stage B. There is severe missingness, a likely reversed class mapping, and a one-record target-count discrepancy. Resolve mapping from authoritative source records and evaluate the frozen Stage B on a defensibly aligned cohort before making external-validation claims. |
| Calibration and uncertainty | Exploratory only | Calibration and conformal results concern baseline/voting candidates rather than Stage B. Either evaluate the selected final pipeline with nested, leakage-safe methods or present these phases as exploratory comparators without transferring the result to Stage B. |
| Explainability | Exploratory only | SHAP was run for the Phase 6 baseline on training rows. Do not label it an explanation of the final Stage B model. If explanations are central to the paper, run and document a valid Stage B explanation analysis. |
| Fairness | Descriptive only | Subgroup metrics have small groups and unverified codebooks and labels. Avoid fairness or equity claims. Define groups and intended use and evaluate uncertainty-aware subgroup metrics on appropriate independent data before such claims. |
| Statistical interpretation | Needs tightening | Fold standard deviations are descriptive, not confidence intervals; multiple models and stages were compared on small data. Avoid “significant,” “robust,” or superiority language unless supported by an appropriate prespecified inferential design. |
| Literature and references | Needs bibliographic verification | A focused narrative review and references are present. Confirm each citation against the publisher/repository metadata, verify all identifiers, update venue-specific literature coverage, and distinguish similar but non-comparable endpoints/datasets. The review is explicitly not systematic. |
| Manuscript format | **Not submission formatted** | `paper/ieee_paper.md` is a Markdown draft. Add verified author names, affiliations, and required declarations; select a specific IEEE conference; apply its current official template, page limit, and submission rules; create and inspect final PDF. IEEE conference requirements vary by venue. |
| Abstract and keywords | Author check | IEEE guidance specifies a self-contained abstract of up to 250 words and 3–5 keywords. The current abstract and six keyword phrases require editing/count verification before submission. |
| Reporting checklist | Needs completion | Use TRIPOD+AI to assess transparent reporting for prediction-model studies, but first clarify whether this dataset/task truly constitutes a prediction model. Complete applicable items and explain any not applicable. Consider PROBAST+AI risk-of-bias appraisal for the design and intended setting. |
| Ethics and governance | Author verification required | Identify the dataset's license and any applicable consent, privacy, ethics-review, and institutional requirements. The present files do not establish whether an exemption or review applies. Do not invent an approval statement. |
| Claims and intended use | Bounded appropriately | The manuscript and app state that this is exploratory and not a diagnostic tool. Keep the target as “dataset class” until independently verified. |

## Claims currently supportable

- The supplied file has the reported dimensions and the saved cleaning step removed one exact duplicate.
- On the saved 241-row training partition, the reported five-fold baseline cross-validation estimates show Logistic Regression ROC-AUC 0.908 ± 0.039 and accuracy 0.859 ± 0.045.
- In the staged pooled OOF comparison, Stage B led on ROC-AUC (0.916), average precision (0.922), and Brier score (0.112); Stage A had higher accuracy (0.859) and F1 (0.872).
- The integrated Stage F configuration did not outperform simpler stages across reported metrics.
- The project documents an exploratory, leakage-aware empirical comparison for a small supplied tabular dataset.

## Claims currently unsupported

- Heart-attack prediction, myocardial-infarction diagnosis, prospective risk, or a specified prediction horizon.
- Clinical validity, utility, safety, population-level generalization, or deployment readiness.
- An untouched holdout estimate, confirmatory external validation of Stage B, or calibrated/uncertainty-aware Stage B output.
- Causal interpretation from SHAP or a fairness/equity conclusion from descriptive subgroup comparisons.
- Novel algorithmic contribution or general superiority of feature selection, tuning, calibration, or ensemble learning.

## Required actions before submission

1. Obtain authoritative provenance, codebook, endpoint definition, units, source license, and label orientation for the supplied data. Resolve or retain the paper's neutral dataset-class framing.
2. Choose a specific IEEE conference and verify its current scope, deadline, page limit, template, anonymization, and artifact rules on that conference's official site.
3. Verify each reference's authors, title, journal/conference, year, pages/article number, and DOI against publisher or repository records.
4. Reconcile every manuscript number with saved CSV/JSON evidence, make the six-keyword list conform to the target venue's requirements, and ensure the abstract is within the required word count.
5. Add author names, affiliations, funding, conflict-of-interest, data/code availability, ethics/privacy, and acknowledgments only after the authors verify the facts.
6. Review model-development and model-reporting limitations with a domain expert. Apply the relevant items of TRIPOD+AI and a suitable risk-of-bias framework.
7. Decide whether a new untouched independent evaluation can be obtained. If not, keep the exploratory scope and prior holdout screening prominent in the title, abstract, methods, results, and discussion.
8. If the intended contribution depends on calibration, uncertainty, SHAP, subgroup fairness, or external validation, evaluate those components on the actual selected Stage B pipeline using a prespecified design and appropriate independent data.
9. Build the manuscript with the venue's official template, check references and figure/table placement, produce the required PDF, and run the venue's PDF compliance checker where required.
10. Ask all authors to approve the final manuscript and disclosures before submission.

## Official guidance consulted

- [IEEE conference authoring tools and templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/)
- [IEEE guidance on structuring a paper](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/)
- [IEEE conference paper types and requirements](https://conferences.ieeeauthorcenter.ieee.org/become-an-ieee-conference-author/types-of-ieee-conference-papers/)
- [IEEE Xplore PDF requirements](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/meet-ieee-xplore-requirements/)
- [TRIPOD+AI scope and checklist context](https://www.tripod-statement.org/scope/)
- [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505)

IEEE guidance describes general authoring practices. The selected conference's official instructions control the final format, submission route, and page limit.
