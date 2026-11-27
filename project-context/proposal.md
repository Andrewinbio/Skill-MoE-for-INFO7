# Skill-MoE aggregation and calibration

Andrew DePass · INFO 762 Biomedical NLP · Fall 2026

**Approved implementation amendment:** The primary experiment is now the
same-model Qwen3-8B sampled-response study specified in
`primary-protocol-final.md` and `Skill-MoE-for-INFO7/analysis/primary-study.json`.
It uses three Qwen3 draws plus Qwen3 aggregation over 4,178 non-smoke questions,
an 8 GPU-hour cap, frozen fit/validation/held-out splits and a 2-percentage-point
accuracy-loss tolerance. Router-confidence comparisons and heterogeneous
recruitment are outside this approved scope. The original proposal text below
is retained to make the research-scope amendment explicit.

## Research question

> Can expert disagreement predict aggregation failures and enable calibrated, cost-efficient inference in Skill-MoE?

I will test whether answers from the selected experts provide a useful signal about two outcomes: whether the LLM aggregator will answer incorrectly and whether calling it will improve on expert voting. I hypothesize that disagreement adds predictive information beyond maximum router selection probability and that selective aggregation can reduce aggregator calls within a predefined accuracy tolerance.

## Data and scope

Use the authors’ Skill-MoE implementation and available benchmark data. Start with one feasible multiple-choice benchmark subset and an existing expert configuration. Consider MedMCQA and MMLU-Pro based on availability and compute. Start with a small smoke test and budget for one GPU on Cheaha. Record the upstream revision, model configuration, question IDs, and run settings.

For each question, record router probabilities, selected expert IDs in draw order, normalized expert answers, the aggregator answer, and the reference answer for scoring. Record stage-level timing and token counts when available. Score both expert voting and aggregation on the same questions, using a documented tie-break and invalid-answer policy.

## Predictors and outcomes

- **Modal disagreement** is one minus the fraction of expert votes cast for the most frequent answer.
- **Answer entropy** provides a second disagreement measure. Document its normalization and handling of missing or invalid answers.
- **Router confidence** is the highest expert-selection probability. Use it as a comparator available before expert generation.
- **Repeated selections** can leave fewer distinct models than selected slots. Record the number of unique experts and compare the per-slot analysis with a sensitivity analysis that gives each distinct expert one vote.

Model two binary outcomes separately. **Aggregation failure** means the aggregator answer is wrong. **Helpful aggregation** means voting is wrong and aggregation is correct. Also count **harmful aggregation**, where voting is correct and aggregation is wrong, because both outcomes affect the accuracy of an adaptive policy. Use only information available before aggregation as policy inputs.

## Calibration and policy evaluation

Split questions into fit, validation, and held-out evaluation sets, keeping duplicate questions together. Fit simple logistic models using router confidence alone, disagreement features alone, and both together. Assess discrimination with AUROC and AUPRC and assess probability calibration with Brier score and reliability plots. Select the policy threshold on validation data and freeze it before held-out evaluation.

Compare always aggregating, always voting, random bypass matched to the adaptive call rate, a raw-disagreement threshold, and a calibrated policy that invokes the aggregator when the predicted probability of helpful aggregation exceeds the threshold. Report held-out answer accuracy, paired differences, aggregator call and skip rates, and measured timing or token use where available. In a replay-only evaluation, report skipped calls as a cost proxy. Expert generation remains part of the cost of the adaptive policy.

## Deliverables

Produce a runnable GitHub repository with Cheaha instructions, reproducible analyses, and a course poster with methods, results, limitations, repository link, and actual team contributions.

## Decisions to settle in Step 01

- Confirm the initial benchmark subset, expert configuration, and compute budget.
- Set the maximum acceptable accuracy drop relative to always aggregating before tuning.
- Agree on team ownership and record any benchmark or checkpoint access constraints.

## References

- [Skill-MoE paper](https://openreview.net/forum?id=M2pbI9uKjW)
- [Upstream code](https://github.com/dinobby/Skill-MoE) and [benchmark data](https://github.com/dinobby/Skill-MoE/tree/main/test_data)
- [JAMIA calibration reference](https://academic.oup.com/jamia/article/27/4/621/5762806)
- [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
