# Professor feedback on the Skill-MoE project

## Revised aim

> Can expert disagreement predict aggregation failures and enable calibrated, cost-efficient inference in Skill-MoE?

## Feedback summary

The points below summarize the available feedback. The revised aim above is quoted directly.

- Review calibration literature and use it to guide how you assess predicted probabilities.
- Focus the experiment on when aggregation adds value. Compare the LLM aggregator with expert voting on the same questions and examine when aggregation changes an incorrect vote into a correct answer.
- Account for repeated model selections. Three selected slots can contain fewer than three distinct models, and disagreement based on three outputs is already a coarse signal. Add a second disagreement measure alongside the modal vote-share measure.
- Compare expert disagreement with router confidence. Consider AUROC for predictive performance and report the fraction of aggregator calls the policy skips alongside answer accuracy.
- Compare the adaptive policy with always using the aggregator, always using expert voting, and randomly bypassing the aggregator.
- Consider MedMCQA and MMLU-Pro. Build a question-level dataset that records the individual expert answers and the aggregator answer so both can be evaluated on the same questions.

## Calibration references shared by the professor

- [JAMIA article on clinical model calibration](https://academic.oup.com/jamia/article/27/4/621/5762806)
- [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
