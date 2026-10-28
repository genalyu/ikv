# Experimental v3 supervision screening

This opt-in module leaves all v2 runs and runtime bindings unchanged. It retains history-only eligibility, all score labels including boundary ties, graded pair preferences, and the tested fixed-mass soft selector.

Two independently switchable supervision changes are screened: regression allocates .4/.2/.4 mass to teacher scores above/equal/below the Top-k boundary (renormalized over nonempty strata), and half of ranking comparisons preferentially sample STRICT teacher positives rather than allowing a large boundary-tie pool to dominate. Other comparisons remain random; negative candidates mix random and student-high-scoring items. There is no early-frame bonus, age quota or group label. All-equal teacher labels have zero pair preference and retain calibration supervision.

Use the predeclared tau .03 screening value consistently in the v2 reference and all three v3 ablations (strata-only, strict-sampling-only, both). Fix initialization, data, current/history mask, optimizer, sampler, schedule and exposure count. Screen on episode-disjoint development only, then replicate a useful setting across retained sizes before capacity claims. Original validation/test are not optimization targets. This is a loss ablation proposal, not evidence of deployability or label deficiency.
