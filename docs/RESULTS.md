# Reading the evidence

All included numbers are aggregate projections of preserved main-project outputs.
PROVENANCE.json records the exact inputs and transformations. No fresh training or
simulation is represented by this portfolio export.

Controller comparisons use aggregate totals across 20 routes per package. Positive
reductions mean less of a metric; waiting/time-loss regressions remain visible.
The independent alternate comparison is retained to show route sensitivity.
The baseline is the existing topology-specific controller portfolio, not always
the default SUMO program. Historical ranker selection used a development dataset;
the reviewed trainer now uses a separate final test split for future fits.

Waiting is integrated stopped time, time loss is a speed-deficit proxy, stops are
threshold transitions, and deceleration sums speed reductions. CO2 integrates
SUMO's modeled emission rates. These measures do not establish measured real-world
emission reductions or universal comfort/safety improvements.

Reconstruction events are anonymous. Boundary observations were used for fitting,
so accurate replay is not independent reconstruction ground truth. Internal
coverage and unmatched-penalized loss matter alongside matched timing MAE. A newer
physically valid trial is retained even though boundary tolerance blocks promotion.

Forward evaluation holds out whole vehicle identities, across their repeated
simulations. The comparison baseline receives the true detector sequence when
predicting timing/motion. It is intentionally strong but is not a detector-classifier
baseline. Several motion targets still regress; the surrogate is not yet an
accepted replacement for physical inverse validation.
