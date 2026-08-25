# Ensembles

`stacking.py` implements the shared chronological stacking primitive. Base models fit on
the early part of a training fold; the meta-learner receives only later out-of-sample base
probabilities, after which bases are refitted for test inference. All-nine voting,
probability averaging and market-specific selection are orchestrated in `../experiments/`.
