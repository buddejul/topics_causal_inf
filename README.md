# Topics Causal Inference

## Paper Structure

### Section 1 - Designing a high-dimensional DGP

Simulation results

- Replication DGP 3 WA 2018
- Extension DGP 4 with a larger number of covariates
- d = 8, 15, 30
- settings: 40 reps, 200 trees (for speed, also show SD)
- For all DGPs: Also show for simulation based on WGAN
  - Evaluate using true CATE; in principle, should be able to estimate this because for
    any DGP

### Section 2

- Important question: How do we evaluate?
  - Also RMSE or Coverage of true function, if that's what we care about in the end
  - Would need to think about some decision criterion (RMSE --> Squared loss.)
  - Look at missclassification error!
- Do generic_ml with the same causal forest method; maybe also Lass/Elastic net (with
  CV)

### TODO

- Implement causal forest with generic_ml
- Coverage in addition to RMSE
- Add "mixed" DGP where only the CATE is specified
