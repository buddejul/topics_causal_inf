# Notes

## TODOs

Todo ds-wgan

- Check generator objects can be used in python --> works: safe df as well and downgrade to numpy 2.0
- Repeat simulation for dgp3 to dgp4 (maybe dgp5)
- Want to Show:
  - simulations based on wgan get RMSE right when dim is low (dgp3)
  - simulations based on wgan get RMSE wrong when dim is high (dgp4)

## August 8

- Hard to think about how to evaluate a CATE estimator using this approach:
  - Even if the wgan could perfectly learn the joint distribution, hard to think about computing MSE
  - In practice we wouldn't know the true treatment effect function, so we have to estimate it in a first step
  - but if it is high-dimensional, we cannot estimate it consistently
  - hence this is not a useful approach to designing CATE estimation monte carlos, except for lower-dimensional settings
  - we could still evaluate performance if we treat the true CATE function as known; if we can show there is no reason to believe it would perform well when were estimating
  - Solution:
    - Generate large superpopulation once
    - Then sample from this population; in that case we know the CATE for any combination that could appear in the sample; can perform RMSE against a new sample from the superpopulation
