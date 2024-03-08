# Part I
### Bug Fixes
- fix parameters in barplot functions a it was not specifying x & y variables

### Choose Model
From the datacientist conclusions we can define that reducing to the 10 most important features and ballancing classes are both good ideas. But when choosing a model it does not give us much information.
after rerunning the experiment with different data splits (random split 40 and random split 41) we can see that the Logistic Regression Model outperforms the Desition Tree on the "not delayed prediciton" but underperfors on the "delayed prediction" (we are optimizing the model against the world not just a single test data), taking in to consideration that wrongly predicting a flight not delayed seems worse than the alternative, and that the difference for the delayed prediction are greater that for the not delayed we choose the Desition Tree as the best model for this case.

Another argument that can be given is that the improvement of the Desition Tree is far greater than the one for the Regression wich could we us greater chances of improvement in the Future.

**TL;DR: We choose de XGBoost**