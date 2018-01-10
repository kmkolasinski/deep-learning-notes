# CapsulePooling2D

CapsulePooling2D is a Keras layer which is a naive inspiration taken from original [Capsules paper](https://arxiv.org/abs/1710.09829) applied to the Convolutional networks. The assumption was to create a Layer which will implement dynamic routing between so caled capsules and will preserve translational invariance of the whole network. Here the capsule is just a single hypercolumn of feature map. `CapsulePooling2D` works just like normal pooling layer except it iteratively chooses which feature column will be pooled within pooling region. So each pooling region can be considered as a separate capsule problem.

# Algorithm

In the iterative process firstly the average feature column is computed by normal avg pooling operation, 
then a scalar dot product of that column is computed with rest of the capsules in the corresponding pooling area. 
This results in 2D map of scores. In the next step computed score map is normalized with softmax function for each pooling region (`score_map` in the code). This is done by `normalize_pool_map` function i.e. after this operation scores in each pooling region sum to one. Having this, the input feature map is multiplied by those score weights `FeatureMap = FeatureMap * NormalizedScoreMap`.
Then average pooling is again computed to estimate new average column, but this time every feature column has different weight.
The `score_map` is aggreated during iteration process list it was done in original paper. After few iteration the algorithm should converge and result in pooling like operation, which chooses most dominating column from pool area. 

# Notes

* It seems it does not work :D
* Compared with MaxPooling (see notebook)
