# CapsulePooling2D

A naive translation from original Capsules idea to Convolutional networks.
The assumption was to create Layer which will implement dynamic routing 
between capsules which will preserve translational invariance of the whole
network. `CapsulePooling2D` works just like normal pooling layer except it iteratively 
chooses which feature column will be pooled, so each pooling region can
be considered as a separate capsule problem. 

# Algorithm

In the iterative process firstly the average feature column is selected `C_avg` (normal avg pooling is applied), 
then scalar dot product of that column is computed with rest of the columns. This results
in 2D map of size of pooling region. The computed 2D map is then normalized with softmax function for each pooling region (`score_map` in the code). This is done by `normalize_pool_map` function. Having computed weights in the pooling region,
the input feature map is multiplied by normalized score weights `FeatureMap = FeatureMap * ScoreMap`.
Then average pooling is again computed to estimate new `C_avg`. The `score_map` is aggreated during iteration process.

