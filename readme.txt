# Intial setup
Here we imported all the libraries that were need to make perform this model and also the data set needed.
We dropped the column id, unnamed 32, diagnosis and introduced a new column diagnosis_enc with the binary values instead of B and M.
Defined X axis and Y axis that we are going to use throughout the different models.

# Train set split
X= feature matrix
Y= Target labels
Test_size = 0.2, here we are defining that 20% of the data is set aside for testing and 80% of it for the training
random_state=42: Ensures reproducibility (same split every time you run it)
X_train = 80% of X  To train the model
X_test = 20% of X  To test the model
y_train = 89% of y Training labels
y_test = 20% of y  Test labels

X_train  (455,30)  455 rows 30 columns for training
y_train  (455,)    544 labels (1 per sample)
X_test   (144,30)  114 rows 30 columns for testing
y_test   (114,)    114 labels (1 per test sample)

# Decision tree
## Full decision tree
Defined the model, it Initializes a decision tree model.
y_pred = model.predict(X_test) Using trained model to predict labels (0 and 1) on unseen test data (X_test), it gives you the predicted classifications for each sample in the test set.
0.95 Accuracy means the model predicted 95% of test samples correctly.

🔹 concave points_mean <= 0.051  (This threshold was chosen because it helps the model best separate the target classes.)
If the tumors average number of concave points is less than or equal to 0.051? If YES, the sample goes to the left node(true), if NO the sample goes to the right node(False).
🔹 gini = 0.467
Gini = 0 → perfectly pure (all samples are of the same class)
Gini = 0.5 → maximum impurity (equal mix of classes)
A Gini of 0.467 means this node has some class mixing, not perfectly pure, but slightly biased toward one class.
🔹 samples = 455
The node contains 455 samples
🔹 values = [286, 169]  (This is the breakdown of 455 samples)
286 are class 0 (Benign)
169 are class 1 (Malignant)
It says that the node is majority class = Benign, but there is a significant number of Malignant cases too.
🔹 class = Benign     This is the predicted class at this node — the one with the majority count (286 > 169).

Following the diagram, on the basis of sepration concave points_mean <= 0.051 gets divided into two parts: True and False

True: Radius_worst <= 16.83
Of the 455 samples, 282 samples satisfy both concave points_mean <= 0.051 and radius_worst <= 16.83.
Among these 282 samples:
266 are Benign
16 are Malignant
The Gini impurity is 0.107, which is quite low, meaning this node is mostly pure (mostly benign samples). The predicted class here remains Benign because the majority are benign.

False:
(Radius_worst > 16.83):
The other 173 samples meet the condition concave points_mean <= 0.051 but have radius_worst > 16.83.
This node further splits based on:  concave points_worst <= 0.147
For this node, overall:
20 samples are Benign
153 samples are Malignant
The Gini impurity is 0.204, indicating this node is mostly pure (mostly malignant samples).
The predicted class here is Malignant because most samples are malignant.

At the root split, the tree looks at concave points_mean to divide all samples.
For samples where concave points_mean <= 0.051, the next split checks radius_worst.
If radius_worst is small (≤ 16.83), the node is predominantly Benign with low impurity (good separation).
If radius_worst is larger (> 16.83), the node tends toward Malignant cases, but it further checks concave points_worst to split again.
The tree is using these features to separate the benign and malignant classes effectively by choosing thresholds that best reduce impurity.

### Decision tree: pre-pruning
pre-pruning = We are pre-pruning the model and limited its maximum depth to 3, the purpose of this plot is to preven the overfitting.

We trained a simpler, shallower decision tree to reduce overfitting.
Evaluated accuracy to check if this pruning still performs well.
Visualized the pruned tree to understand the key decision splits and how the model is classifying the data.

Feature importance:
model.feature_importances_= It gives us a numeric score for each feature (column in X) based on how much it contributes to reducing impurity (like Gini) across the tree, Higher score = more important for making decisions.
The length of the bar tells us how important that feature is in the decision tree.
The decision tree relied on the most on these features = Concave_points_mean, radius_worst, concave points_worst, texture_mean, perimeter_worst, fractal_dimension_se, texture_worst, concave points_se, area_se, smoothness_worst, concavity_se.

Matrix of confusion:
68 True Negatives (TN): Benign correctly predicted as Benign
3 False Positives (FP): Benign wrongly predicted as Malignant
3 False Negatives (FN): Malignant wrongly predicted as Benign 
40 True Positives (TP): Malignant correctly predicted as Malignant
Classification report:
Precision:
For class 1 (Malignant): 93% of the time when the model said "malignant", it was right.
Recall:
For class 1: 93% of malignant cases were detected.
F1-score:
Balance between precision and recall. It is Useful when both false positives and false negatives are important.
Accuracy:
(TP + TN) / total = (68 + 40) / 114 = 0.947 ≈ 95%
The model gives only 6 total errors out of 114.
False negatives (3) are especially important in medical contexts — we're catching 93% of malignant tumors, which is strong.

ROC CURVE:
AUC (Area Under the curve) = 0.96 = 96%
The model is very good at distinguishing between benign and malignant cases.
At various thresholds, the classifier has high true positive rates and low false positives.

### Decision tree: post-pruning
This method computes the effective alphas (regularization strengths) used for pruning. Smaller ccp_alpha (Cost-Complexity Pruning Alpha) means more complex tree; higher ccp_alpha results in simpler trees.
CCP_ALPHA controls the trade-off between: 
The complexity of the tree (number of nodes), and
Its accuracy on the training data.
Post-pruning tries to simplify the tree by cutting off branches that have little predictive power, reducing:
Overfitting,
Model complexity,
And often improving generalization.
Best accuracy after post-pruning: 0.96  This means the pruned model performs very well — and possibly better than the original unpruned tree

### Cross validation
best_model correctly predicted 96% of the test labels, which is a very high.

# Matrix of correlation
It is a table showing the correlation coefficients between multiple variables. It helps you understand how strongly variables are related to each other (from -1 to +1)
+1.0: Perfect positive correlation
0.0: No correlation
–1.0: Perfect negative correlation
Here we took only the variables that were more important as per the diagram of feature selection. It helped us to simplify the data.
Our matrix of correlation explains: All the selected variables are not correlated with each other and each variable is individually has equal importance.

# Violin plot
A violin plot is a powerful data visualization tool that shows the distribution, probability density, and summary statistics (like median and quartiles) of a variable
*concave_points_mean:
                        Malignant (0): Distribution is higher overall
                        Benign (1): Lower values, more concentrated toward zero
                        Highly discriminative feature — good for classification
*radius_worst:
                Malignant: Higher range and density at larger values
                Benign: Lower, tighter range   
                Also a strong separator — useful for classification
*fractal_dimension_worst:
                            Both classes overlap more
                            Similar distributions but slightly higher variance for malignant
                            Might be less useful on its own
*area_se:
            Malignant: High variance and wider spread
            Benign: Lower and tighter
            Likely to be helpful, especially with models that use variance (e.g., Random Forest)

# Logistic regression:
*Co-ef
Positive coef → increases probability of benign.
Negative coef → increases probability of malignant

*std err (Standard Error):
Indicates how precise the coefficient estimate is.
Smaller is better — large errors imply high uncertainty in the estimate.

*z (Z-score):
Computed as: coef / std err
Measures how many standard deviations the estimate is from 0.
The higher in absolute value, the more "statistically significant" the variable.

*P>|z| (P-value):
Probability the coefficient is actually zero (i.e., the variable has no real effect). If p < 0.05, the variable is considered statistically significant.

✔️ Significant variables here (p < 0.05):
concave points_worst (p = 0.001)
texture_mean (p = 0.000)
radius_worst (p = 0.016)
concave points_se (p = 0.002)
area_se (p = 0.000)
❌ Not significant (p > 0.05):
concave points_mean (p = 0.184)
fractal_dimension_worst (p = 0.790)

concave_points_se: Very strong negative effect → higher values are highly associated with malignancy.
area_se, texture_mean, radius_worst: Positively associated with benign outcomes (or lower cancer risk).
fractal_dimension_worst: Not significant; wide CI suggests it's not a reliable predictor here.

# Matrix of confusion
68 samples were actually benign and correctly predicted as benign.
3 samples were actually benign, but the model incorrectly predicted them as malignant.
40 samples were actually malignant and correctly predicted as malignant.
3 samples were actually malignant, but were incorrectly predicted as benign.