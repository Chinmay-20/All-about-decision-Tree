# All-about-decision-Tree

Decision tree algorithm
Class pdf
A decision tree produces a sequence of rules that can be used to classify data.
It builds classification or regression models in the form of tree structure.
It breaks down a dataset into smaller and smaller subsets while at same time an associated decision tree is incrementally developed.
Final result is a tree with decision nodes and leaf nodes
Decision node has 2 or more branches and the leaf node represents classification or decision.
Top most decision node is the root node.
Decision trees can handle both categorical and numerical data.
Disadvantage: it can create complex tree that do not generalize well 
Decision trees are unstable because small variations in data might result in completely different trees being generated.

Hands on machine learning with scikit learn and tensorflow
Decision tree can perform classification, regression and multi output tasks.
Decision tree are fundamental concept of random forests.
Decision tree don’t require feature scaling or centering at all.
A node’s samples attribute(refer diagram) count how many training instances it applies to. Eg 100 training instances have petal length greater than 2.45 among which 54 have s petal width smaller than 1.75cm
The node’s value attribute tells you how many training instances of each class this node apple to: eg  values = [0,1,45] it tells that 0 for setosa, 1 for versicolor, 45 for virginica.
gini attribute measures impurity. A node is pure of gini=0. And it happens only if all training instances it applies to belong to same or to be specific one class. There is no mixture of other classes.
For eg depth 1 left node applies to only setosa training instances so it is pure.
Depth 2 left node gini score is 1 - (0/54)2 - (49/54)2 - (5/54)2 =0.168, where 54 is total samples and 0, 49, 5 are individual class samples
Scikit learn uses CART (Classification and Regression Tree) algorithm which produces only binary trees, whereas ID3 algorithms can produce Decision Trees with nodes that have more than two children nodes.
Depending on max_depth , depths of decision tree are decided.
Decision trees are white box models 	because they are fairly intuitive and their decisions are easy to interpret. Decision Tree provide nice and simple classification rules that can be applied manually. In contrast Random Forest or Neural networks are considered black box models because it is usually hard to explain in simple terms why predictions were made i.e. it is hard to know what actually contributed to this prediction.
Decision tree can also estimate probability that an instance belongs to particular class k. It traverses the tree to find leaf node for this instance and returns ratio of training for that input

CART algo
The idea is simple the algo first splits training set in two subsets using single feature k and a threshold tk. Once it has successfully split training set in two it splits subsets using same logic, recursively. It stops recursing once it reaches maximum depth defined by max_depth hyperparameter or if it cannot find the split that will reduce impurity.
A few other hyperparameter control additional stopping conditions are min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes)
It chooses k and tk by searching for the pair that produces puresest subsets (weighted by their size)
There is a cost function for classification used by CART
CART algo is greedy algo. It greedily searches for an optimum split at top level then repeats the process at each level. A greedy algo produces reasonably good solution but not guaranteed to be optimal solution. Finding optimal tree is known to be NP Complete problem. It requires O(exp(m)) time making problem intractable even for small training sets.

Computational Complexity
Predictions require traversing decision tree from root to leaf.
Decision Trees are approximately balanced so traversing requires O(log2(m)) nodes. And overall prediction complexity is O(log2(m)) independent of number of features. 	
However training algo compares less features if max_features is set
The training complexity of O(n*m log(m))
For small training sets Scikit Learn can speed up training by presorting data (set presort=True)  but this slows down training for large datasets.

Gini impurity or entropy
By default gini impurity measure is used, but we can select entropy impurity measure by setting criterion hyperparameter to “entropy”.
Entropy is 0 if all instances are of same class
Most of the times it does not make difference to use entropy or impurity. It leads to same trees. Gini impurity is slightly faster to compute 
Difference between entropy and impurity is gini impurity tends to isolate most frequent class in its own branch of tree, while entropy tends to produce slighlty more balanced trees.

Regularization Hyperparameters
Decision tree makes few assumptions about training data. If left unconstrained most likely it will overfit. So it is called non parametric model. Because number of parameters is not determined prior to training do model structure is freely to stick closely to data. In other case parametric model (linear model) has a predetermined number of parameters so its degree of freedom is limited. 
To avoid overfitting  we need to restrict Decision tree’s freedom during training. This process is called regularization.
In decision tree we can restrict depth by using max_depth hyperparameter. The default value is 0. Reducing max_depth will regularize model and reduce risk of overfitting.
There are other parameters like : Min_samples_split : Minimum number of samples a leaf node must have before it can be split.
Min_samples_leaf: minimum number of samples a leaf node must have
Min_weight_fraction_leaf: same as min_samples_leaf but expressed as fraction of total number of weighted instances
Max_leaf_nodes: maximum number of leaf nodes
Max_features: maximum number of features that are evaluated for splitting at each node.
Increasing min_* or reducing max_* hyperparameters will regularize the model.

MOONS DATASET

Regression:
The decision tree after training looks similar to classification tree but the main difference is instead of predicting a class in each node it predicts a value
The prediction results in Mean Squared Error 
The predicted value for each region is always average target value of instances in that region.
The algorithm splits each region in a way that makes most training instances as close as possible to predicted value.
The CART algo works almost same way except, instead of trying to split training set in a way that minimizes impurity it now tries to split training set in a way that minimizes MSE
Regression tree also overfits data. But using min_samples_leaf=10 results in much more reasonable model.

Decision trees loves orthogonal decision boundaries(all splits are perpendicular to an axis) which makes them sensitive to training set rotation. 
PCA is used to which results in better orientation of training data.
Decision trees are very sensitive to small variations in training data.
Training algo used by Scikit-Learn is stochastic you may get very different models even on same training data (unless you set random_state hyperparameter)
Random forests can limit this instability by averaging predictions over many trees

Exercises remaining 
