![](stats.jpg "Stats++")

# Full Stack Data Scientist Potentially Interview Questions

## Stats++

Q1: Why is a Covariance Matrix always positive semi-definite?

Q2: Why can eigenvalue decomposition only be performed on square matrices?

Q3: If $X_1$  and $X_2$​  are two normally distributed random variables that jointly follow a bivariate normal distribution, why does the lack of correlation between them imply that they are independent, even though, in general, zero correlation does not imply independence?

Q4: Why do we divide the sum of squares by $(n - 1)$ instead of $n$ (where $n$ is the sample size) when calculating sample variance?

Q5: If $X_1$ and $X_2$ are two independent random variables following chi-square distribution with $n_1$ and $n_2$ degrees of freedom, respectively. What distribution does $X_1 + X_2$ follow?

Q6: Why is the determinant of an orthogonal matrix ($Q$) either 1 or -1?

Q7: What is Minkowski distance? What are its applications in machine learning?

Q8: Pearson's correlation is used to estimate the strength and the direction of the linear relationship between two continuous variables. What metric is used to judge the association between two categorical variables?

Q9: Let's say there are three variables $X$, $Y$, and $Z$. You are interested in estimating the strength of the linear relationship between $X$ and $Y$ while controlling for $Z$. Which correlation coefficient would you calculate? 🤔

Q10: You might have heard of KL divergence and the KS test. How are they different?

Q11: There are two random variables, $X$ and $Y$, and the covariance between them is $0.5$. Also, the variances are: $Var(X) = 2$ and $Var(Y) = 1$.Two new random variables, $P$ and $Q$, are defined as follows: $P = 2X + 4Y$
$Q = 4X - 2Y$. So, what's the covariance between $P$ and $Q$?

Q12: Comparing the means of two groups is common to assess the effect of a treatment. One assumption of the independent t-test is that the variances of the two groups must be equal.Which test can you use to check if the variances of the groups are equal? 🤔If they are found to be unequal, which test would you use to compare the means? 🧐

Q13: Mean is used to measure the central tendency of a distribution, while variance is used to measure the spread of the distribution.In this context, what is Kurtosis? 🤔 What does it measure? 🧐

Q14: What is a log-normal distribution? Why is a random variable following a log-normal distribution always greater than zero?

Q15: What is the standard error of mean (SEM)?

Q16: A population is normally distributed, and we want to test if the population mean (μ) is equal to μ₀.The population standard deviation is unknown.We compute the following test statistic, T, based on a random sample of n independent observations drawn from the population:T = (X̄ - μ₀) / (S / √n)where:
X̄ is the sample mean, 
S is the sample standard deviation, 
n is the sample size.What probability distribution does T follow? 🧐

Q17: How is the mean squared error of an estimator related to its variance and bias 🤔?

Q18: The outcome of four coin tosses is [1, 1, 1, 0], where 1 represents getting heads and 0 represents getting tails.Using Maximum Likelihood Estimation (MLE), what is the estimated probability of getting head in a single coin toss 🤔?

Q19: Let $X$, $Y$, and $Z$ be three random variables. If $X$ and $Y$ are independent, does this imply that $X$ and $Y$ are conditionally independent given $Z$ 🤔?

Q20: What is a weakly stationary time series?

Q21: What is Mahalanobis distance?

Q22: What is Granger Causality?

Q23: How you test for spurious correlation between two variables controlling for a third variable?

Q24: A company surveys 20 data scientists to evaluate the association between job satisfaction and work-life balance, both rated from 1 to 5. Is there a association between the two?

Sample:
s = [(5, 4), (3, 3), (4, 4), (2, 2), (1, 1), (4, 3), (5, 5), (3, 2), (2, 3), (4, 4), (5, 5), (3, 3), (1, 2), (2, 1), (4, 5), (3, 2), (5, 4), (1, 2), (2, 3), (3, 3)]
(Each tuple's first element is job satisfaction and the second is work-life balance.)

Q25: You are to evaluate the association between job roles and the preferred movie genre using the following data:

data = [('Data Scientist', 'Action', 45)
('Data Scientist', 'Comedy', 30)
('ML Engineer', 'Action', 25)
('ML Engineer', 'Drama', 20)
('GenAI Developer', 'Comedy', 35)
('GenAI Developer', 'Action', 50)
('Data Scientist', 'Drama', 15)
('Data Scientist', 'Comedy', 10)
('ML Engineer', 'Drama', 5)
('GenAI Developer', 'Action', 40)]

Which statistic do you use?

Q26: Normality is one of the key assumptions of one-way ANOVA. If it is violated, which non-parametric test is generally used?

Q27: How are Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) different?

Q28: What is the issue of multiple testing?

Q29: Let's say there is a function $f(x) = x'Ax$ where $x$ is a column vector of size $n$ x $1$ and A is a $(n$ x $n)$ square matrix. What is the gradient of $f(x)$ with respect to $x$?

Q30: If $X$ follows a distribution function $F_X(x)$, and you collect a sample of size $n$ (IID), what distribution does the minimum of the sample follow?

Q31: What is the difference between the unbiasedness and the consistency of an estimator?

Q32: What is Jensen's Inequality?

Q33: What is Chebyshev's Inequality?

Q34: How do you calculate the distance between two parallel hyperplanes?

Q35: How do quadratic forms (energy) relate to a matrix's definiteness, and what does this reveal about its behavior?

Q36: What does the norm of a vector represent?

Q37: What does the inner product of two vectors represent, and how is it related to the kernel trick?

Q38: What is Singular Value Decomposition? How is it related to rank one matrices?

Q39: If $A$ is a matrix of any shape with all real elements, how are its singular values related to the eigenvalues of $A'A$?

![](ai.jpg)
## Artificial Intelligence

Q40: What is Nucleus sampling? How is it used to control the text-generation process by LLMs?

Q41: How would you explain to a layman the contextualized embedding of a word in a sentence that a transformer model learns?

Q42: Large language modelling has gained immense popularity, particularly with the rise of ChatGPT.Two prominent approaches are Masked Language Modelling (e.g., BERT) and Autoregressive Language Modelling (e.g., GPT).What is the key difference between these two modelling approaches 🤔?

Q43: What is Layer Normalization used for?

Q44: How does temperature scaling influence the diversity of responses generated by a large language model (LLM)?

Q45: What is attention mechanism?

Q46: How are knowledge graphs used to improve factual accuracy of LLMs?

Q47: What is the difference between the Gaussian Error Linear Unit (GELU) and the Rectified Linear Unit (ReLU) as activation functions?

Q48: What is the loss function used in Variational AutoEncoders?

Q49: How is a Variational Autoencoder (VAE) trained, and how does its latent space differ from that of an Autoencoder (AE)?

Q50: What is a generative model? How is it different from a classification model?

Q51: Does the perplexity of predictions decrease or increase with increasing context size?

Q52: Why is causal attention used in autoregressive language models such as GPT?

Q53: How does using multiple attention heads improve the performance of transformers?

Q54: How does RMS Normalization differ from Layer Normalization?

Q55: How is the greedy search decoding strategy different from the beam search decoding strategy?

Q56: What are model counterfactuals?If you train a classification model to predict whether to deny credit, and then generate a model counterfactual for a person (to identify the smallest changes in input features needed to reverse the prediction to grant credit), does it imply that these changes cause the credit approval?

![](mle.jpg)
## ML++

Q57: How do you calculate mean squared error in a vectorized way?

Q58: What is cosine similarity? What does it measure? How is it different from Euclidean distance? In what scenarios is it a better measure for comparison than Euclidean distance?

Q59: Why can Eigen values of a projection (hat) matrix only be 0 or 1?

Q60: What are the applications of Singular Value Decomposition (SVD) in Machine Learning?

Q61: What is the Moore-Penrose inverse, & how is it related to least squares?

Q62: What is the coefficient of determination ($R²$)? Why is it not always a good evaluation metric in multiple linear regression? Alternative?

Q63: What is log-sum-exp trick?

Q64: Given the following tokens in alphabetically sorted order with their assigned indices in the vocabulary (assume no other relevant tokens are available):

    •	"i": 1
	•	"khar": 2
	•	"pra": 3
	•	"rah": 4
	•	"Rah": 5
	•	"ul": 6
	•	"tion": 7
	•	"za": 8
What will be the BPE encoding of the word "Rahulization"?

Q65: What is the difference between Lemmatization and Stemming?

Q66: What is the difference between cross-entropy loss and sparse cross-entropy loss?

Q67: If you apply a dropout layer with a drop rate of 0.5 to a matrix of ones, you will see that some elements of the output matrix become zero, while the remaining elements are set to 2. Why?

Q68: What is the difference between standardization and normalization?

Q69: What is "Weight Sharing" in the context of neural networks? What are its benefits?

Q70: What are threshold-dependent metrics used to evaluate the performance of a binary classifier?

Q71: R-squared ($R²$) is used to evaluate the goodness of fit for a regression model. It is interpreted as the proportion of the variance in the response variable that is explained by the regression model's predictors.

Q72: What's a simple metric to evaluate a time-series forecasting model?

Q73: What is the difference between row-oriented and column-oriented data?

Q74: Labelers A and B have classified a set of 20 images as either "cat" or "dog." Which metric would you use to evaluate the level of concordance between their labels?

Q75: What is cosine distance? Is it a distance metric? 🤔

Q76: In deep learning, we generally use equal-sized batches during training. Why is this the case?

Q77: If a convolutional layer takes an input image with 10 channels and applies a convolution operation to produce 32 output channels using a 7x7 square kernel, how many total learnable parameters (including bias) are there 🤔?

Q78: Though multicollinearity doesn't affect the predictive performance of a model, you may still want to address it even if model interpretability is not your goal.Why 🤔?

Q79: Why is L1 regularization (LASSO) used for automated feature selection in linear models 🤔?

Q80: You have a model trained for multi-class classification, where one class in your dataset is a minority (approximately 5%).Two popular averaging methods, micro and macro, are used to generalize binary evaluation metrics such as precision and recall.What are the differences between the two methods? 🤔For the given problem, which method would you choose? 🧐

Q81: You are to model the linear relationship between a non-negative countable response variable and a set of explanatory variables. Which model can you use?

Q82: Maintaining an ML model in production can be complicated. You may need to retrain the model, update thresholds, or continuously train the model as new data comes in. The approach you choose largely depends on the type of drift that has occurred.In this context, how are covariate drift and concept drift different?

Q83: Let's say you have 9 unique tokens from a corpus along with 1 special token.The max context length for your transformer model is 4, and the dense vector representation to be learned is set to 3.Instead of using fixed positional embeddings, you want to use learnable positional embeddings.Given this scenario, how many learnable parameters are in the positional embedding layer 🤔?How many in the token embedding layer 🤷?

Q84: Convert the following data from wide format to long format using Pandas, and then plot a box plot using Seaborn.

Data: 
data = [ 
(100, 85, 80, 95, 78, 92), 
(96, 78, 75, 88, 82, 85), 
(56, 92, 90, 91, 80, 87), 
(89, 88, 82, 94, 85, 90), 
(99, 95, 89, 97, 88, 93) 
]

Columns: StudentID, Linear Algebra, Multivariate Statistics and Classical ML, Foundations of Deep Learning, Chaos Theory and Generative Al, Distributed Computing and Big Data

Q85: What is training-serving skew?

Q86: How are $R^2$ and adjusted $R^2$ different?

Q87: What is exponentially weighted average (EWA)? How is it used for time-series forecasting?

Q88: How condition number and variance inflation factor help detecting multicollinearity?

Q89: Any method to detect multivariate anomalies if the underlying data distribution is Gaussian (more or less)?

Q90: Suppose you want to convert a continuous score variable into a categorical class variable, with the number of classes predetermined based on your judgment or an underlying phenomenon. Which method can you use?

Q91: How are PCA and Feature Agglomeration different?

Q92: What metric serves as the equivalent of $R^2$ (coefficient of determination) in a logistic regression model?

Q93: Any model-agnostic way to estimate feature importance?

Q94: How would you estimate the correlation between a binary categorical variable and a continuous variable?

Q95: How does the balanced focal loss function address both class imbalance and the challenge of hard-to-classify instances in machine learning models?

Q96: Why is Mutual Information score a better criterion for feature selection than rank and linear correlation coefficients?

Q97: What are the different types of anomalies that can occur in a time-series (multi-channel) dataset?

Q98: How can you address class imbalance in a dataset by synthetically generating new samples for the minority class, ensuring the new samples are similar to the original data distribution but not identical duplicates?

Q99: How can you transform a long-format DataFrame into a wide-format DataFrame in pandas?

Q100: How is Principal Component Analysis used for Multivariate Anomaly Detection?

Q101: How is AutoEncoder used for Dimensionality Reduction (onto Latent Space)? 

Q102: How do you solve an over(under)-determined system of linear equations?

Q103: What is Huber Loss, and why is it preferred over Mean Squared Error (MSE) in the presence of outliers?

Q104: How does the Jacobian come into play when calculating the gradient of a scalar loss function with matrix parameters?

Q105: In a logistic regression model, what is the interpretation of the coefficient of a predictor (assuming it is not part of any interaction term)? What does this coefficient represent?

Q106: How do shrinkage methods address multi-collinearity in linear regression?

Q107: How are single linkage and complete linkage different in the context of Agglomerative clustering?

Q108: Why do we perform eigenvalue decomposition of the sample covariance matrix to identify principal components?

Q109: Do all orthogonal matrices perform rotational transformation preserving norm and orientation?

Q110: Are factor loadings in an orthogonal factor analysis model unique, or do they change with different rotations?