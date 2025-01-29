# **Full Stack Data Science Potential Interview Questios** 📚

## **Statistics++** 📊

### **Linear Algebra & Matrix Operations**
1. How do you calculate mean squared error in a vectorized way?
2. Why can eigenvalue decomposition only be performed on square matrices?
3. Why can eigenvalues of a projection (hat) matrix only be 0 or 1?
4. What are the applications of Singular Value Decomposition (SVD) in Machine Learning?
5. What is the Moore-Penrose inverse, & how is it related to least squares?
6. Why do we perform eigenvalue decomposition of the sample covariance matrix to identify principal components?
7. Why is the determinant of an orthogonal matrix $Q$ either 1 or -1?
8. If you apply a dropout layer with a drop rate of 0.5 to a matrix of ones, you will see that some elements of the output matrix become zero, while the remaining elements are set to 2. Why?
9. There are two random variables, $X$ and $Y$, and the covariance between them is 0.5. Also, the variances are: $Var(X) = 2$ and $Var(Y) = 1$. Two new random variables, $P$ and $Q$, are defined as follows: $P = 2X + 4Y$, $Q = 4X - 2Y$. So, what's the covariance between $P$ and $Q$?
10. What is cosine distance? Is it a distance metric?
11. How do you calculate the distance between two parallel hyperplanes?
12. How do quadratic forms (energy) relate to a matrix's definiteness, and what does this reveal about its behavior?
13. What does the norm of a vector represent?
14. What does the inner product of two vectors represent, and how is it related to the kernel trick?
15. What is Singular Value Decomposition? How is it related to rank one matrices?
16. Do all orthogonal matrices perform rotational transformation preserving norm and orientation?
17. Are factor loadings in an orthogonal factor analysis model unique, or do they change with different rotations?
18. If $A$ is a matrix of any shape with all real elements, how are its singular values related to the eigenvalues of $A^TA$?
19. Let's say there is a function $f(x) = x^TAx$ where $x$ is a column vector of size $n \times 1$ and $A$ is a $(n \times n)$ square matrix. What is the gradient of $f(x)$ with respect to $x$?

### **Statistical Theory & Methods**
1. If $X_1$ and $X_2$ are two normally distributed random variables that jointly follow a bivariate normal distribution, why does the lack of correlation between them imply that they are independent, even though, in general, zero correlation does not imply independence?
2. Why do we divide the sum of squares by $(n - 1)$ instead of $n$ (where $n$ is the sample size) when calculating sample variance?
3. If $X_1$ and $X_2$ are two independent random variables following chi-square distribution with $n_1$ and $n_2$ degrees of freedom, respectively. What distribution does $X_1 + X_2$ follow?
4. What is a log-normal distribution? Why is a random variable following a log-normal distribution always greater than zero?
5. What is the standard error of mean (SEM)?
6. A population is normally distributed, and we want to test if the population mean ($\mu$) is equal to $\mu_0$. The population standard deviation is unknown. We compute the following test statistic, $T$, based on a random sample of $n$ independent observations drawn from the population:

   $
   T = \frac{\bar{X} - \mu_0}{S/\sqrt{n}}
   $
   
   where: $\bar{X}$ is the sample mean, $S$ is the sample standard deviation, $n$ is the sample size. What probability distribution does $T$ follow?
7. The outcome of four coin tosses is [1, 1, 1, 0], where 1 represents getting heads and 0 represents getting tails. Using Maximum Likelihood Estimation (MLE), what is the estimated probability of getting head in a single coin toss?
8. Let $X$, $Y$, and $Z$ be three random variables. If $X$ and $Y$ are independent, does this imply that $X$ and $Y$ are conditionally independent given $Z$?
9. What is Chebyshev's Inequality?
10. If $X$ follows a distribution function $F_X(x)$, and you collect a sample of size $n$ (IID), what distribution does the minimum of the sample follow?
11. What is the difference between the unbiasedness and the consistency of an estimator?
12. What is Jensen's Inequality?

### **Statistical Inference & Testing**
1. What is the coefficient of determination ($R^2$)? Why is it not always a good evaluation metric in multiple linear regression? Alternative?
2. Pearson's correlation is used to estimate the strength and the direction of the linear relationship between two continuous variables. What metric is used to judge the association between two categorical variables?
3. Let's say there are three variables $X$, $Y$, and $Z$. You are interested in estimating the strength of the linear relationship between $X$ and $Y$ while controlling for $Z$. Which correlation coefficient would you calculate?
4. How are $R^2$ and adjusted $R^2$ different?
5. What is the issue of multiple testing?
6. Comparing the means of two groups is common to assess the effect of a treatment. One assumption of the independent t-test is that the variances of the two groups must be equal. Which test can you use to check if the variances of the groups are equal? If they are found to be unequal, which test would you use to compare the means?
7. How would you evaluate the association between job roles and preferred movie genre using the provided data?
8. Normality is one of the key assumptions of one-way ANOVA. If it is violated, which non-parametric test is generally used?
9. How do you test for spurious correlation between two variables controlling for a third variable?
10. A company surveys 20 data scientists to evaluate the association between job satisfaction and work-life balance, both rated from 1 to 5. Is there an association between the two? Sample: $s = [(5, 4), (3, 3), ..., (3, 3)]$

## **Artificial Intelligence** 🤖

### **Deep Learning Fundamentals**
1. What is "Weight Sharing" in the context of neural networks? What are its benefits?
2. If a convolutional layer takes an input image with 10 channels and applies a convolution operation to produce 32 output channels using a $7\times7$ square kernel, how many total learnable parameters (including bias) are there?
3. What is Layer Normalization used for?
4. How does RMS Normalization differ from Layer Normalization?
5. What is the difference between the Gaussian Error Linear Unit (GELU) and the Rectified Linear Unit (ReLU) as activation functions?
6. How is AutoEncoder used for Dimensionality Reduction (onto Latent Space)?
7. What is the loss function used in Variational AutoEncoders?
8. How is a Variational Autoencoder (VAE) trained, and how does its latent space differ from that of an Autoencoder (AE)?
9. In deep learning, we generally use equal-sized batches during training. Why is this the case?

### **Natural Language Processing & LLMs**
1. What is the difference between Lemmatization and Stemming?
2. Given the tokens in alphabetically sorted order with their assigned indices in the vocabulary, what will be the BPE encoding of the word "Rahulization"?
3. What is Nucleus sampling? How is it used to control the text-generation process by LLMs?
4. How would you explain to a layman the contextualized embedding of a word in a sentence that a transformer model learns?
5. What is the attention mechanism?
6. How are knowledge graphs used to improve factual accuracy of LLMs?
7. Why is causal attention used in autoregressive language models such as GPT?
8. How does using multiple attention heads improve the performance of transformers?
9. Does the perplexity of predictions decrease or increase with increasing context size?
10. How is the greedy search decoding strategy different from the beam search decoding strategy?
11. How does temperature scaling influence the diversity of responses generated by a large language model (LLM)?

### **Machine Learning Methods**
1. What is cosine similarity? What does it measure? How is it different from Euclidean distance? In what scenarios is it a better measure for comparison than Euclidean distance?
2. What is Minkowski distance? What are its applications in machine learning?
3. What are threshold-dependent metrics used to evaluate the performance of a binary classifier?
4. What's a simple metric to evaluate a time-series forecasting model?
5. What metric serves as the equivalent of $R^2$ (coefficient of determination) in a logistic regression model?
6. Though multicollinearity doesn't affect the predictive performance of a model, you may still want to address it even if model interpretability is not your goal. Why?
7. Why is L1 regularization (LASSO) used for automated feature selection in linear models?
8. In a logistic regression model, what is the interpretation of the coefficient of a predictor (assuming it is not part of any interaction term)? What does this coefficient represent?
9. How do shrinkage methods address multi-collinearity in linear regression?
10. Why does Gini impurity favor the majority class in classification compared to entropy?

## **Engineering & Implementation** 🛠

### **Data Engineering & Processing**
1. What is the difference between row-oriented and column-oriented data?
2. Labelers A and B have classified a set of 20 images as either "cat" or "dog." Which metric would you use to evaluate the level of concordance between their labels?
3. How can you transform a long-format DataFrame into a wide-format DataFrame in pandas?
4. What is training-serving skew?
5. What are the different types of anomalies that can occur in a time-series (multi-channel) dataset?
6. What is the difference between cross-entropy loss and sparse cross-entropy loss?
7. You have two models trained for multi-class classification, where one class in your dataset is a minority (approximately 5%). Two popular averaging methods, micro and macro, are used to generalize binary evaluation metrics such as precision and recall. What are the differences between the two methods? For the given problem, which method would you choose?

### **Feature Engineering & Selection**
1. What is the difference between standardization and normalization?
2. How can you address class imbalance in a dataset by synthetically generating new samples for the minority class, ensuring the new samples are similar to the original data distribution but not identical duplicates?
3. How does the balanced focal loss function address both class imbalance and the challenge of hard-to-classify instances in machine learning models?
4. Why is Mutual Information score a better criterion for feature selection than rank and linear correlation coefficients?
5. How is Principal Component Analysis used for Multivariate Anomaly Detection?
6. How does condition number and variance inflation factor help detecting multicollinearity?
7. Any model-agnostic way to estimate feature importance?

### **Production & Monitoring**
1. Maintaining an ML model in production can be complicated. You may need to retrain the model, update thresholds, or continuously train the model as new data comes in. The approach you choose largely depends on the type of drift that has occurred. In this context, how are covariate drift and concept drift different?
2. What is a generative model? How is it different from a classification model?
3. How do you solve an over(under)-determined system of linear equations?
4. What is Huber Loss, and why is it preferred over Mean Squared Error (MSE) in the presence of outliers?
5. How does the Jacobian come into play when calculating the gradient of a scalar loss function with matrix parameters?
6. What is the log-sum-exp trick?
7. What is weakly stationary time series?
8. What is Granger Causality?
9. How are Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) different?
10. What is exponentially weighted average (EWA)? How is it used for time-series forecasting?