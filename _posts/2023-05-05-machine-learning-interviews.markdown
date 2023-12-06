---
layout: post
title: Machine Learning Interviews
description: Solutions to questions from Chip Huyen's Machine Learning Interviews book
image: /assets/images/machine-learning-interviews/splash.png
authors: [tlyleung]
permalink: machine-learning-interviews
---

> **Author's Note**
> 
> This post contains solutions to the questions in Chip Huyen's book [Machine Learning Interviews](https://huyenchip.com/ml-interviews-book/contents/part-ii.-questions.html). An initial pass was generated using ChatGPT with some answers requiring significant rewrites. Additional help was obtained from the book's [official Discord channel](https://discord.gg/XjDNDSEYjh). Please [contact me](/authors/tlyleung) for corrections/omissions.

# Mathematics

## Algebra

### Vectors

1. **Dot product**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the geometric interpretation of the dot product of two vectors?**

        The dot product of two vectors is a measure of the magnitude of one vector projected along the direction of the other. It's calculated as the product of the magnitudes of the two vectors and the cosine of the angle between them: $$\vert a \vert \vert b \vert \cos \theta$$. When the vectors are perpendicular, the dot product is 0 because the cosine of a 90° angle is 0. When they are parallel, the dot product is the product of their magnitudes because the cosine of 0° is 1.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Given a vector $$u$$, find vector $$v$$ of unit length such that the dot product of $$u$$ and $$v$$ is maximum.**

        The dot product of two vectors is maximized when the vectors are parallel. So, to find a vector $$v$$ of unit length such that the dot product of $$v$$ and a given vector $$u$$ is maximum, we should take $$v$$ as the unit vector in the direction of $$u$$. That is, $$v = \frac{u}{\Vert u \Vert}$$, where $$\Vert u \Vert$$ is the norm (or length) of $$u$$.

2. **Outer product**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Given two vectors $$a = [3, 2, 1]$$ and $$b = [-1, 0, 1]$$. Calculate the outer product $$a^Tb$$.**
    
        $$a^Tb = \begin{bmatrix}-3 & 0 & 3\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix}$$

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Give an example of how the outer product can be useful in ML.**

        The outer product can be useful in machine learning in several ways. One example is in the computation of the covariance matrix in statistics and machine learning, which is essentially a scaled outer product of a data matrix with itself. This matrix provides insight into the relationships between different dimensions or features in the dataset.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **What does it mean for two vectors to be linearly independent?**

    Two vectors are linearly independent if neither can be expressed as a linear combination of the other. In other words, no scalar multiplication or addition of one vector will result in the other vector.

4. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given two sets of vectors $$A = {a_1, a_2, a_3, ..., a_n}$$ and $$B = {b_1, b_2, b_3, ... , b_m}$$. How do you check that they share the same basis?**

    To check if they share the same basis, first, ensure that both sets of vectors are linearly independent. If either set isn't, it doesn't form a basis. Second, take the union of the sets $$A$$ and $$B$$ and check if this set is still linearly independent. If it is, then $$A$$ and $$B$$ share the same basis.

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given $$n$$ vectors, each of $$d$$ dimensions. What is the dimension of their span?**

    The dimension of the span of $$n$$ vectors each of $$d$$ dimensions is at most the smaller of $$n$$ and $$d$$. It's the number of linearly independent vectors in the set. It could be less than $$n$$ or $$d$$ if some vectors are linearly dependent.

6. **Norms and metrics**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What's a norm? What is $$L_0, L_1, L_2, L_{norm}$$?**

        A norm is a function that assigns a strictly positive length or size to all vectors in a vector space, with the exception of the zero vector which is assigned a length of zero. It follows certain rules such as triangle inequality, scalar multiplication, and that it's positive definite.
        - $$L_0$$ norm is not actually a norm, but often used to denote the number of non-zero elements in a vector.
        - $$L_1$$ norm, also known as Manhattan or taxicab norm, is the sum of the absolute values of the elements.
        - $$L_2$$ norm, also known as Euclidean norm, is the square root of the sum of the squares of the elements.
        - $$L_p$$ norm, for any real number $$p \geq 1$$, is the $$p$$-th root of the sum of the elements to the power of $$p$$.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?**

        Norms and metrics are both ways of measuring 'distance'. A norm is a kind of metric, but not all metrics are norms. A norm can be used to define a metric, called the induced metric, where the distance between two vectors is the norm of their difference. However, not all metrics can be used to define a norm as they may not obey the homogeneity and triangle inequality properties that a norm does. For example, the discrete metric (which gives the distance between two distinct points as always 1) cannot define a norm.

### Matrices

1. <span class="badge text-bg-secondary bg-success">Easy</span> **Why do we say that matrices are linear transformations?**

    Matrices are said to be linear transformations because they represent operations that can be applied to vectors that preserve the operations of vector addition and scalar multiplication, key properties of linear systems. When we multiply a matrix with a vector, we're effectively transforming the vector into a new vector in the vector space.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?**

    The inverse of a matrix $$A$$ is a matrix $$A^{-1}$$ such that when A is multiplied with $$A^{-1}$$, the result is the identity matrix $$I$$. Not all matrices have an inverse; only square matrices that are non-singular (i.e., their determinant is not zero) have an inverse. The inverse of a matrix, when it exists, is always unique.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **What does the determinant of a matrix represent?**

    The determinant of a matrix represents the scale change of an area or volume when the corresponding linear transformation is applied. In other words, the determinant gives the factor by which the linear transformation changes the volume in the input space. It also provides information about whether the matrix is invertible (non-zero determinant) or not (zero determinant).

4. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens to the determinant of a matrix if we multiply one of its rows by a scalar $$t \times R$$?**

    If we multiply one of the rows of a matrix by a scalar $$t$$, the determinant of the matrix is also multiplied by $$t$$. This is because the determinant of a matrix represents a volume, and multiplying a dimension by $$t$$ scales the volume by the same factor.

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **A $$4 \times 4$$ matrix has four eigenvalues $$3, 3, 2, -1$$. What can we say about the trace and the determinant of this matrix?**

    The trace of a matrix is the sum of its eigenvalues, and the determinant of a matrix is the product of its eigenvalues. Therefore, given four eigenvalues 3, 3, 2, and -1, the trace of the matrix is 3 + 3 + 2 - 1 = 7 and the determinant is 3 * 3 * 2 * -1 = -18.

6. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given the following matrix:**

    $$\begin{bmatrix} 1 & 4 & -2 \\ -1 & 3 & 2 \\ 3 & 5 & -6 \end{bmatrix}$$
    
    **Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?**

    Since none of the rows/columns can be obtained by a linear combination of each other, they're linearly independent, meaning that the determinant is non-zero.
    
7. <span class="badge text-bg-secondary bg-warning">Medium</span> **What’s the difference between the covariance matrix $$A^TA$$ and the Gram matrix $$AA^T$$?**

    The covariance matrix $$A^TA$$ is a measure of how each pair of elements in $$A$$ varies together. The Gram matrix $$AA^T$$ is a measure of how each pair of rows in $$A$$ are similar to each other (the dot product of each pair of rows).

8. **Given $$A \in R^{n \times m}$$ and $$b \in R^n$$**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **Find $$x$$ such that: $$Ax = b$$.**

        The solution to the equation $$Ax = b$$ is given by $$x = A^{-1}b$$, assuming that $$A$$ is invertible.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **When does this have a unique solution?**

        This equation has a unique solution when $$A$$ is square and invertible (its determinant is not zero).

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why is it when $$A$$ has more columns than rows, $$Ax = b$$ has multiple solutions?**

        If $$A$$ has more columns than rows, the system is underdetermined, which means there are infinite solutions. This happens because there are more variables than equations.

    4. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given a matrix $$A$$ with no inverse. How would you solve the equation $$Ax = b$$? What is the pseudoinverse and how to calculate it?**

        When $$A$$ is not invertible, the equation $$Ax = b$$ can be solved using a pseudoinverse, denoted as $$A^+$$. It provides the best fit solution to the system of equations. The pseudoinverse of A can be calculated using the Singular Value Decomposition (SVD) method.

9. **Derivative is the backbone of gradient descent.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What does derivative represent?**

        The derivative of a function at a given point represents the rate at which the function is changing at that point. In other words, it's a measure of how a small change in the input will affect the output.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What’s the difference between derivative, gradient, and Jacobian?**

        The derivative is a scalar that represents the rate of change of a function at a point. The gradient is a vector that points in the direction of the greatest rate of increase of the function. The Jacobian is a matrix that represents the first derivatives of a vector-valued function.

10. <span class="badge text-bg-secondary bg-danger">Hard</span> **Say we have the weights $$w \in R^{d \times m}$$ and a mini-batch $$x$$ of $$n$$ elements, each element is of the shape $$1 \times d$$ so that $$x \in R^{n \times d}$$. We have the output $$y = f(x; w) = xw$$. What’s the dimension of the Jacobian $$\frac{\delta y}{\delta x}$$?**

    If $$y = f(x; w) = xw$$, where $$w$$ is in $$R^{d \times m}$$ and $$x$$ is in $$R^{n \times d}$$, the output $$y$$ is in $$R^{n \times m}$$. The Jacobian of $$y$$ with respect to $$x$$, $$J = ∂y/∂x$$, is a third-order tensor of shape $$(n, m, d)$$.

11. <span class="badge text-bg-secondary bg-danger">Hard</span> **Given a very large symmetric matrix A that doesn’t fit in memory, say $$A \in R^{1M \times 1M}$$ and a function $$f$$ that can quickly compute $$f(x) = Ax$$ for $$x \in R^{1M}$$. Find the unit vector $$x$$ so that $$x^TAx$$ is minimal.**

    This can be framed as an optimization problem to minimize the function $$g(x) = x^TAx$$. The derivative of $$g(x)$$ is $$2Ax$$. Given the function $$f(x) = Ax$$, we can use $$f(x)$$ to compute the gradient $$g'(x) = 2f(x)$$ and perform gradient descent, normalizing x at each step to keep it a unit vector. The minimization of $$x^TAx$$ corresponds to finding the eigenvector of $$A$$ corresponding to the smallest eigenvalue, a problem typically solved with iterative methods like the power method when $$A$$ is large and sparse. [TODO]

### Dimensionality Reduction

1. <span class="badge text-bg-secondary bg-success">Easy</span> **Why do we need dimensionality reduction?**

    We need dimensionality reduction for several reasons:
    - To minimize the computational resources required to process, store, and analyze data.
    - To make data visualization more feasible and meaningful when we have high-dimensional data.
    - To help in mitigating the "curse of dimensionality" issue, which can cause overfitting in machine learning models.
    - To remove redundancy and noise in the data, as high-dimensional data often contain irrelevant features.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **Eigendecomposition is a common factorization technique used for dimensionality reduction. Is the eigendecomposition of a matrix always unique?**

    The eigendecomposition of a matrix is not always unique. If a matrix has repeated eigenvalues, then different sets of eigenvectors can be chosen as the basis of the corresponding eigenspaces. However, if all eigenvalues are distinct, the eigendecomposition is unique up to the order of the eigenpairs.

3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Name some applications of eigenvalues and eigenvectors.**

    Eigenvalues and eigenvectors have numerous applications across fields. Here are a few in the context of data analysis and machine learning:
    - Principal Component Analysis (PCA) uses eigenvalues and eigenvectors for dimensionality reduction.
    - They are used in understanding linear transformations and systems of linear equations.
    - They are used in the Google PageRank algorithm, which ranks websites in its search engine results.
    - They are useful in the study of dynamic systems, like population growth models.

4. <span class="badge text-bg-secondary bg-warning">Medium</span> **We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range 0--1 and one is in the range 10--1000. Will PCA work on this dataset?**

    Yes, PCA can be applied to the dataset. However, it is generally a good practice to standardize the data before applying PCA because PCA is a variance maximizing exercise. It projects your original data onto directions which maximize the variance. If one feature has a very broad range (10--1000), PCA might determine that feature is the principal component, even though it may just be an artifact of the scale of measurement.

5. <span class="badge text-bg-secondary bg-danger">Hard</span> **Eigendecomposition**

    1. **Under what conditions can one apply eigendecomposition? What about SVD?**

        Eigendecomposition can be applied when the matrix is square (number of rows equals number of columns). If the matrix is also symmetric (equal to its transpose), all its eigenvalues are real and it can be orthogonally diagonalized. SVD, or Singular Value Decomposition, can be applied to any $$m \times n$$ matrix and does not require the matrix to be square or symmetric.

    2. **What is the relationship between SVD and eigendecomposition?**

        SVD and eigendecomposition are both methods of factorizing a matrix. If matrix $$A$$ is square, we can express $$A$$ as $$PDP^{-1}$$ using eigendecomposition. But if $$A$$ is not square, we have to use SVD where $$A = U \sum V^T$$.

    3. **What’s the relationship between PCA and SVD?**

        Principal Component Analysis is a technique for dimensionality reduction, while Singular Value Decomposition is a method for factorizing a matrix. The principal components from PCA are equivalent to the left singular vectors from the SVD of the data matrix.

6. <span class="badge text-bg-secondary bg-danger">Hard</span> **How does t-SNE (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?**

    t-SNE (T-distributed Stochastic Neighbor Embedding) is a machine learning algorithm for visualization based on stochastic neighbor embedding originally developed by Geoffrey Hinton and his students. It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. It works by minimizing the divergence between two distributions: a distribution that measures pairwise similarities of the input objects and a distribution that measures pairwise similarities of the corresponding low-dimensional points in the embedding. We need t-SNE because it is particularly good at preserving local structure in the data, making it excellent for visual exploration of complex datasets.

### Calculus and Convex Optimization

1. **Differentiable functions**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What does it mean when a function is differentiable?**

        A function is differentiable at a point when it has a derivative at that point. This means that the function is smooth and doesn't have any breaks, bends, or corners at that point. It also implies that the function is continuous at that point.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Give an example of when a function doesn’t have a derivative at a point.**

        An example would be the function $$f(x) = \lvert x \rvert$$ at $$x=0$$. The function has a corner at this point, so it is not differentiable there.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren’t differentiable?**

        An example of a non-differentiable function used in machine learning is the ReLU (Rectified Linear Unit) activation function. It's not differentiable at zero. However, in practice, we can use sub-gradient or define the gradient at zero to be 0 or 1, which allows us to use it with backpropagation.

2. **Convexity**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What does it mean for a function to be convex or concave?**

        A function is convex if the line segment between any two points on the function's graph does not fall below the graph. A function is concave if the line segment between any two points on the graph does not rise above the graph.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why is convexity desirable in an optimization problem?**

        Convexity is desirable in an optimization problem because if a function is convex, any local minimum is also a global minimum. It guarantees that if we find a solution that is the best in its local neighborhood, then it is the best solution over the entire domain.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Show that the cross-entropy loss function is convex.**

        Cross entropy loss function is given by $$L(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$. It can be shown that this function is convex by calculating the second derivative and showing that it's always non-negative. [TODO]

3. **Given a logistic discriminant classifier $$p(y=1\|x) = \sigma (w^Tx)$$, where the sigmoid function is given by $$\sigma(z) = (1 + \exp(-z))^{-1}$$. The logistic loss for a training sample $$x_i$$ with class label $$y_i$$ is given by $$L(y_i, x_i;w) = -\log p(y_i\|x_i)$$.** [TODO]

    1. **Show that $$p(y=-1\|x) = \sigma(-w^Tx)$$.**

        By definition, $$p(y=-1\|x) = 1 - p(y=1\|x) = 1 - \sigma(w^Tx) = \sigma(-w^Tx)$$.

    2. **Show that $$\Delta_wL(y_i, x_i; w) = -y_i(1-p(y_i\|x_i))x_i$$**.
    
        $$L(y_i, x_i; w) = -y_i log(p(y_i\|x_i)) - (1-y_i) log(1-p(y_i\|x_i))$$$$L(y_i, x_i; w) = -y_i \log(p(y_i\|x_i)) - (1-y_i) \log(1-p(y_i\|x_i))$$, from this the derivative can be computed to get $$\Delta wL(y_i, x_i; w) = -y_i(1-p(y_i\|x_i))x_i$$.

    3. **Show that $$\Delta_wL(y_i, x_i; w)$$ is convex.**

        To show that $$\Delta wL(y_i, x_i; w)$$ is convex, we can compute the Hessian of $$L$$, and show that it is positive semi-definite.

4. **Most ML algorithms we use nowadays use first-order derivatives (gradients) to construct the next training iteration.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How can we use second-order derivatives for training models?**

        Second order derivatives can be used for training models to accelerate convergence. Specifically, using the second derivative or the Hessian matrix, we can have a better approximation of the function we are trying to optimize, leading to faster and potentially more accurate convergence.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Pros and cons of second-order optimization.**

        Pros: Faster convergence, less sensitivity to learning rate.
        
        Cons: More computational resources needed, not suitable for high dimensional problems.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why don’t we see more second-order optimization in practice?**

        We don't see more second-order optimization in practice because these methods often require computation and storage of the Hessian matrix, which can be expensive or even infeasible for large-scale problems.

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **How can we use the Hessian (second derivative matrix) to test for critical points?**

    The Hessian matrix can be used to test for critical points by looking at the eigenvalues of the Hessian. If all the eigenvalues of the Hessian are positive, then the function reaches a local minimum at that point. If all the eigenvalues are negative, the function reaches a local maximum. If there are both positive and negative eigenvalues, the point is a saddle point.

6. <span class="badge text-bg-secondary bg-success">Easy</span> **Jensen’s inequality forms the basis for many algorithms for probabilistic inference, including Expectation-Maximization and variational inference. Explain what Jensen’s inequality is.**

    Jensen’s inequality states that the convex transformation of an expectation is always greater than or equal to the expectation of the convex transformation. Formally, if $$\phi$$ is a convex function, then $$E[\phi(x)] \geq \phi(E[x])$$.

7. <span class="badge text-bg-secondary bg-success">Easy</span> **Explain the chain rule.**

    The chain rule is a basic derivative rule that states that the derivative of a composition of functions is the product of the derivative of the inner function and the derivative of the outer function. If we have a function composed of two functions like $$f(g(x))$$, the derivative of this function with respect to x would be $$f'(g(x))*g'(x)$$.

8. <span class="badge text-bg-secondary bg-warning">Medium</span> **Let $$x \in R_n$$, $$L = \textrm{crossentropy}(\textrm{softmax}(x), y)$$ in which $$y$$ is a one-hot vector. Take the derivative of $$L$$ with respect to $$x$$.**

    For cross-entropy loss and softmax, the derivative simplifies quite nicely. It turns out that if we let $$p=\textrm{softmax}(x)$$ and $$L = \textrm{crossentropy}(p,y)$$, then $$\frac{\partial L}{\partial x} = p - y$$. [TODO]

9. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given the function $$f(x, y) = 4x^2 - y$$ with the constraint $$x^2 + y^2 =1$$. Find the function’s maximum and minimum values.**

    For this constrained optimization problem, we can use the method of Lagrange multipliers. The Lagrangian is given by $$L(x, y, λ) = 4x^2 - y + λ(1 - x^2 - y^2)$$. Taking the partial derivatives and setting them to zero gives the system of equations that can be solved to find the maximum and minimum values. [TODO]

## Probability and Statistics

### Probability

1. <span class="badge text-bg-secondary bg-success">Easy</span> **Given a uniform random variable $$X$$ in the range of $$[0, 1]$$ inclusively. What’s the probability that $$X=0.5$$?**

    The probability that a continuous uniform random variable takes on any specific value, including 0.5, is 0, as there are infinitely many possibilities in the continuous range.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **Can the values of PDF be greater than 1? If so, how do we interpret PDF?**

    Yes, values of the Probability Density Function (PDF) can be greater than 1. The PDF is interpreted as the relative likelihood of the random variable being equal to a value. The area under the PDF over an interval gives the probability that the random variable takes a value in that interval.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the difference between multivariate distribution and multimodal distribution?**

    A multivariate distribution involves more than one random variable (like a bivariate or trivariate distribution), while a multimodal distribution is a distribution with more than one peak or mode.

4. <span class="badge text-bg-secondary bg-success">Easy</span> **What does it mean for two variables to be independent?**

    Two variables are independent if the outcome of one variable does not affect the outcome of the other. In terms of probability, they're independent if the probability of their intersection equals the product of their probabilities: $$P(A \cap B) = P(A)P(B)$$.

5. <span class="badge text-bg-secondary bg-success">Easy</span> **It’s a common practice to assume an unknown variable to be of the normal distribution. Why is that?**

    The normal distribution is often assumed in statistics and machine learning due to the Central Limit Theorem, which states that the sum of many independent random variables, regardless of their distribution, tends to follow a normal distribution.

6. <span class="badge text-bg-secondary bg-success">Easy</span> **How would you turn a probabilistic model into a deterministic model?**

    A probabilistic model can be turned into a deterministic model by choosing the most probable output for each input, i.e., the mode of the conditional distribution $$P(\textrm{output} \vert \textrm{input})$$.

7. <span class="badge text-bg-secondary bg-danger">Hard</span> **Is it possible to transform non-normal variables into normal variables? How?**

    Yes, it is possible to transform non-normal variables into normal variables using techniques such as the Box-Cox transformation, logarithmic transformation, or the Yeo-Johnson transformation.

8. <span class="badge text-bg-secondary bg-warning">Medium</span> **When is the t-distribution useful?**

    The t-distribution is useful when the sample size is small, or when the standard deviation of the population is unknown. It's often used in hypothesis testing and constructing confidence intervals.

9. **Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **What's the probability that it will crash in the next month?**

    If the storage system crashed 5 times in a year, then it crashes on average about once every 2.4 months. Therefore, the probability that it will crash in the next month is approximately 0.417. [TODO]

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What's the probability that it will crash at any given moment?**

    Assuming the crashes are uniformly distributed throughout the year, the probability that it will crash at any given moment is 5/(365*24*60*60), as there are approximately 31,536,000 seconds in a year. [TODO]

10. <span class="badge text-bg-secondary bg-warning">Medium</span> **Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently, what's the probability that the next 20 predictions are all correct?**

    If the classifier has been wrong 10 times out of 100, then it has a 90% accuracy rate. The probability that the next 20 predictions are all correct is (0.9)^20 ≈ 0.122.

11. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given two random variables $$X$$ and $$Y$$. We have the values $$P(X \mid Y)$$ and $$P(Y)$$ for all values of $$X$$ and $$Y$$. How would you calculate $$P(X)$$?**

    The probability of $$X$$ can be calculated by marginalizing over Y, that is, summing or integrating $$P(X \mid Y)P(Y)$$ over all values of $$Y$$.

12. <span class="badge text-bg-secondary bg-warning">Medium</span> **You know that your colleague Jason has two children and one of them is a boy. What’s the probability that Jason has two sons? Hint: it’s not $$\frac{1}{2}$$.**

    The possible child pairs are {Boy, Boy}, {Boy, Girl}, {Girl, Boy}. Since we know one of the children is a boy, the possible pairs become {Boy, Boy}, {Boy, Girl}, {Girl, Boy}. Hence, the probability that Jason has two sons is $$\frac{1}{3}$$.

13. **There are only two electronic chip manufacturers: A and B, both manufacture the same amount of chips. A makes defective chips with a probability of 30%, while B makes defective chips with a probability of 70%.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **If you randomly pick a chip from the store, what is the probability that it is defective?**

        Since A and B manufacture the same amount of chips, and they are defective with a probability of 30% and 70% respectively, the probability that a chip picked at random is defective is 0.5*(0.3 + 0.7) = 0.5.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Suppose you now get two chips coming from the same company, but you don’t know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?**

        If the first chip is functioning, the probability that the second chip is also good depends on the same manufacturer. If it's from A, the probability is 0.7, and if it's from B, the probability is 0.3. Given that the probability of the first chip follows the same probability, we have $$0.7 \times 0.7 + 0.3 \times 0.3 = 0.58$$.

14. **There’s a rare disease that only 1 in 10000 people get. Scientists have developed a test to diagnose the disease with the false positive rate and false negative rate of 1%.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Given a person is diagnosed positive, what’s the probability that this person actually has the disease?**
    
        Let $$A$$ be the event that a person has the disease and $$B$$ the event that a person is diagnosed positive. From the information given, $$P(A) = 0.0001$$,  $$P(B \mid A') = 0.01$$ and $$P(B' \mid A) = 0.01$$.

        Consequently, using Bayes’ Theorem:

        $$P(A\mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$
        $$P(A') = 0.9999$$
        $$P(B) = P(B \mid A')P(A') = 0.01 * 0.9999 = 0.009999$$
        $$P(B') = 0.990001$$
        $$P(B|A) = \frac{P(B)}{P(A)}$$
      
        Using Bayes’ Theorem: $$P(A\mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

        [TODO]
      
    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What’s the probability that a person has the disease if two independent tests both come back positive?**

        [TODO]

15. <span class="badge text-bg-secondary bg-warning">Medium</span> **A dating site allows users to select 10 out of 50 adjectives to describe themselves. Two users are said to match if they share at least 5 adjectives. If Jack and Jill randomly pick adjectives, what is the probability that they match?**

    The number of ways Jack and Jill can each select 10 adjectives out of 50 is $$\binom{50}{10}$$, and the number of ways they can select at least 5 of the same adjectives is the sum of $$\binom{10}{i} \binom{40}{10-i}$$ for $$i$$ from 5 to 10. The desired probability is the latter divided by the square of the former.

16. <span class="badge text-bg-secondary bg-warning">Medium</span> **Consider a person A whose sex we don’t know. We know that for the general human height, there are two distributions: the height of males follows $$h_m = N(\mu_m, \sigma_m^2)$$ and the height of females follows $$h_j = N(\mu_j, \sigma_j^2)$$ . Derive a probability density function to describe A’s height.**

    The height of A follows a mixture of two normal distributions. If the probability that A is male is $$p$$, the density function is $$p h_m + (1-p) h_j$$.

17. <span class="badge text-bg-secondary bg-danger">Hard</span> **There are three weather apps, each the probability of being wrong ⅓ of the time. What’s the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it’s going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy 50% of the time? Consider the cases where all the apps are independent and where they are dependent.**

    If $$A$$ is the event that it will be foggy and $$B$$ is the event that all the apps predict it’s going to be foggy, using Bayes’ Theorem:
    $$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)} = \frac{\frac{2}{3}^3 \frac{1}{2}}{\frac{2}{3}^3\frac{1}{2} + \frac{1}{3}^3\frac{1}{2}} = \frac{\frac{8}{27}}{\frac{9}{27}} = \frac{8}{9}$$

    If the apps are dependent, the calculation will be different and the probability will be lower.

18. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given $$n$$ samples from a uniform distribution $$[0, d]$$. How do you estimate $$d$$? (Also known as the German tank problem)**

    The maximum of the $$n$$ samples is a biased estimator of $$d$$. A better estimator is $$M + M/n - 1$$, where $$M$$ is the maximum of the $$n$$ samples. [TODO]

19. <span class="badge text-bg-secondary bg-warning">Medium</span> **You’re drawing from a random variable that is normally distributed, $$X \sim N(0,1)$$, once per day. What is the expected number of days that it takes to draw a value that’s higher than 0.5?**

    This can be solved by considering the cumulative distribution function (CDF) of the normal distribution. Since $$P(X < 0.5) = \Phi(0.5) \approx 0.6915$$, the expected number of days, using the geometric distribution is  is $$\frac{1}{1 - 0.6915} \approx 3.24 \textrm{ days}$$.

20. <span class="badge text-bg-secondary bg-warning">Medium</span> **You’re part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than 50%?**

    This problem is known as the [Birthday Paradox](https://en.wikipedia.org/wiki/Birthday_problem) and with a class of 23 students, the probability of two students sharing the same birthday is greater than 50%.

21. <span class="badge text-bg-secondary bg-danger">Hard</span> **You decide to fly to Vegas for a weekend. You pick a table that doesn’t have a bet limit, and for each game, you have the probability $$p$$ of winning, which doubles your bet, and $$1-p$$ of losing your bet. Assume that you have unlimited money (e.g. you bought Bitcoin when it was 10 cents), is there a betting strategy that has a guaranteed positive payout, regardless of the value of $$p$$?**

    The Martingale betting strategy of doubling the bet after each loss guarantees a net gain of one initial bet after each win, regardless of the value of $$p$$, provided you have unlimited money and there's no bet limit.

22. <span class="badge text-bg-secondary bg-danger">Hard</span> **Given a fair coin, what’s the number of flips you have to do to get two consecutive heads?**

    Let the expected number of coin flips be $$x$$. The case analysis goes as follows:
    1. If the first flip is a tails, then we have wasted one flip. The probability of this event is $$\frac{1}{2}$$ and the total number of flips required is $$x+1$$.
    2. If the first flip is a heads and second flip is a tails, then we have wasted two flips. The probability of this event is $$\frac{1}{4}$$ and the total number of flips required is $$x+2$$.
    3. If the first flip is a heads and second flip is also heads, then we are done. The probability 
    of this event is $$\frac{1}{4}$$ and the total number of flips required is 2.
    
    Summing these, we get: $$x = \frac{1}{2}(x+1) + \frac{1}{4}(x+2) + \frac{1}{4}(2) = 6$$.
    

23. <span class="badge text-bg-secondary bg-danger">Hard</span> **In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon? Hint: The law of small numbers.**

    This phenomenon can be explained by the *Law of Small Numbers*, which is a bias that occurs when a small sample size is used. With small population sizes, there is a higher chance of seeing extreme results due to random variance.

24. <span class="badge text-bg-secondary bg-warning">Medium</span> **Derive the maximum likelihood estimator of an exponential distribution.**

    The maximum likelihood estimator for the parameter $$\lambda$$ of an exponential distribution with density function $$f(x \mid \lambda) = \lambda e^{-\lambda x}$$ for $$x \geq 0$$ is $$\frac{1}{\textrm{sample mean}}$$. [TODO]

### Statistics

1. <span class="badge text-bg-secondary bg-success">Easy</span> **Explain frequentist vs. Bayesian statistics.**

    Frequentist statistics and Bayesian statistics are two different schools of thought in the field of statistical inference. Frequentist statistics interpret probability as the long-run frequency of events. Bayesian statistics, on the other hand, interpret probability as a degree of belief. In other words, Bayesian statistics allows for incorporating prior knowledge and updating this knowledge as new data is observed.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **Given the array $$[1, 5, 3, 2, 4, 4]$$, find its mean, median, variance, and standard deviation.**

    - Mean: $$\mu = \frac{1+5+3+2+4+4}{6} = \frac{19}{6} \approx 3.167$$
    - Median: $$3.5$$
    - Variance: $$\sigma^2 = \frac{1}{N} \sum ((1-\mu)^2, (5-\mu)^2, (3-\mu)^2, (2-\mu)^2, (4-\mu)^2, (4-\mu)^2) =  1.97$$ 
    - Standard deviation: $$\sigma = \sqrt{Var(x)} \approx \sqrt{1.97} = 1.4$$

3. <span class="badge text-bg-secondary bg-warning">Medium</span> **When should we use median instead of mean? When should we use mean instead of median?**

    The median is more robust to outliers and skewed data and is therefore a better measure of central tendency when our data is not symmetric. The mean is sensitive to extreme values but gives us a mathematical handle on the data due to its nice properties.

4. <span class="badge text-bg-secondary bg-warning">Medium</span> **What is a moment of function? Explain the meanings of the zeroth to fourth moments.**

    Moments are summary measures that describe properties of a distribution.

    - The zeroth moment is the total probability which is 1 for a probability distribution.
    - The first moment is the mean.
    - The second moment is the variance, which measures spread. The third moment is skewness, which measures asymmetry
    - The fourth moment is kurtosis, which measures the heaviness of the tail ends of a distribution.

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **Are independence and zero covariance the same? Give a counterexample if not.**

    Independence and zero covariance are not the same. Independence is a stronger condition than zero covariance. Zero covariance implies that two variables are uncorrelated, but they can still be dependent. For example, if $$X$$ is a random variable and $$Y = X^2$$, then $$Cov(X,Y) = 0$$ but $$X$$ and $$Y$$ are clearly not independent.

6. <span class="badge text-bg-secondary bg-success">Easy</span> **Suppose that you take 100 random newborn puppies and determine that the average weight is 1 pound with the population standard deviation of 0.12 pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the 95% confidence interval for the average weight of all newborn puppies.**

    $$0.95 = \frac{x - \mu}{\sigma}$$
    $$x = 0.95 \times 0.12 + 1 = 1.114$$

7. <span class="badge text-bg-secondary bg-warning">Medium</span> **Suppose that we examine 100 newborn puppies and the 95% confidence interval for their average weight is $$[0.9, 1.1]$$ pounds. Which of the following statements is true?**

    1. ~~Given a random newborn puppy, its weight has a 95% chance of being between 0.9 and 1.1 pounds.~~
    2. ~~If we examine another 100 newborn puppies, their mean has a 95% chance of being in that interval.~~
    3. We're 95% confident that this interval captured the true mean weight.

8. <span class="badge text-bg-secondary bg-danger">Hard</span> **Suppose we have a random variable $$X$$ supported on $$[0, 1]$$ from which we can draw samples. How can we come up with an unbiased estimate of the median of $$X$$?**

    To estimate the median, you can take many samples from X, sort them and pick the middle one. This is an unbiased estimate because the median is the middle point of a distribution, and this procedure is exactly attempting to capture that.

9. <span class="badge text-bg-secondary bg-danger">Hard</span> **Can correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?**

    No, correlation cannot be greater than 1. It ranges from -1 to 1. A correlation of 0.3 suggests a weak positive linear relationship between two variables.

10. **The weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12. Your favorite newborn puppy weighs 1.1 pounds.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Calculate your puppy’s z-score (standard score).**

        $$z = \frac{x-\mu}{\sigma} = \frac{1.1-1}{0.12} \approx 0.833$$ 

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?**

        Since the one-tailed z-score equivalent to 90% of the population is 1.28, we have $$1.28 = \frac{x-1}{0.12}$$, $$x = 1.153$$.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Suppose the weight of newborn puppies followed a skew distribution. Would it still make sense to calculate z-scores?**

        Yes, z-scores can still be calculated for skewed distributions, but they may not have the same interpretability as in a normal distribution.

11. <span class="badge text-bg-secondary bg-danger">Hard</span> **Tossing a coin ten times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?**

    The best estimator for the actual value $$r$$ is the estimator $$p = \frac{h}{h+t} = \frac{10}{15}$$. Assuming a confidence level of 90% is desired, the corresponding Z value is 1.6449. So, the estimator has a margin of error $$E$$, where $$\|p - r\| < E$$. Now, $$E = \frac{Z}{2\sqrt{n}} = \frac{1.6449}{2\sqrt{15}} = 0.212$$. So the interval that contains $$r$$ with 90% confidence is thus: $$p - E < r < p + E$$, i.e. $$0.454 < r < 0.878$$. [TODO]

12. **Statistical significance**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?**

        We assess statistical significance by conducting hypothesis tests and looking at the p-value. If the p-value is less than a threshold (typically 0.05), we reject the null hypothesis and conclude the pattern is statistically significant.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the distribution of p-values?**

        The distribution of p-values under the null hypothesis is uniform.

    3. <span class="badge text-bg-secondary bg-danger">Hard</span> **Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?**

        It's important to remember that statistical significance does not imply practical significance. Moreover, a p-value is not the probability that the null hypothesis is true; it's the probability of observing the data given that the null hypothesis is true.

13. **Variable correlation**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **What happens to a regression model if two of their supposedly independent variables are strongly correlated?**

        If two independent variables in a regression model are strongly correlated, it can lead to multicollinearity, which makes reduces a model's predictive power.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do we test for independence between two categorical variables?**

        We can test independence between two categorical variables using the Chi-square test.

    3. <span class="badge text-bg-secondary bg-danger">Hard</span> **How do we test for independence between two continuous variables?**

        For continuous variables, we can use the correlation coefficient or mutual information.

14. <span class="badge text-bg-secondary bg-success">Easy</span> **A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?**

    A/B testing is a straightforward method to compare two versions of a solution. It allows for direct comparison and reduces the influence of confounding variables. However, it requires significant traffic to achieve statistical significance, it may not capture interactions between features and it funnels a lot of traffic through a non-optimal option. An alternative is the multi-armed bandit.

15. <span class="badge text-bg-secondary bg-warning">Medium</span> **You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be 95% sure that one placement is better?**

    This depends on the difference in click rates. If the difference is small, more visitors and clicks are needed to achieve 95% confidence. A statistical power analysis can be used to determine the needed sample size.

16. <span class="badge text-bg-secondary bg-warning">Medium</span> **Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?**

    Doubling the number of ads may not necessarily double the revenue. The relationship might not be linear and showing more ads could lead to user dissatisfaction. It is advisable to test the impact through a controlled experiment.

17. **Imagine that you have the prices of 10,000 stocks over the last 24 month period and you only have the price at the end of each month, which means you have 24 price points for each stock. After calculating the correlations of 10,000 * 9,992 pairs of stock, you found a pair that has the correlation to be above 0.8.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the probability that this happens by chance?**

        To compute the probability that a pair of stocks has a correlation above 0.8 by chance, we would need to make some assumptions about the distribution of the stock returns. If we assume that the monthly returns of the stocks are normally distributed and independent, then the correlation follows a Fisher transformation and we can compute the probability of seeing a correlation of 0.8 or more by chance.

        But even without these calculations, it's important to note that when testing a large number of hypotheses (in this case, comparing a large number of stock pairs), we're likely to encounter significant results purely by chance. This is known as the multiple comparisons problem, or the problem of "p-hacking."

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How to avoid this kind of accidental patterns?**

        To avoid these accidental patterns or "false discoveries," we can use multiple comparison corrections techniques, such as the Bonferroni correction or the False Discovery Rate (FDR) correction. These techniques adjust the significance level (the p-value) based on the number of comparisons to reduce the chance of false positives.

        Additionally, it's crucial to validate any significant findings on an out-of-sample dataset. In the context of stock prices, this could mean validating the correlation on future stock prices or on a separate dataset not used in the initial correlation analysis.

18. <span class="badge text-bg-secondary bg-danger">Hard</span> **How are sufficient statistics and Information Bottleneck Principle used in machine learning?**

    Sufficient statistics and the Information Bottleneck Principle both relate to the idea of compressing data in a way that retains as much relevant information as possible.

    Sufficient Statistics: In the context of machine learning, a sufficient statistic is a type of statistic that encapsulates all the information in the data that is relevant to the parameter estimation problem at hand. For instance, in estimating the mean and variance of a normal distribution, the sum and the sum of squares of the data are sufficient statistics. They compactly represent the data. When training a model, if we can identify and compute sufficient statistics, we can use them to simplify the learning process, often reducing computational requirements. This is particularly prevalent in algorithms like Naive Bayes or when using exponential family distributions.

    Information Bottleneck Principle: This is a method used in information theory to quantify the trade-off between the complexity of a representation of data and the preservation of relevant information. The idea is to create a compressed representation of the input (a "bottleneck") that retains as much information as possible about the output. It's used as a principle for deep learning to explain why certain layers in deep neural networks seem to create representations that maintain relevant information while discarding irrelevant details. It's also used as a principle for designing neural network architectures (like autoencoders) or loss functions.


# Computer Science

## Algorithms

1. **Write a Python function to recursively read a JSON file.**

    ```python
    def read_json(data):
      if isinstance(data, dict):
          for key, value in data.items():
              print(f"{key}: {read_json(value)}")        
      elif isinstance(data, list):
          for value in data:
              read_json(value)
      else:
          print(f"Value: {data}")
    ```

2. **Implement an $$O(N\log N)$$ sorting algorithm, preferably quick sort or merge sort.**

    ```python
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    ```


3. **Find the longest increasing subsequence in a string.**

    ```python
    def longest_increasing_subsequence(s):
        if not s:
            return ""
        
        # dp[i] represents the length of the longest increasing
        # subsequence that ends with the element at index i
        dp = [1] * len(s)
        for i in range (1, len(s)):
            for j in range(i):
                if s[i] > s[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        # find the index of the max length in dp
        max_length = max(dp)
        index = dp.index(max_length)

        # create longest increasing subsequence
        seq = s[index]
        for i in range(index-1, -1, -1):
            if s[i] < seq[0] and dp[i] == dp[index] - 1:
                seq = s[i] + seq
                index = i
                
        return seq
    ```

4. **Find the longest common subsequence between two strings.**

    ```python
    def longest_common_subsequence(s1, s2):
        matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        
        for i in reversed(range(len(s1))):
            for j in reversed(range(len(s2))):
                if s1[i] == s2[j]:
                    matrix[i][j] = matrix[i + 1][j + 1] + 1
                else:
                    matrix[i][j] = max(matrix[i + 1][j], matrix[i][j + 1])
        
        # reconstruct the longest common subsequence
        i, j = 0, 0
        lcs = ''
        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                lcs += s1[i]
                i += 1
                j += 1
            elif matrix[i + 1][j] > matrix[i][j + 1]:
                i += 1
            else:
                j += 1

        return lcs
    ```

5. **Traverse a tree in pre-order, in-order, and post-order.**

    ```python
    class TreeNode:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def preorder_traversal(root):
        if root:
            print(root.value)
            preorder_traversal(root.left)
            preorder_traversal(root.right)

    def inorder_traversal(root):
        if root:
            inorder_traversal(root.left)
            print(root.value)
            inorder_traversal(root.right)

    def postorder_traversal(root):
        if root:
            postorder_traversal(root.left)
            postorder_traversal(root.right)
            print(root.value)
    ```

6. **Given an array of integers and an integer $$k$$, find the total number of continuous subarrays whose sum equals $$k$$. The solution should have $$O(N)$$ runtime.**

    ```python
    def subarray_sum(nums, k):
        c = collections.Counter()
        c[0] += 1
        s = 0
        total = 0

        for num in nums:
            s += num
            if s-k in c:
                total += c[s-k]
            c[s] += 1

        return total
    ```

7. **There are two sorted arrays `nums1` and `nums2` with $$m$$ and $$n$$ elements respectively. Find the median of the two sorted arrays. The solution should have $$O(\log(m+n))$$ runtime.**

    ```python
    def findMedianSortedArrays(nums1, nums2):
        # Make sure nums1 is the smaller array. If not, swap them.
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        x, y = len(nums1), len(nums2)

        start = 0
        end = x

        while start <= end:
            partitionX = (start + end) // 2
            partitionY = (x + y + 1) // 2 - partitionX

            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
            minRightX = float('inf') if partitionX == x else nums1[partitionX]

            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
            minRightY = float('inf') if partitionY == y else nums2[partitionY]

            if maxLeftX <= minRightY and maxLeftY <= minRightX:
                # We have partitioned array at correct place
                # Now get max of left elements and min of right elements to compute median
                if (x + y) % 2 == 0:
                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2
                else:
                    return max(maxLeftX, maxLeftY)
            elif maxLeftX > minRightY:  # move towards left side
                end = partitionX - 1
            else:  # move towards right side
                start = partitionX + 1

        raise Exception("Arrays are not sorted")
    ```

8. **Write a program to solve a Sudoku puzzle by filling the empty cells. The board is of the size $$9 \times 9$$. It contains only 1--9 numbers. Empty cells are denoted with `*`. Each board has one unique solution.**

    ```python
    def solve_sudoku(board):
        find = find_empty(board)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1,10):
            if valid(board, i, (row, col)):
                board[row][col] = i

                if solve_sudoku(board):
                    return True

                board[row][col] = '*'

        return False


    def valid(board, num, pos):
        # Check row
        for i in range(len(board[0])):
            if board[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(board)):
            if board[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if board[i][j] == num and (i,j) != pos:
                    return False

        return True


    def find_empty(board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '*':
                    return (i, j)  # row, col

        return None
    ```

9. **Given a memory block represented by an empty array, write a program to manage the dynamic allocation of that memory block. The program should support two methods: `malloc()` to allocate memory and `free()` to free a memory block.**

    ```python
    class MemoryManager:
        def __init__(self, size):
            self.memory = [None]*size
            self.blocks = {}

        def malloc(self, size):
            # Find a block of memory
            start = None
            length = 0
            for i in range(len(self.memory)):
                if self.memory[i] is None:
                    if start is None:
                        start = i
                    length += 1
                    if length == size:
                        break
                else:
                    start = None
                    length = 0

            # Check if a large enough block was found
            if start is None or length < size:
                return None

            # Allocate the block
            block_id = max(self.blocks.keys(), default=0) + 1
            self.blocks[block_id] = (start, size)
            for i in range(start, start+size):
                self.memory[i] = block_id
            return block_id

        def free(self, block_id):
            # Free the block
            start, size = self.blocks[block_id]
            for i in range(start, start+size):
                self.memory[i] = None
            del self.blocks[block_id]
    ```

10. **Given a string of mathematical expression, such as `10 * 4 + (4 + 3) / (2 - 1)`, calculate it. It should support four operators `+`, `-`, `:`, `/`, and the brackets `()`.**

    ```python
    def calculate(expression):
        def apply_operator(operators, values):
            operator = operators.pop()
            right = values.pop()
            left = values.pop()
            if operator == '+':
                values.append(left + right)
            elif operator == '-':
                values.append(left - right)
            elif operator == '*':
                values.append(left * right)
            elif operator == '/':
                values.append(left / right)

        def greater_precedence(op1, op2):
            precedences = {'+': 1, '-': 1, '*': 2, '/': 2}
            return precedences[op1] > precedences[op2]

        operators = []
        values = []
        i = 0
        while i < len(expression):
            if expression[i] == ' ':
                i += 1
                continue
            if expression[i] in '0123456789':
                j = i
                while j < len(expression) and expression[j] in '0123456789':
                    j += 1
                values.append(int(expression[i:j]))
                i = j
            elif expression[i] in '+-*/':
                while operators and operators[-1] in '+-*/' and greater_precedence(operators[-1], expression[i]):
                    apply_operator(operators, values)
                operators.append(expression[i])
                i += 1
            elif expression[i] == '(':
                operators.append(expression[i])
                i += 1
            elif expression[i] == ')':
                while operators[-1] != '(':
                    apply_operator(operators, values)
                operators.pop()  # Discard the '('
                i += 1
        while operators:
            apply_operator(operators, values)

        return values[0]
    ```

11. **Given a directory path, descend into that directory and find all the files with duplicated content.**

    ```python
    import os
    import hashlib

    def find_duplicates(dir_path):
        # Dictionary to store the hashes of files
        file_dict = {}

        # Walk the directory
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                filehash = hash_file(filepath)
                # Append the file path to the list of files with this hash
                file_dict.setdefault(filehash, []).append(filepath)

        # Filter out the file hashes that have only one file
        duplicates = {filehash: filepaths for filehash, filepaths in file_dict.items() if len(filepaths) > 1}

        return duplicates


    def hash_file(filepath):
        hasher = hashlib.sha1()
        with open(filepath, 'rb') as file:
            buf = file.read()
            hasher.update(buf)
        return hasher.hexdigest()

    # Usage
    duplicates = find_duplicates('/path/to/directory')
    for paths in duplicates.values():
        print('Duplicate files:')
        for path in paths:
            print(' ', path)
    ```

12. **In Google Docs, you have the Justify alignment option that spaces your text to align with both left and right margins. Write a function to print out a given text line-by-line (except the last line) in Justify alignment format. The length of a line should be configurable.**

    ```python
    def justify_text(text, line_length):
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            # Check if adding the new word to the current line would
            # make it longer than the maximum line length
            if len(' '.join(current_line + [word])) > line_length:
                # If so, store the current line and start a new one
                lines.append(current_line)
                current_line = [word]
            else:
                # Otherwise, add the word to the current line
                current_line.append(word)

        # Don't forget to store the last line as well
        lines.append(current_line)

        # Join words in each line with extra spaces for justification
        for line in lines[:-1]:  # Don't justify the last line
            if len(line) == 1:
                print(line[0].ljust(line_length))
            else:
                words, spaces = len(line), line_length - sum(len(word) for word in line)
                spaces_per_word = spaces // (words - 1)
                extra_spaces = spaces % (words - 1)

                for i in range(words - 1):
                    print(line[i], end='')
                    print(' ' * (spaces_per_word + (i < extra_spaces)))

                print(line[-1])
        
        # Print the last line with left alignment
        print(' '.join(lines[-1]).ljust(line_length))
    ```

13. **You have 1 million text files, each is a news article scraped from various news sites. Since news sites often report the same news, even the same articles, many of the files have content very similar to each other. Write a program to filter out these files so that the end result contains only files that are sufficiently different from each other in the language of your choice. You’re free to choose a metric to define the “similarity” of content between files.**

    ```python
    from datasketch import MinHash, MinHashLSH
    from nltk import ngrams
    import os

    # The threshold determines the cut-off for considering whether files are similar
    threshold = 0.8

    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Walk through the files in the directory
    for dirpath, dirnames, filenames in os.walk("/path/to/files"):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r') as file:
                # For each file, we compute a MinHash
                m = MinHash(num_perm=128)
                text = file.read()
                for d in ngrams(text, 3):
                    m.update("".join(d).encode('utf-8'))
                # Add to the index
                lsh.insert(filename, m)

    # Keep track of which files to keep
    files_to_keep = set()

    # Now we need to determine which files are unique enough to keep
    for dirpath, dirnames, filenames in os.walk("/path/to/files"):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'r') as file:
                text = file.read()
                m = MinHash(num_perm=128)
                for d in ngrams(text, 3):
                    m.update("".join(d).encode('utf-8'))
                
                # Check if there are any candidates in the LSH index that are similar
                result = lsh.query(m)
                if len(result) <= 1:
                    files_to_keep.add(filepath)
    ```

## Complexity and Numerical Analysis

1. **Matrix multiplication**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **You have three matrices: $$A \in R^{100 \times 5}, B \in R^{5 \times 200}, C \in R^{200 \times 20}$$ and you need to calculate the product $$ABC$$. In what order would you perform your multiplication and why?**

        In matrix multiplication, the order of operations affects the computational complexity. The number of operations to multiply two matrices of size $$p \times q$$ and $$q \times r$$ is $$pqr$$.

        Multiplying $$(AB)C$$ has a total of $$5 \times 10^5$$ operations
        - $$AB$$ results in a $$100 \times 200$$ matrix: $$100 \times 5 \times 200 = 10^5$$ operations
        - $$(AB)C$$ results in a $$100 \times 20$$ matrix: $$100 \times 200 \times 20 = 4 \times 10^5$$ operations
        
        Multiplying $$A(BC)$$ has a total of $$3 \times 10^4$$ operations
        - $$BC$$ results in a $$5 \times 20$$ matrix: $$5 \times 200 \times 20 = 2 \times 10^4$$ operations
        - $$A(BC)$$ results in a $$100 \times 20$$ matrix: $$100 \times 5 \times 20 = 10^4$$ operations

    2. [**M] Now you need to calculate the product of $$N$$ matrices $$A_1A_2...A_n$$. How would you determine the order in which to perform the multiplication?**

        The problem of determining the optimal sequence of operations in matrix chain multiplication is a well-studied problem in dynamic programming. This involves computing a table where the entry for each pair of matrices gives the minimum cost of multiplying them together.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **What are some of the causes for numerical instability in deep learning?**

    Numerical instability in deep learning could be caused by various issues such as vanishing/exploding gradients, overfitting, ill-conditioned matrices, numerical precision issues, or a large learning rate.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **In many machine learning techniques (e.g. batch norm), we often see a small term $$\epsilon$$ added to the calculation. What’s the purpose of that term?**

    The small term $$\epsilon$$ added to some calculations in machine learning techniques is for numerical stability. For instance, in batch normalization, it prevents division by zero when the standard deviation is close to zero.

4. <span class="badge text-bg-secondary bg-success">Easy</span> **What made GPUs popular for deep learning? How are they compared to TPUs?**

    GPUs became popular for deep learning because they are designed for parallel processing, which is critical for matrix operations common in deep learning. Compared to TPUs (Tensor Processing Units), GPUs are more flexible and can be used for both training and inference, while TPUs are specifically designed for deep learning tasks and can provide higher performance for these specific tasks.

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **What does it mean when we say a problem is intractable?**

    When we say a problem is intractable, it means the problem is computationally very difficult to solve, in the sense that there doesn't exist an algorithm that can solve the problem in a reasonable amount of time as the problem size increases.

6. <span class="badge text-bg-secondary bg-danger">Hard</span> **What are the time and space complexity for doing backpropagation on a recurrent neural network?**

    The time complexity of backpropagation in a recurrent neural network is usually $$O(TN^2)$$, where $$T$$ is the sequence length and $$N$$ is the number of hidden units. The space complexity is $$O(TN)$$, as the activations of all nodes in all layers must be stored for backpropagation.

7. <span class="badge text-bg-secondary bg-danger">Hard</span> **Is knowing a model’s architecture and its hyperparameters enough to calculate the memory requirements for that model?**

    Knowing a model's architecture and its hyperparameters is often not enough to calculate the memory requirements for that model. This is because memory requirements can also be influenced by the batch size, sequence length (for RNNs), and whether or not you are storing intermediate activations for backpropagation, among other factors.

8. <span class="badge text-bg-secondary bg-danger">Hard</span> **Your model works fine on a single GPU but gives poor results when you train it on 8 GPUs. What might be the cause of this? What would you do to address it?**

    The cause could be an issue with the synchronization and communication between the GPUs or an inappropriate batch size for multi-GPU training. It's also possible that the model or the algorithm isn't well-suited for parallelization. To address this, one might need to look into different multi-GPU training strategies, consider using gradient accumulation or adjust the batch size.

9. <span class="badge text-bg-secondary bg-danger">Hard</span> **What benefits do we get from reducing the precision of our model? What problems might we run into? How to solve these problems?**

    Reducing the precision of a model can significantly reduce the memory requirements and computational cost. However, it may cause numerical stability issues or a decrease in model accuracy due to the reduced numerical precision. This can be addressed using techniques like mixed precision training which uses a combination of different precisions where it's beneficial.

10. <span class="badge text-bg-secondary bg-danger">Hard</span> **How to calculate the average of 1M floating-point numbers with minimal loss of precision?**

    To calculate the average of 1M floating-point numbers with minimal loss of precision, you can use the Kahan summation algorithm or pairwise summation which improves the numerical precision by reducing the error introduced in the addition of a sequence of finite precision floating point numbers.

11. <span class="badge text-bg-secondary bg-danger">Hard</span> **How should we implement batch normalization if a batch is spread out over multiple GPUs?**

    If a batch is spread out over multiple GPUs, batch normalization can be implemented by aggregating the statistics (mean and variance) across the GPUs before applying the normalization. This is known as synchronized batch normalization or cross-GPU batch normalization.

12. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given the following code snippet. What might be a problem with it? How would you improve it?**

    ```python
    import numpy as np

    def within_radius(a, b, radius):
        if np.linalg.norm(a - b) < radius:
            return 1
        return 0

    def make_mask(volume, roi, radius):
        mask = np.zeros(volume.shape)
        for x in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                for z in range(volume.shape[2]):
                    mask[x, y, z] = within_radius((x, y, z), roi, radius)
        return mask
    ```

    This was an actual [StackOverflow question](https://stackoverflow.com/questions/39667089/python-vectorizing-nested-for-loops/39667342). The accepted answer suggested a vectorised approach to increase computational efficiency.

    ```python
    def make_mask(volume, roi, radius):
        m, n, r = volume.shape
        x, y, z = np.mgrid[0:m, 0:n, 0:r]
        X = x - roi[0]
        Y = y - roi[1]
        Z = z - roi[2]
        mask = X**2 + Y**2 + Z**2 < radius**2
        return mask
    ```

# Machine Learning Workflows

## Basics

1. <span class="badge text-bg-secondary bg-success">Easy</span> **Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.**

    The terms supervised, unsupervised, weakly supervised, semi-supervised, and active learning refer to different ways that machine learning models can learn from data.
    - Supervised learning involves learning a function that maps input data to output data based on example input-output pairs.
    - Unsupervised learning involves finding patterns and structure in input data without any corresponding output data.
    - Weakly supervised learning involves learning from a training set that is not perfectly labelled or has noisy labels.
    - Semi-supervised learning involves learning from a training set that contains a small amount of labeled data and a large amount of unlabeled data.
    - Active learning involves a model that can request the labels for certain instances from an oracle or expert to improve its learning efficiency.

2. **Empirical risk minimization.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the risk in empirical risk minimization?**

        The risk in empirical risk minimization is the loss incurred by the model's predictions on the data. It is a measure of the model's ability to fit the data.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why is it empirical?**

        It is empirical because the risk is estimated based on the observed data.

    3. <span class="badge text-bg-secondary bg-success">Easy</span> **How do we minimize that risk?**

        We minimize that risk by iteratively adjusting the model's parameters to improve the fit on the training data, typically using some form of gradient descent.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?**

    Occam's razor in ML suggests that simpler models that fit the data are preferred over more complex ones. This is often applied in model selection and regularization methods to avoid overfitting.

4. <span class="badge text-bg-secondary bg-success">Easy</span> **What are the conditions that allowed deep learning to gain popularity in the last decade?**

    Several conditions have led to the rise of deep learning: the availability of large datasets, advancements in hardware like GPUs, and breakthroughs in algorithms and architectures.

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?**

    A deep neural network can be more expressive than a wide one with the same number of parameters because depth can enable hierarchical representation of features which can be crucial in many tasks.

6. <span class="badge text-bg-secondary bg-danger">Hard</span> **The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?**

    Although a neural network with 1 hidden layer can theoretically approximate any continuous function, in practice it may require an exponential number of nodes to do so. Also, the theorem does not guarantee that we can easily learn the weights of such a network.

7. <span class="badge text-bg-secondary bg-success">Easy</span> **What are saddle points and local minima? Which are thought to cause more problems for training large NNs?**

    Saddle points and local minima are points where the gradient of a loss function is zero. Saddle points are generally thought to cause more problems for training large NNs because they are more prevalent in high dimensions and gradient-based optimizers can get stuck at these points.

8. **Hyperparameters.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What are the differences between parameters and hyperparameters?**

        Parameters are the internal variables that the model learns during training, while hyperparameters are the external configurations that user set before training.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why is hyperparameter tuning important?**

        Hyperparameter tuning is important because the performance of a model can significantly depend on the choice of hyperparameters.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Explain algorithm for tuning hyperparameters.**

        Hyperparameters can be tuned using various methods like grid search, random search, or advanced methods like Bayesian optimization.

9. **Classification vs. regression.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What makes a classification problem different from a regression problem?**

        Classification problems predict a categorical label, while regression problems predict a continuous value.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Can a classification problem be turned into a regression problem and vice versa?**

        Turning a classification problem into a regression problem and vice versa largely depends on the specifics of your task and the nature of your data. However, here are some general ways this can be done:
        - *Classification to Regression:* If your classification problem has ordinal classes (i.e., there's an order to the classes, like 'low', 'medium', 'high'), you can convert it into a regression problem by mapping the classes to numerical values and predicting these numerical values instead. For example, in a customer satisfaction survey, the ratings could be "Very Unhappy", "Unhappy", "Neutral", "Happy", "Very Happy". These could be converted into values from 1 to 5, and treated as a regression problem.
        - *Regression to Classification:* You can turn a regression problem into a classification one by binning the output variable. This means dividing the range of the output variable into distinct categories or 'bins'. For example, if you're trying to predict a person's income (a regression problem), you could create bins like 'low income', 'medium income', and 'high income' and turn it into a classification problem.

10. **Parametric vs. non-parametric methods.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the difference between parametric methods and non-parametric methods? Give an example of each method.**

        Parametric methods make assumptions about the underlying distribution of the data and summarize it using a finite set of parameters (like linear regression). Non-parametric methods do not make such assumptions and can adapt to the data's structure (like decision trees).

    2. <span class="badge text-bg-secondary bg-danger">Hard</span> **When should we use one and when should we use the other?**

        The choice between parametric and non-parametric methods largely depends on the underlying data and the problem at hand:
        - *Parametric Methods:* You might choose a parametric method if you have prior knowledge about the distribution of your data, or if you need a simpler, less data-intensive model. For example, if you're dealing with a simple binary classification problem and you know that your data are linearly separable, you might use logistic regression (a parametric method).
        - *Non-parametric Methods:* Non-parametric methods make fewer assumptions about the data's distribution and can be more flexible, making them useful when you have complex, high-dimensional data, or when you don't know the data's distribution. For instance, if you're dealing with a complex classification problem and you don't know the data's distribution, you might use a decision tree or a support vector machine (both non-parametric methods). However, non-parametric methods can require more data and be more computationally intensive.

11. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why does ensembling independently trained models generally improve performance?**

    Ensembling independently trained models generally improve performance because it reduces the variance of the predictions, making them more robust and less likely to overfit to the training data. This is due to the diversity among models, which allows them to capture different aspects of the data. The errors made by one model are often compensated for by the correct predictions of the other models.

12. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?**

    L1 regularization, also known as Lasso, tends to lead to sparsity because it penalizes the absolute value of the weights, thereby encouraging some weights to go to zero. This is especially useful when dealing with high-dimensional data where feature selection is important. On the other hand, L2 regularization, also known as Ridge, penalizes the square of the weights, thereby pushing them closer to zero but not necessarily to zero. This is because the square function has a more gradual increase near zero compared to the absolute value function. This property of L2 regularization often results in models that are less sparse and can better manage multicollinearity.

13. <span class="badge text-bg-secondary bg-success">Easy</span> **Why does an ML model’s performance degrade in production?**

    An ML model’s performance can degrade in production for several reasons. One of the most common is concept drift, where the underlying data distribution changes over time. The model might also suffer from overfitting, where it performs well on the training data but fails to generalize to unseen data. Other potential issues include changes in the input data quality, differences between the training and production environments, or a mismatch between the performance measure used during training and the actual goal in production.

14. <span class="badge text-bg-secondary bg-warning">Medium</span> **What problems might we run into when deploying large machine learning models?**

    Deploying large machine learning models can introduce a few challenges. These models typically require more computational resources, which can be costly. They can also be slower to run predictions, which might not be suitable for applications that require real-time responses. Additionally, they can be more prone to overfitting, especially if the amount of available data is limited.

15. **Your model performs really well on the test set but poorly in production.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **What are your hypotheses about the causes?**

        If a model performs well on the test set but poorly in production, the issues might be related to overfitting, data leakage, changes in the data distribution (concept drift), or problems with the evaluation metrics.
    
    2. <span class="badge text-bg-secondary bg-danger">Hard</span> **How do you validate whether your hypotheses are correct?**

        To validate these hypotheses, you would first want to check the evaluation metrics used during training and make sure they align with the real-world objectives. You might also want to reevaluate your cross-validation strategy to make sure it's robust. For concept drift, you could monitor the predictions and inputs over time and check for changes in their distributions. For data leakage, you would need to carefully examine your data processing pipeline.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Imagine your hypotheses about the causes are correct. What would you do to address them?**

        If the hypotheses are correct, there are several approaches you might take to address them. If the problem is overfitting, you might want to simplify your model, add regularization, or gather more data. If it's data leakage, you would need to fix your data processing pipeline to prevent information from the future "leaking" into your model training. If it's concept drift, you might need to implement a system to retrain your models periodically with fresh data. If the issue lies with the evaluation metrics, you might need to choose a metric that better aligns with your real-world objectives.

## Sampling and Creating Training Data

1. <span class="badge text-bg-secondary bg-success">Easy</span> **If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?**

    There are $$\binom{6}{2}$$ ways to choose 2 shirts out of 6 and $$\binom{4}{1}$$ ways to choose 1 pair of pants out of 4. Therefore, there are $$\binom{6}{2} \binom{4}{1} = 15 \times 4 = 60$$ ways.

2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?**

    Sampling with replacement means that each time you draw a sample, you put it back into the population before drawing the next one. This means that each draw is independent of the others. In contrast, sampling without replacement means that each sample you draw is removed from the population before drawing the next one. This means that each draw is dependent on the others. You might use sampling with replacement when the population is large and the probability of drawing the same item twice is low, such as drawing a lottery number. You might use sampling without replacement when the population is small and the probability of drawing the same item twice is high, such as drawing cards from a deck.

3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Explain Markov Chain Monte Carlo sampling.**

    Markov chain Monte Carlo (MCMC) sampling is a technique for estimating the distribution of a random variable by constructing a Markov chain that has the desired distribution as its equilibrium distribution. The states of the chain after a large number of steps are then used as samples from the desired distribution. MCMC is especially useful when direct sampling is difficult.

4. <span class="badge text-bg-secondary bg-warning">Medium</span> **If you need to sample from high-dimensional data, which sampling method would you choose?**

    When dealing with high-dimensional data, it's often beneficial to use methods like Gibbs sampling or Hamiltonian Monte Carlo. These techniques allow for efficient exploration of the high-dimensional space by leveraging the structure of the problem.

5. <span class="badge text-bg-secondary bg-danger">Hard</span> **Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.**

    Some of the candidate sampling algorithms include:
    - Importance Sampling: This method focuses on sampling from parts of the distribution that contribute the most to the integral.
    - Rejection Sampling: This method samples from a proposal distribution and rejects samples that don't satisfy a certain condition.
    - Metropolis-Hastings: This method creates a random walk using a proposal distribution and accepts or rejects new samples based on the ratio of their probabilities.
    - Gibbs Sampling: This method involves updating one variable at a time, conditional on the other variables.

6. **Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **How would you sample 100K comments to label?**

        To sample 100K comments to label, you might want to use a stratified sampling approach to ensure that the sample is representative of the different types of comments and users on Reddit. This could involve grouping the comments by user or topic, and then randomly selecting comments within each group.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?**

        To estimate the quality of the labels, you might want to randomly sample a subset of the labels to manually review. The number of labels to review will depend on the degree of confidence you want in your estimate. A commonly used rule of thumb is to sample at least 30 labels, but you might want to sample more if you want a higher degree of confidence or if you expect a high degree of variability in label quality.

7. <span class="badge text-bg-secondary bg-warning">Medium</span> **Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?**

    
    The argument might be subject to selection bias. The articles that get translated might not be a random sample of all articles. They could be the ones that are expected to be the most popular or the most relevant to the Chinese-speaking audience, and therefore they might get more views regardless of the translation.
    

8. <span class="badge text-bg-secondary bg-warning">Medium</span> **How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?**

    One way to determine whether two sets of samples come from the same distribution is to use a statistical test such as the Kolmogorov-Smirnov test. This test compares the cumulative distribution functions of the two samples and returns a p-value indicating the probability that the samples could have come from the same distribution if the null hypothesis were true.

9. <span class="badge text-bg-secondary bg-danger">Hard</span> **How do you know you’ve collected enough samples to train your ML model?**

    There's no fixed rule for how many samples are enough to train your ML model. It depends on the complexity of the model and the variability of the data. As a general rule, you want to have enough samples so that the model can learn the underlying patterns in the data without overfitting. You can use techniques like cross-validation to assess whether your model is able to generalize well to unseen data.

10. <span class="badge text-bg-secondary bg-warning">Medium</span> **How to determine outliers in your data samples? What to do with them?**

    You can determine outliers in your data samples by looking at the statistical properties of the data. For instance, you might consider any data point that is more than 1.5 interquartile ranges (IQRs) below the first quartile or above the third quartile to be an outlier. Once you've identified the outliers, you need to decide what to do with them. If they're due to errors or noise, you might want to remove them. If they're due to unusual but valid observations, you might want to keep them but consider using a robust model that can handle outliers.

11. **Sample duplication**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **When should you remove duplicate training samples? When shouldn’t you?**

        You should remove duplicate training samples when you believe they're due to errors in data collection or processing, or when they could bias the learning process. However, you shouldn't remove duplicates when they're a valid representation of the underlying distribution, or when they carry important information about the target variable.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What happens if we accidentally duplicate every data point in your train set or in your test set?**

        If you accidentally duplicate every data point in your training set, it could lead to overfitting, as the model might learn to perfectly predict the duplicated samples instead of learning the underlying patterns in the data. If you duplicate every data point in your test set, it could lead to an overestimate of the model's performance, as the same data points will be used multiple times to evaluate the model.

12. **Missing data**

    1. <span class="badge text-bg-secondary bg-danger">Hard</span> **In your dataset, two out of 20 variables have more than 30% missing values. What would you do?**

        If two out of 20 variables have more than 30% missing values, you could consider several approaches. You might choose to impute the missing values using a method like mean imputation or multiple imputation. Alternatively, you might choose to exclude those variables from the analysis, especially if they're not highly correlated with the target variable.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How might techniques that handle missing data make selection bias worse? How do you handle this bias?**

        Techniques that handle missing data could make selection bias worse if the missingness is related to the outcome variable. This is because they could introduce a systematic error into the estimates. To handle this bias, you could use methods like inverse probability weighting or multiple imputation, which aim to correct for the bias introduced by the missing data.

13. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why is randomization important when designing experiments (experimental design)?**

    Randomization is important when designing experiments because it helps to ensure that the results of the experiment are due to the treatment and not to confounding factors. By randomly assigning subjects to treatment and control groups, you can help to ensure that any differences between the groups are due to chance rather than systematic bias.

14. **Class imbalance.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How would class imbalance affect your model?**

        Class imbalance could affect your model by biasing it towards the majority class, leading to a high error rate for the minority class.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why is it hard for ML models to perform well on data with class imbalance?**

        It's hard for ML models to perform well on data with class imbalance because they tend to be biased towards the majority class. This means that they might not learn the patterns associated with the minority class very well.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?**

        If your model is making a lot of false negatives, you might want to use techniques like oversampling the minority class, undersampling the maority class, or using a cost-sensitive learning algorithm that assigns a higher cost to false negatives. You might also want to use an evaluation metric that takes into account both precision and recall, such as the F1 score.

15. **Training data leakage**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?**

        If you oversample the rare class before splitting your data into train and test sets, you might end up with the same samples in both sets. This would lead to an overestimate of your model's performance, as it would be tested on samples it has already seen.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?**

        Randomly splitting your data into train and test splits could lead to data leakage if there's a temporal dependence in your data. For instance, if a comment is a reply to another comment, and one of them ends up in the training set and the other in the test set, the model could learn to make predictions based on information that wouldn't be available at the time of prediction in a real-world setting.

16. <span class="badge text-bg-secondary bg-warning">Medium</span> **How does data sparsity affect your models?**

    Data sparsity affects your models by making them more complex and potentially leading to overfitting. In sparse data, many features have zero or near-zero variance, and the model might learn to make predictions based on these features, which could lead to poor generalization to new data.

17. **Feature leakage**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What are some causes of feature leakage?**

        Feature leakage occurs when your model is trained on data that includes information that wouldn't be available at the time of prediction in a real-world setting. Causes of feature leakage include pre-processing the entire dataset before splitting it into train and test sets, and including future information in the training data.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why does normalization help prevent feature leakage?**

        Normalization helps prevent feature leakage by ensuring that the scale of the features doesn't affect the model's learning process. If features are not normalized, the model might learn to make predictions based on the scale of the features rather than their actual values.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do you detect feature leakage?**

        To detect feature leakage, you could look for unusually high performance on the test set, which could suggest that the model is using information that wouldn't be available at the time of prediction. You could also inspect your data preprocessing and feature engineering steps to ensure that they don't include future information.

18. <span class="badge text-bg-secondary bg-warning">Medium</span> **Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?**

    If you randomly shuffle your data before splitting it into train, valid, and test sets, you could end up with a temporal mismatch between your training and test data. For instance, if tweets from the same event or topic end up in different splits, your model could be trained on future information, leading to an overestimate of its performance.

19. <span class="badge text-bg-secondary bg-warning">Medium</span> **You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?**

    For numerical features, you might want to normalize or standardize them to ensure that they're on the same scale. For textual features, you might want to use techniques like bag-of-words or TF-IDF to convert them into a numerical format that can be used by the neural network.

20. <span class="badge text-bg-secondary bg-danger">Hard</span> **Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?**

    If you start using all the features available instead of just a subset, you might experience the curse of dimensionality. The training error might decrease, as the model will have more information to learn from. However, the test error might increase, as the model could overfit to the training data and fail to generalize well to unseen data.

## Objective functions, metrics, and evaluation

1. **Convergence**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **When we say an algorithm converges, what does convergence mean?**

        Convergence in the context of a machine learning algorithm usually means that the iterative process of optimization has reached a point where the changes in the loss function or model parameters have become very small, indicating that the model has (hopefully) found an optimal or near-optimal set of parameters. This state implies that further training is unlikely to improve the model significantly.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **How do we know when a model has converged?**

        We know a model has converged when the changes in the loss function or parameters become smaller than a predefined threshold, or when the improvement in validation performance ceases or starts deteriorating.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **Draw the loss curves for overfitting and underfitting.**

    In overfitting, the training loss continues to decrease as the model trains, but the validation loss decreases initially and then starts to increase. In underfitting, both the training and validation loss are high because the model fails to learn the underlying pattern in the data.

    [TODO]

3. **Bias-variance trade-off**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the bias-variance trade-off?**

        The bias-variance trade-off is a fundamental problem in machine learning that refers to the balancing act between a model's ability to fit the data well (low bias, high variance) and its ability to generalize to new data (high bias, low variance). Overfitting corresponds to low bias and high variance, while underfitting corresponds to high bias and low variance.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How’s this tradeoff related to overfitting and underfitting?**

        High bias, low variance models tend to underfit the data, as they oversimplify the problem and miss out on important patterns. On the other hand, low bias, high variance models tend to overfit the data, as they capture the noise in the training data and fail to generalize well to new data.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do you know that your model is high variance, low bias? What would you do in this case?**

        If your model is high variance and low bias, it likely has a good fit on the training set but poor performance on the validation set. To address this, you could try to gather more training data, implement regularization, or simplify the model.

    4. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do you know that your model is low variance, high bias? What would you do in this case?**

        If your model is low variance, high bias, it will likely have similar performance on both the training and validation sets, but the performance will not be very good. You can try to improve the model by adding more features, making the model more complex, or reducing regularization.

4. **Cross-validation**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Explain different methods for cross-validation.**

        Common methods of cross-validation include k-fold cross-validation, stratified k-fold cross-validation, leave-one-out cross-validation, and time-series cross-validation. In k-fold cross-validation, the data is divided into k subsets, and the model is trained k times, each time using a different subset as the validation set and the remaining subsets as the training set.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why don’t we see more cross-validation in deep learning?**

        We don't see as much cross-validation in deep learning mainly because of computational reasons. Deep learning models typically require a lot of computational resources and time to train, and running k-fold cross-validation would multiply the training time by k.

5. **Train, valid, test splits**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s wrong with training and testing a model on the same data?**

        If you train and test a model on the same data, you risk overfitting to that data, which means your model will not generalize well to new, unseen data.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why do we need a validation set on top of a train set and a test set?**

        A validation set is needed to tune hyperparameters and make decisions during the model training process, such as when to stop training. Using a test set for these decisions would lead to a biased estimate of the model's performance.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Your model’s loss curves on the train, valid, and test sets look like this. What might have been the cause of this? What would you do?**

        ![png](/assets/images/machine-learning-interviews/1.svg){: .multiply }

        The decrease in test error and increase in validation error seems unusual because the validation and test sets usually show similar trends -- they both ideally represent unseen, real-world data. If they diverge, it could be due to several reasons:

        Mismatched Distributions: The validation and test sets might come from different distributions. In other words, they might not represent the same underlying population of data. This could happen if there was some bias or mistake in how the data was split. For example, they could differ in terms of time periods, demographic distributions, or other key features.

        Test Set Size: If your test set is too small, the decrease in test error could simply be a result of statistical noise and may not represent a meaningful trend.

        Data Leakage: If the test set data in some way influences or is present in the training process, it can result in an overly optimistic test score. This is known as data leakage, and it could be due to pre-processing steps, feature extraction, or other factors.

6. <span class="badge text-bg-secondary bg-success">Easy</span> **Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?**

    It's essential to look beyond accuracy when evaluating a model for a medical diagnosis system. A 99.99% accuracy might sound impressive, but it might be misleading, especially in cases of class imbalance. I would ask about the model's precision, recall, and F1 score, especially for the minority class (cancer-positive patients in this case).

7. **F1 score**

    1. [**E] What’s the benefit of F1 over the accuracy?**

        The F1 score is the harmonic mean of precision and recall. It's more informative than accuracy in scenarios where the data is imbalanced because it takes both false positives and false negatives into account.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Can we still use F1 for a problem with more than two classes. How?**

        Yes, we can use the F1 score for multiclass problems by applying it to each class individually and then averaging the results. This can be done using the "micro" or "macro" averaging method, depending on whether we want to weight each class equally or based on its size.

8. **Given a binary classifier that outputs the following confusion matrix.**

    |              | Predicted True | Predicted False |
    |--------------|---------------:|----------------:|
    | Actual True  | 30             | 20              |
    | Actual False | 5              | 40              |
    {: .table .table-striped }

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Calculate the model’s precision, recall, and F1.**

        Precision = TP / (TP + FP) = 30 / (30 + 5) = 0.857.

        Recall = TP / (TP + FN) = 30 / (30 + 20) = 0.6.

        F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.857 * 0.6) / (0.857 + 0.6) = 0.705.
    
    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What can we do to improve the model’s performance?**

        Adjusting the threshold for classification can help improve performance. For example, if false negatives are very costly, we might want to lower the threshold for predicting a positive class.

9. **Consider a classification where 99% of data belongs to class A and 1% of data belongs to class B.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **If your model predicts A 100% of the time, what would the F1 score be?**

        If your model predicts A 100% of the time, the F1 score would be 1 when A is mapped to 1 and B is mapped to 0, but it would be 0 when A is mapped to 0 and B is mapped to 1.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **If we have a model that predicts A and B at a random (uniformly), what would the expected F1 be?**

        If we have a model that predicts A and B at a random (uniformly), the expected F1 score would be very low, especially for class B, due to a high number of false positives and false negatives.

10. <span class="badge text-bg-secondary bg-warning">Medium</span> **For logistic regression, why is log loss recommended over MSE (mean squared error)?**

    Log loss is recommended over MSE for logistic regression because it corresponds to the negative log-likelihood of the observed labels given the predicted probabilities, which is the exact quantity that logistic regression is trying to optimize.

11. <span class="badge text-bg-secondary bg-warning">Medium</span> **When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?**

    We might prefer RMSE over MAE when large errors are particularly undesirable, as RMSE squares the errors before averaging them, thus giving more weight to large errors. Conversely, we might prefer MAE when all errors are equally important, as MAE treats all errors the same regardless of their magnitude.

12. <span class="badge text-bg-secondary bg-warning">Medium</span> **Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.**

    For a binary classification task, the negative log-likelihood is $$-\log(P(y \mid x))$$, where $$P(y \mid x)$$ is the predicted probability of the true class. The binary cross-entropy loss is $$-y \log(p) - (1-y)\log(1-p)$$, where $$y$$ is the true label and $$p$$ is the predicted probability. If we substitute $$P(y \mid x)$$ for $$p$$ in the binary cross-entropy formula, we see that the two are the same.

13. <span class="badge text-bg-secondary bg-warning">Medium</span> **For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?**

    Cross-entropy is usually a better choice than MSE for multiclass classification because it directly models the probability distribution of the data, making it more suitable for tasks where the goal is to predict probabilities of various classes. MSE, on the other hand, can lead to slower and potentially less stable training.

14. <span class="badge text-bg-secondary bg-success">Easy</span> **Consider a language with an alphabet of 27 characters. What would be the maximal entropy of this language?**

    In information theory, the maximal entropy of a discrete distribution is achieved when all outcomes are equally likely. So for a language with an alphabet of 27 characters, where each character is equally likely, the maximum entropy would be $$\log_2(27) = 4.755 \textrm{ bits}$$.

15. <span class="badge text-bg-secondary bg-success">Easy</span> **A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?**

    A common measure of the difference between two probability distributions is the Kullback-Leibler (KL) divergence. Another measure is the Jensen-Shannon (JS) divergence, which is symmetric and always has a finite value.

16. **MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How do MPE and MAP differ?**

        MAP gives the most probable single assignment to a set of variables, whereas MPE gives the most probable assignment to all the variables in the model, assuming that the variables are dependent on each other.

    2. <span class="badge text-bg-secondary bg-danger">Hard</span> **Give an example of when they would produce different results.**

        For example, consider a model where we want to predict two dependent variables Y and Z given some data X. MAP might give us the most probable values for Y and Z independently, but those might not be the most probable values when Y and Z are considered together. In contrast, MPE would give us the most probable values for Y and Z considering their dependence.

17. <span class="badge text-bg-secondary bg-success">Easy</span> **Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use?**

    If you want to ensure that the predicted stock price is within 10% of the actual price, you might want to use a metric that heavily penalizes predictions that are off by more than 10%. Mean Absolute Percentage Error (MAPE) could be a good choice here, as it directly measures the average percentage error.

# Machine Learning Algorithms

## Classic Machine Learning Algorithms

1. <span class="badge text-bg-secondary bg-success">Easy</span> **What are the basic assumptions to be made for linear regression?**

    The basic assumptions for linear regression include linearity (the relationship between features and target is linear), independence (the observations are independent of each other), homoscedasticity (the variance of error terms is constant across all levels of the independent variables), and normality (the error terms follow a normal distribution).

2. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens if we don’t apply feature scaling to logistic regression?**

    If feature scaling is not applied to logistic regression, the model might take longer to converge because features with larger ranges might dominate the objective function. This could also impact the interpretation of the coefficients and might lead to less accurate predictions.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **What are the algorithms you’d use when developing the prototype of a fraud detection model?**

    For a fraud detection model prototype, you could consider algorithms like Decision Trees, Logistic Regression, SVM, Naive Bayes, or ensemble methods like Random Forests and XGBoost. Anomalies detection methods like Isolation Forest or Autoencoders (neural networks) could also be useful.

4. **Feature selection**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Why do we use feature selection?**

        Feature selection is used to reduce overfitting, improve accuracy, and reduce training time by removing irrelevant or redundant features.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What are some of the algorithms for feature selection? Pros and cons of each.**

        Some algorithms for feature selection include filter methods (like Pearson correlation), wrapper methods (like recursive feature elimination), and embedded methods (like Lasso regularization). Filter methods are generally faster but less accurate, while wrapper methods are more accurate but computationally expensive. Embedded methods offer a good balance, as they incorporate feature selection as part of the model training process.

5. **k-means clustering**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How would you choose the value of k?**

        The value of k can be chosen using methods like the elbow method or the silhouette method, which measure the quality of clustering for different k values.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?**

        If the labels are known, you can use measures like Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI) to evaluate the performance.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How would you do it if the labels aren’t known?**

        If the labels are known, you can use measures like Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI) to evaluate the performance.

    4. <span class="badge text-bg-secondary bg-danger">Hard</span> **Given the following dataset, can you predict how K-means clustering works on it? Explain.**

        ![png](/assets/images/machine-learning-interviews/2.svg){: .multiply }

        Unless the data contains another dimension where the samples can be clustered together, k-means clustering would not work well for this case since the concentric circles would have the same mean. If you know that the clusters will always be concentric circles, the radius can be used for clustering.

6. **k-nearest neighbor classification**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How would you choose the value of k?**

        The value of k in k-nearest neighbors (KNN) can be chosen by cross-validation, where different k values are tried, and the one with the best performance on the validation set is chosen.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens when you increase or decrease the value of k?**

        Increasing k makes the decision boundary smoother and more robust to noise but may increase bias. Decreasing k makes the decision boundary more sensitive to noise but may decrease bias.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How does the value of k impact the bias and variance?**

        The value of k affects the bias-variance tradeoff. Small values of k have low bias but high variance (more sensitive to noise), while larger values of k have higher bias but lower variance (more robust to noise).

7. **k-means and GMM are both powerful clustering algorithms.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **Compare the two.**

        k-means assumes clusters are spherical and have equal variance, while Gaussian Mixture Models (GMM) allow for elliptical clusters and can model clusters with different variances.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **When would you choose one over another?**

        k-means is typically faster and easier to understand, so it might be chosen for large datasets or as a preliminary analysis. GMM is more flexible but also more complex, so it might be chosen when we suspect the clusters have different shapes or variances.

8. **Bagging and boosting are two popular ensembling methods. Random forest is a bagging example while XGBoost is a boosting example.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **What are some of the fundamental differences between bagging and boosting algorithms?**

        Bagging reduces variance by averaging multiple models trained on different subsets of the data, while boosting reduces bias by training multiple models sequentially, each trying to correct the errors made by the previous model.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How are they used in deep learning?**

        In deep learning, bagging can be used by training multiple networks on different subsets of the data and averaging their predictions. Boosting is less commonly used in deep learning, as deep networks are typically complex enough to model the data without the need for boosting.

9. **Given this directed graph.**

    ![png](/assets/images/machine-learning-interviews/3.svg){: .multiply }

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Construct its adjacency matrix.**

        $$\begin{bmatrix}0 & 1 & 0 & 1 & 1\\0 & 0 & 1 & 1 & 0\\0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0\\\end{bmatrix}$$

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **How would this matrix change if the graph is now undirected?**

        The matrix would become symmetric. 

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **What can you say about the adjacency matrices of two isomorphic graphs?**

        Isomorphic graphs are graphs that have the same structure, even if the labels of the nodes are different. This means that there is a one-to-one correspondence between their vertices that preserves adjacency. Consequently, their adjacency matrices are similar; in other words, one can be obtained from the other by permuting rows and columns using the same permutation.

10. **Imagine we build a user-item collaborative filtering system to recommend to each user items similar to the items they’ve bought before.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **You can build either a user-item matrix or an item-item matrix. What are the pros and cons of each approach?**

        User-item matrices are more straightforward to implement and can recommend items similar to what a user has liked in the past, but they can suffer from sparsity if there are many items and not enough user interaction data. Item-item matrices can provide more robust recommendations by finding items similar to each other, but they may require more computation to update as new items are added.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **How would you handle a new user who hasn’t made any purchases in the past?**

        For a new user who hasn’t made any purchases, you could use a technique known as "cold start", where you recommend popular items or use demographic information about the user to provide initial recommendations.

11. <span class="badge text-bg-secondary bg-success">Easy</span> **Is feature scaling necessary for kernel methods?**

    Yes, feature scaling is necessary for kernel methods. Kernel methods, like SVM with an RBF kernel, calculate the distance between samples. Therefore, if features are not on the same scale, those with a larger scale can dominate the distance calculation, leading to suboptimal performance.

12. **Naive Bayes classifier**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How is Naive Bayes classifier naive?**

        The Naive Bayes classifier is considered 'naive' because it assumes that all features are independent of each other, which is rarely true in real-life scenarios.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Let’s try to construct a Naive Bayes classifier to classify whether a tweet has a positive or negative sentiment. We have four training samples:**

        | Tweet                           | Label    |
        |---------------------------------|----------|
        | This makes me so upset          | Negative |
        | This puppy makes me happy       | Positive |
        | Look at this happy hamster      | Positive |
        | No hamsters allowed in my house | Negative |
        {: .table .table-striped }
        
        According to your classifier, what's the sentiment of the sentence `The hamster is upset with the puppy`?

        
        In a Naive Bayes classifier, we use the probabilities of the features (in this case, words in a tweet) given the classes (here, positive and negative sentiment) to make our prediction. However, since Naive Bayes assumes independence between the features, we only need to calculate these probabilities for each word separately, then multiply them together to get the probability for the whole sentence.
        
        Let's try to make the prediction. The sentence is "The hamster is upset with the puppy". We disregard "the", "is", "with", as these are usually considered as stop words that don't carry much sentiment information. So we're left with "hamster", "upset", and "puppy".
        
        We calculate the probabilities as follows:
        
        1. The word "hamster" appears once in the positive class and once in the negative class. So the probability of "hamster" given positive, P(hamster\|Positive) = 1/7, and P(hamster\|Negative) = 1/8.
        2. The word "upset" appears only in the negative class. So P(upset\|Positive) = 0/7 = 0, and P(upset\|Negative) = 1/8.
        3. The word "puppy" appears only in the positive class. So P(puppy\|Positive) = 1/7, and P(puppy\|Negative) = 0/8 = 0.
        
        Now we calculate the overall probabilities for the sentence being positive or negative. We also need the probabilities of the classes themselves, which are both 0.5 in this case (two positive samples and two negative samples).
        
        P(Positive\|sentence) = P(hamster\|Positive) * P(upset\|Positive) * P(puppy\|Positive) * P(Positive) = 1/7 * 0 * 1/7 * 0.5 = 0.
        
        P(Negative\|sentence) = P(hamster\|Negative) * P(upset\|Negative) * P(puppy\|Negative) * P(Negative) = 1/8 * 1/8 * 0 * 0.5 = 0.
        
        Both probabilities are 0, because of the words "upset" and "puppy" that only appear in one class. This is a known problem with the Naive Bayes classifier called "zero frequency", and it's typically solved by using a smoothing technique, like Laplace smoothing.
        
        But even without applying smoothing, given the zero probabilities, and the nature of the words, we can assume the sentence is more likely to have a negative sentiment because of the presence of the word "upset". [TODO]
        

13. **Two popular algorithms for winning Kaggle solutions are Light GBM and XGBoost. They are both gradient boosting algorithms.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What is gradient boosting?**

        Gradient boosting is a machine learning technique that involves training a sequence of weak models, typically decision trees, each trying to correct the mistakes of the previous one.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What problems is gradient boosting good for?**

        Gradient boosting is particularly effective for regression tasks, binary classification tasks, and ranking tasks. It also works well when the data has complex interactions and non-linear relationships.

14. **SVM**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s linear separation? Why is it desirable when we use SVM?**

        Linear separation in SVM refers to the ability to separate two classes using a linear decision boundary (a straight line in 2D, a plane in 3D, etc.). It's desirable because it allows the SVM to find a clear margin of separation between classes.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How well would vanilla SVM work on this dataset?**

        ![png](/assets/images/machine-learning-interviews/4a.svg){: .multiply }

        It would work well. The two classes are linearly separable with a large margin.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How well would vanilla SVM work on this dataset?**

        ![png](/assets/images/machine-learning-interviews/4b.svg){: .multiply }

        The margin is smaller due to two close samples. The separating hyperplane has a much smaller margin that is closer to one set of samples than the other.

    4. <span class="badge text-bg-secondary bg-warning">Medium</span> **How well would vanilla SVM work on this dataset?**

        ![png](/assets/images/machine-learning-interviews/4c.svg){: .multiply }

        Even though the train error will be higher (the two sets of samples aren’t linearly separable), the line lines in a logical location that may result in a lower test error.

## Deep Learning Architectures and Applications

### Natural Language Processing

1. **RNNs**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the motivation for RNN?**

        Recurrent Neural Networks (RNNs) were motivated by the need to process sequential data where current output can be dependent on previous inputs. Conventional feedforward neural networks do not maintain any state between inputs. RNNs, on the other hand, retain a kind of 'memory' of previous inputs in their hidden layers.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the motivation for LSTM?**

        The motivation behind Long Short-Term Memory (LSTM) units is to address the vanishing gradient problem encountered by traditional RNNs, which makes it difficult for them to learn and maintain long-term dependencies in the sequence data.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How would you do dropouts in an RNN?**

        Dropout can be applied in RNNs but it should be done carefully. It can be applied to the input and output layers as usual. For the recurrent layers, it's often applied to the non-recurrent connections only. This is because applying dropout on the recurrent connections can hinder the ability of the RNN to learn long-term dependencies.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s density estimation? Why do we say a language model is a density estimator?**

    Density estimation is the construction of an estimate of the probability distribution for a dataset. A language model is a density estimator because it assigns probabilities to sequences of words or sentences, effectively estimating the probability distribution over the space of all possible sequences.

3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Language models are often referred to as unsupervised learning, but some say its mechanism isn’t that different from supervised learning. What are your thoughts?**

    Language models are often trained in an unsupervised way because they don't require explicit labels - the target output for each input is determined by the data itself (e.g., predicting the next word in a sequence). However, the learning mechanism can be seen as similar to supervised learning in that the model is learning to predict an output (the next word) based on input (the current word or sequence of words).

4. **Word embeddings.**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why do we need word embeddings?**

        Word embeddings are needed to represent words as continuous vectors, which capture the semantic meaning and relationships between words. This is useful for most NLP tasks, as it allows models to generalize from seen words to unseen words, and to understand semantic similarities between words.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What’s the difference between count-based and prediction-based word embeddings?**

        Count-based methods (like LSA) build a co-occurrence matrix and perform matrix factorization to generate embeddings, which effectively capture the global semantic information. Prediction-based methods (like Word2Vec) predict context given a word or vice versa, and are better at capturing local syntactic relations and certain semantic relations.

    3. <span class="badge text-bg-secondary bg-danger">Hard</span> **Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?**

        Context-based embeddings can struggle with words that have multiple meanings depending on context (polysemy), as they generate a single representation for each word. Also, they may fail to capture deeper or more abstract semantic relationships beyond those evident from local context.

5. **Given 5 documents:**

    - **D1: The duck loves to eat the worm**
    - **D2: The worm doesn’t like the early bird**
    - **D3: The bird loves to get up early to get the worm**
    - **D4: The bird gets the worm from the early duck**
    - **D5: The duck and the birds are so different from each other but one thing they have in common is that they both get the worm**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **Given a query Q: “The early bird gets the worm”, find the two top-ranked documents according to the TF/IDF rank using the cosine similarity measure and the term set {bird, duck, worm, early, get, love}. Are the top-ranked documents relevant to the query?**
    
        [TODO]

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Assume that document D5 goes on to tell more about the duck and the bird and mentions “bird” three times, instead of just once. What happens to the rank of D5? Is this change in the ranking of D5 a desirable property of TF/IDF? Why?**

        [TODO]

6. <span class="badge text-bg-secondary bg-success">Easy</span> **Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?**

    For a dataset with only 10,000 tokens, it's generally better to use an n-gram model. Neural language models typically require large amounts of data to perform well due to the complexity of the patterns they can learn.

7. <span class="badge text-bg-secondary bg-success">Easy</span> **For n-gram language models, does increasing the context length (n) improve the model’s performance? Why or why not?**

    Increasing the context length (n) can improve the model's ability to capture longer-term dependencies in the language, but it also drastically increases the model's complexity and data requirements. The number of possible n-grams grows exponentially with n, and many of them might not appear in the training data, leading to sparsity problems.

8. <span class="badge text-bg-secondary bg-warning">Medium</span> **What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?**

    One problem with using softmax in word-level language models is that it can be computationally expensive for large vocabularies. One solution is to use methods like hierarchical softmax or sampling techniques which reduce the computation. Another problem is that softmax outputs a probability distribution, which might not work well for tasks like word embedding where you might want the model to predict multiple plausible contexts.

9. <span class="badge text-bg-secondary bg-success">Easy</span> **What's the Levenshtein distance of the two words “doctor” and “bottle”?**

    The Levenshtein distance between “doctor” and “bottle” is 5. The operations are:
    - replace 'd' with 'b'
    - replace 'o' with 'o'
    - replace 'c' with 't'
    - replace 't' with 't'
    - insert 'l'
    - replace 'r' with 'e'

10. <span class="badge text-bg-secondary bg-warning">Medium</span> **BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?**

    Pros of BLEU include its simplicity, ease of use, and correlation with human judgement in many cases. Cons include its inability to consider the meaning of words (it's purely statistical), it doesn't work well for sentences that are too short or too long, and it can't evaluate the fluency of the translation.

11. <span class="badge text-bg-secondary bg-danger">Hard</span> **On the same test set, LM model A has a character-level entropy of 2 while LM model A has a word-level entropy of 6. Which model would you choose to deploy?**

    Entropy is a measure of uncertainty or randomness. Lower entropy suggests a better model because it implies less uncertainty in the predictions. However, comparing character-level and word-level entropy isn't straightforward, as the scales are different (there are far more possible words than characters). In general, though, both metrics are useful for model comparison and the choice might depend on the specific task and trade-offs between performance and complexity.

12. <span class="badge text-bg-secondary bg-warning">Medium</span> **Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?**

    Whether to make a Named Entity Recognition (NER) model case-sensitive or case-insensitive depends on the specifics of the corpus and the entities you are trying to recognize. If capitalization is an important clue for recognizing entities in your corpus (which is often the case), then a case-sensitive model would be more appropriate.

13. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why does removing stop words sometimes hurt a sentiment analysis model?**

    Removing stop words can sometimes hurt a sentiment analysis model because stop words can convey important sentiment information. For instance, negation words like "not" can significantly change the sentiment of a sentence, but might be removed as stop words.

14. <span class="badge text-bg-secondary bg-warning">Medium</span> **Many models use relative position embedding instead of absolute position embedding. Why is that?**

    Relative position embeddings are used because they allow the model to generalize better across different positions in the text. They help the model to understand the notion of 'distance' in terms of positions between words or tokens, which can be especially useful in tasks like understanding sentence structure or semantics.

15. <span class="badge text-bg-secondary bg-danger">Hard</span> **Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What’s the purpose of this?**

    Some NLP models, like the original Transformer model, use the same weights for the input embedding layer and the pre-softmax linear transformation to reduce the number of parameters and also based on a theoretical connection between the input and output embeddings in these models. Sharing these weights effectively constrains the model to map the input and output into the same vector space, which can improve performance on some tasks.

### Computer Vision

1. <span class="badge text-bg-secondary bg-warning">Medium</span> **For neural networks that work with images like VGG-19, InceptionNet, you often see a visualization of what type of features each filter captures. How are these visualizations created?**

    Visualization of filters in neural networks is done by applying the filter to a standardized input, like an image of random noise, then updating the input image to maximize the activation of the filter. This results in an image that represents what the filter is "looking for" in the input.

2. **Filter size**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **How are your model’s accuracy and computational efficiency affected when you decrease or increase its filter size?**

        Increasing the filter size increases the computational cost and the number of parameters in the model, potentially improving accuracy but also increasing the risk of overfitting. On the other hand, decreasing the filter size reduces computational cost and the number of parameters, which may lead to faster training but possibly less accurate models.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **How do you choose the ideal filter size?**

        The ideal filter size depends on the specific task and the scale of features that are important in the input images. Empirically, smaller filters (3x3 or even 1x1) have been found to work well in many situations.

3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Convolutional layers are also known as “locally connected.” Explain what it means.**

"Locally connected" refers to the fact that in a convolutional layer, each unit is connected only to a small region of the input, as opposed to a fully connected layer where each unit is connected to all inputs. This architecture is inspired by the organization of the visual cortex and helps to make the model more efficient and invariant to small translations.

4. <span class="badge text-bg-secondary bg-warning">Medium</span> **When we use CNNs for text data, what would the number of channels be for the first conv layer?**

    For text data represented as one-hot vectors, the number of channels in the first conv layer would typically be the size of the vocabulary.

5. <span class="badge text-bg-secondary bg-success">Easy</span> **What is the role of zero padding?**

    Zero padding is used to control the spatial dimensions (width and height) of the output volumes from the convolutional layers. This is useful for designing networks where we want the output volumes of the convolution layer to match the input dimensions.

6. <span class="badge text-bg-secondary bg-success">Easy</span> **Why do we need upsampling? How to do it?**

    Upsampling is used in CNNs to increase the spatial dimensions of the output. This is particularly useful in tasks like image segmentation where we need to produce an output of the same size as the input. Methods for upsampling include nearest neighbor, bilinear interpolation, transposed convolutions, and unpooling.

7. <span class="badge text-bg-secondary bg-warning">Medium</span> **What does a 1x1 convolutional layer do?**

    A 1x1 convolutional layer effectively acts as a pointwise fully connected layer across the depth of the input. It's often used to adjust the depth dimension of the inputs.

8. **Pooling**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens when you use max-pooling instead of average pooling?**

        When you use max-pooling instead of average pooling, the output is the maximum value from each pool instead of the average. This means the pooling operation is more sensitive to the presence of strong features.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **When should we use one instead of the other?**

        The choice between max pooling and average pooling depends on the specifics of your task. Max pooling tends to work better for tasks where the presence of a specific feature is more important than its absence.

    3. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens when pooling is removed completely?**

        If pooling is removed completely, the model would lose some of its ability to create hierarchical representations, potentially making it less effective at capturing complex patterns.

    4. <span class="badge text-bg-secondary bg-warning">Medium</span> **What happens if we replace a 2 x 2 max pool layer with a conv layer of stride 2?**

        Replacing a 2 x 2 max pool layer with a conv layer of stride 2 would maintain the spatial downsampling effect, but the conv layer would also be able to learn to extract features from its input, potentially leading to a more powerful model.

9. <span class="badge text-bg-secondary bg-warning">Medium</span> **When we replace a normal convolutional layer with a depthwise separable convolutional layer, the number of parameters can go down. How does this happen? Give an example to illustrate this.**

    In a depthwise separable convolutional layer, the convolution operation is split into a depthwise convolution and a pointwise convolution (1x1 convolution). This reduces the number of parameters because the depthwise convolution applies a single filter to each input channel, and the 1x1 convolution then merges the outputs. For example, for an input with 256 channels and a 3x3 filter, a traditional convolution would require 256 * 256 * 3 * 3 = 589,824 parameters, while a depthwise separable convolution would require 256 * 3 * 3 + 256 * 256 = 2,304 + 65,536 = 67,840 parameters.

10. <span class="badge text-bg-secondary bg-warning">Medium</span> **Can you use a base model trained on ImageNet (image size 256 x 256) for an object classification task on images of size 320 x 360? How?**

    Yes, you can use a model trained on ImageNet for an object classification task on images of size 320 x 360. You would need to adjust the input layer of the model to accept the new image size, and possibly also adjust the pooling or convolution strides to ensure that the spatial dimensions of the outputs are suitable.

11. <span class="badge text-bg-secondary bg-danger">Hard</span> **How can a fully-connected layer be converted to a convolutional layer?**

    A fully-connected layer can be seen as a convolutional layer with a kernel size that covers the entire input. So, to convert a fully-connected layer to a convolutional layer, you would set the filter size to be the same as the input size.

12. <span class="badge text-bg-secondary bg-danger">Hard</span> **Pros and cons of FFT-based convolution and Winograd-based convolution.**

    FFT-based convolution uses the Fast Fourier Transform to convert the convolution operation into a multiplication operation in the frequency domain. It can be faster than direct convolution for large input or filter sizes, but has a higher computational complexity for small sizes. It can also introduce rounding errors due to the transformations. Winograd-based convolution is a method that reduces the number of multiplications required in the convolution. It can be more efficient than direct or FFT-based convolution, but requires the filter size to be small. It also involves more complex implementation and may introduce rounding errors.

### Reinforcement Learning

1. <span class="badge text-bg-secondary bg-success">Easy</span> **Explain the explore vs exploit tradeoff with examples.**

    The explore-exploit tradeoff is a fundamental dilemma in Reinforcement Learning. Exploration involves the agent taking random actions to discover new states and rewards, whereas exploitation involves the agent taking the action that it currently believes to have the highest expected reward. Balancing these two is crucial. For example, in a restaurant recommendation system, exploration would involve trying new restaurants to gather more data, while exploitation would involve choosing the best restaurant known so far.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **How would a finite or infinite horizon affect our algorithms?**

    The horizon refers to the length of the future the agent is considering while making its decisions. A finite horizon means the agent only looks a fixed number of steps into the future, which might lead to short-term thinking, but simplifies computation. An infinite horizon means the agent considers all future steps while making a decision, which may lead to better long-term decisions but requires a discount factor to ensure that the sum of future rewards is finite.

3. <span class="badge text-bg-secondary bg-success">Easy</span> **Why do we need the discount term for objective functions?**

    The discount term in the objective function helps to model the preference for immediate rewards over delayed rewards and ensures that the sum of rewards in an infinite horizon problem is finite. This concept models real-world preferences where immediate rewards are often more valuable than delayed ones.

4. <span class="badge text-bg-secondary bg-success">Easy</span> **Fill in the empty circles using the minimax algorithm.**

    ![Minimax algorithm on game tree](/assets/images/machine-learning-interviews/5.svg){: .figure .rounded }

5. <span class="badge text-bg-secondary bg-warning">Medium</span> **Fill in the alpha and beta values as you traverse the minimax tree from left to right.**

    ![Alpha-beta pruning on game tree](/assets/images/machine-learning-interviews/6.svg){: .figure .rounded }

6. <span class="badge text-bg-secondary bg-success">Easy</span> **Given a policy, derive the reward function.**

    The reward function in Reinforcement Learning typically isn't derived from the policy---it's considered part of the environment that the agent is learning to interact with. A policy is a mapping from states to actions, indicating what action the agent will take in each state. The reward function, on the other hand, assigns a numerical value to each state or state-action pair, indicating the desirability of that state or state-action pair. While we can estimate the expected reward under a certain policy, we can't derive the reward function from the policy itself.

7. <span class="badge text-bg-secondary bg-warning">Medium</span> **Pros and cons of on-policy vs. off-policy.**

    On-policy methods learn about the current policy being used, while off-policy methods learn about a different policy using data from the current policy. Pros of on-policy include simplicity and the ability to provide good performance on the task at hand, while cons include the fact that it needs to continually explore to find the best policy. Off-policy methods, on the other hand, can learn from data generated by any policy, making them more flexible and able to learn from historic/batch data, but they can be more complex and might suffer from high variance.

8. <span class="badge text-bg-secondary bg-warning">Medium</span> **What's the difference between model-based and model-free? Which one is more data-efficient?**

    Model-based Reinforcement Learning methods build a model of the environment and use it to plan, while model-free methods learn a policy or value function directly from experience without requiring a model of the environment. Model-based methods are typically more data-efficient as they can "imagine" or simulate potential future states before experiencing them, but they might be less accurate if the model is wrong or incomplete. Model-free methods, on the other hand, do not rely on an approximation of the environment, and can learn directly from real experience, but they often require more data to learn effectively.

### Other

1. <span class="badge text-bg-secondary bg-warning">Medium</span> **An autoencoder is a neural network that learns to copy its input to its output. When would this be useful?**

    Autoencoders are useful for dimensionality reduction, anomaly detection, denoising, and learning latent representations of data. By learning to reproduce its input, an autoencoder essentially learns a compressed, distributed representation of the input. For example, in dimensionality reduction, the hidden layers of the autoencoder have fewer units than the input layer, forcing the autoencoder to learn a compressed representation of the input.

2. **Self-attention**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the motivation for self-attention?**

        The motivation for self-attention is to allow each token in an input sequence to interact with every other token, making it possible to capture long-range dependencies that RNNs and CNNs might struggle with.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why would you choose a self-attention architecture over RNNs or CNNs?**

        You might choose a self-attention architecture over RNNs or CNNs when the input sequences are long, and dependencies between distant tokens are important. Self-attention also allows for parallel computation across tokens in a sequence, which can be more efficient than the sequential computation of RNNs.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why would you need multi-headed attention instead of just one head for attention?**

        Multi-headed attention allows the model to focus on different aspects of the input for each attention head, potentially capturing a richer set of dependencies than a single attention head would.

    4. <span class="badge text-bg-secondary bg-warning">Medium</span> **How would changing the number of heads in multi-headed attention affect the model’s performance?**

        Changing the number of heads in multi-headed attention affects the capacity of the model to capture different types of relationships in the data. More heads can capture more complex relationships, but also increases computational complexity and the risk of overfitting.

3. **Transfer learning**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **You want to build a classifier to predict sentiment in tweets but you have very little labeled data (say 1000). What do you do?**

        With little labeled data, one approach is to use transfer learning: train a model on a larger labeled dataset in a similar domain, then fine-tune it on your small dataset. For sentiment analysis in tweets, you might fine-tune a model pre-trained on a large corpus of text, like the entire Wikipedia.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What’s gradual unfreezing? How might it help with transfer learning?**

        Gradual unfreezing is a technique where initially only the top layers of the pre-trained model are unfrozen and trained on the new task. Gradually, lower layers are unfrozen and fine-tuned. This helps prevent catastrophic forgetting of the pre-trained weights.

4. **Bayesian methods**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do Bayesian methods differ from the mainstream deep learning approach?**

        Bayesian methods differ from the mainstream deep learning approach in that they provide a probabilistic interpretation to the model's weights. Instead of learning a single best set of weights, a Bayesian neural network learns a distribution over possible weights.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How are the pros and cons of Bayesian neural networks compared to the mainstream neural networks?**

        Pros of Bayesian neural networks include their ability to provide uncertainty estimates and avoid overfitting. Cons include higher computational cost and complexity compared to mainstream neural networks.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why do we say that Bayesian neural networks are natural ensembles?**

        We say that Bayesian neural networks are natural ensembles because a prediction from a Bayesian neural network is equivalent to averaging predictions from many neural networks with weights sampled from the learned distribution.

5. **GANs**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What do GANs converge to?**

        In theory, GANs converge to a Nash equilibrium where the generator produces samples indistinguishable from real data, and the discriminator is unable to differentiate between the two. However, in practice, achieving this equilibrium is often difficult due to various challenges.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why are GANs so hard to train?**

        GANs are hard to train because of the adversarial nature of the training process. The generator and discriminator networks have conflicting objectives, which can lead to instability in training. Problems such as mode collapse, where the generator learns to produce a limited variety of samples, can also arise.

## Training Neural Networks

1. <span class="badge text-bg-secondary bg-success">Easy</span> **When building a neural network, should you overfit or underfit it first?**

    When building a neural network, it is generally better to overfit first, and then add regularization and tune hyperparameters to prevent overfitting. This confirms that the network is capable of fitting the data and the task is learnable.

2. <span class="badge text-bg-secondary bg-success">Easy</span> **Write the vanilla gradient update.**

    The vanilla gradient update for a parameter $$\theta$$: $$\theta = \theta - \textrm{learning rate} \times \textrm{gradient}$$

3. **Neural network in simple NumPy**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Write in plain NumPy the forward and backward pass for a two-layer feed-forward neural network with a ReLU layer in between.**
    
    ```python
    # Forward pass
    # Input: X, Weights: W1, W2, Biases: b1, b2
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = np.dot(A1, W2) + b2
    
    # Backward pass
    # Assume dZ2 is the derivative of the loss wrt Z2
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)  # derivative of ReLU
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)
    ```

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Implement vanilla dropout for the forward and backward pass in NumPy.**
    
    ```python
    # Forward pass with dropout
    # Assume dropout_rate and A1 from previous forward pass
    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=A1.shape)
    A1_dropped = A1 * dropout_mask
    
    # Backward pass with dropout
    # Assume dA1 from previous backward pass
    dA1_dropped = dA1 * dropout_mask
    ```
    

4. **Activation functions.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Draw the graphs for sigmoid, tanh, ReLU, and leaky ReLU.**

        - Sigmoid: Transforms input to a value between 0 and 1. Shape is an "S" curve. [TODO]
        - Tanh: Transforms input to a value between -1 and 1. Shape is an "S" curve.
        - ReLU: Outputs the input directly if it is positive, else, it will output zero.
        - Leaky ReLU: Similar to ReLU, but allows small negative values when the input is less than zero.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Pros and cons of each activation function**

        - Sigmoid:
            - Pros: Outputs are bound between 0 and 1, good for binary problems.
            - Cons: Vanishing gradients for very large/small values, not zero-centered.
        - Tanh:
            - Pros: Outputs are bound between -1 and 1, zero-centered.
            - Cons: Still has vanishing gradient problem.
        - ReLU:
            - Pros: Helps alleviate the vanishing gradient problem, computationally efficient.
            - Cons: Can result in dead neurons where outputs are consistently zero.
        - Leaky ReLU:
            - Pros: Prevents dead neurons by allowing small negative values.
            - Cons: Not always consistent improvement over ReLU.

    3. <span class="badge text-bg-secondary bg-success">Easy</span> **Is ReLU differentiable? What to do when it’s not differentiable?**

        ReLU is not differentiable at x = 0. But in practice, we can ignore this because it's only a single point, and we can arbitrarily define the derivative at x = 0 to be 0 or 1.

    4. <span class="badge text-bg-secondary bg-warning">Medium</span> **Derive derivatives for sigmoid function $$\sigma(x)$$ when $$x$$ is a vector.**

        Derivative of sigmoid: Let $$\sigma(x) = 1 / (1 + \exp(-x))$$. Then its derivative is $$\sigma(x) * (1 - \sigma(x))$$. When $$x$$ is a vector, you apply this derivative element-wise on the vector.

5. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the motivation for skip connection in neural works?**

    The motivation for skip connection (or residual connection) in neural networks is to facilitate the training of deep models by alleviating the problem of vanishing/exploding gradients. They allow gradients to flow directly through several layers by having identity shortcuts from earlier layers to later layers.

6. **Vanishing and exploding gradients.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **How do we know that gradients are exploding? How do we prevent it?**

        We know gradients are exploding when they become very large, often leading to numerical overflow (NaN values). It can be prevented using gradient clipping, where gradients are scaled down if they exceed a threshold.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **Why are RNNs especially susceptible to vanishing and exploding gradients?**

        RNNs are especially susceptible to vanishing and exploding gradients due to the recurrent nature of their architecture. The repeated multiplication in the backpropagation can make gradients either exponentially small (vanish) or large (explode).

7. <span class="badge text-bg-secondary bg-warning">Medium</span> **Weight normalization separates a weight vector’s norm from its gradient. How would it help with training?**

    Weight normalization helps with training by making the optimization landscape smoother and thus easier to navigate. By decoupling the direction of the weight vector from its magnitude, it can speed up convergence and improve generalization.

8. <span class="badge text-bg-secondary bg-warning">Medium</span> **When training a large neural network, say a language model with a billion parameters, you evaluate your model on a validation set at the end of every epoch. You realize that your validation loss is often lower than your train loss. What might be happening?**

    If validation loss is often lower than training loss, it might be due to dropout or other regularization techniques that are applied during training but not during validation.

9. <span class="badge text-bg-secondary bg-success">Easy</span> **What criteria would you use for early stopping?**

    Early stopping criteria might include a lack of improvement (or only minor improvements) on a validation set over a certain number of epochs, or reaching a predetermined maximum number of epochs.

10. <span class="badge text-bg-secondary bg-success">Easy</span> **Gradient descent vs SGD vs mini-batch SGD.**

    Gradient descent uses all training data for each update, SGD uses a single data point, and mini-batch SGD uses a subset of the data. SGD and mini-batch SGD are generally faster and can escape local minima better, but they have more noisy gradient estimates compared to gradient descent.

11. <span class="badge text-bg-secondary bg-danger">Hard</span> **It’s a common practice to train deep learning models using epochs: we sample batches from data without replacement. Why would we use epochs instead of just sampling data with replacement?**

    Epochs help ensure that each data point contributes roughly equally to the model's learning. Sampling with replacement could result in some data points being sampled many times and others not at all in a given pass over the data.

12. <span class="badge text-bg-secondary bg-warning">Medium</span> **Your model’ weights fluctuate a lot during training. How does that affect your model’s performance? What to do about it?**

    Fluctuating weights during training can indicate that the learning rate is too high or the optimization is unstable. This could lead to poor model performance. Reducing the learning rate or using techniques like gradient clipping or batch normalization can help.

13. **Learning rate**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **Draw a graph number of training epochs vs training error for when the learning rate is:**

        1. too high: the error may decrease rapidly initially but then fluctuate wildly and may even increase.
        2. too low: the error would decrease very slowly, which might take a very long time to converge or even fail to converge at all.
        3. acceptable: the error should decrease steadily and eventually converge, assuming the problem is well-posed and the model is suitable.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s learning rate warmup? Why do we need it?**

    Learning rate warmup is a procedure where the learning rate is initially set to a very small value and gradually increased to the standard learning rate. This is done to avoid large gradient updates at the beginning of training, which could lead to unstable training dynamics.

14. <span class="badge text-bg-secondary bg-success">Easy</span> **Compare batch norm and layer norm.**

    Batch normalization and layer normalization are both techniques to normalize the activations in a network, which can speed up learning and improve the final performance of the model. Batch normalization normalizes over the batch dimension, meaning that it computes the mean and variance for each feature over the batch of data. Layer normalization, on the other hand, normalizes over the feature dimension, meaning that it computes the mean and variance for each single data point separately.

15. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why is squared L2 norm sometimes preferred to L2 norm for regularizing neural networks?**

    The squared L2 norm is sometimes preferred over the L2 norm because its gradient is linear, while the gradient of the L2 norm depends on the weights themselves. This makes optimization more straightforward with the squared L2 norm. Additionally, the square operation emphasizes larger weights more than smaller ones, which can help drive smaller weights to zero.

16. <span class="badge text-bg-secondary bg-success">Easy</span> **Some models use weight decay: after each gradient update, the weights are multiplied by a factor slightly less than 1. What is this useful for?**

    Weight decay is a form of regularization that penalizes large weights. It can help prevent overfitting by encouraging the model to use smaller weights, which can lead to simpler models.

17. **It’s a common practice for the learning rate to be reduced throughout the training.**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s the motivation?**

        The motivation is that during the initial stages of training, when the weights are random, we can afford to make larger updates. But as we get closer to the optimal weights, we want to make smaller updates to avoid overshooting the minimum.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What might be the exceptions?**

        Exceptions might include scenarios where the model is not learning and a constant or even increasing learning rate might be necessary, or when using certain learning rate schedules or adaptive learning rate methods like Adam or Adagrad, where the learning rate is adjusted automatically.

18. **Batch size**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens to your model training when you decrease the batch size to 1?**

        If you decrease the batch size to 1, your model training will become more stochastic, which can make it harder to converge but could also prevent it from getting stuck in poor local minima.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **What happens when you use the entire training data in a batch?**

        If you use the entire training data in a batch, your model's learning would become a deterministic gradient descent. It could get stuck in sharp, poor minima instead of the flatter, generalizable minima.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How should we adjust the learning rate as we increase or decrease the batch size?**

        As a rule of thumb, if you increase the batch size, you can also increase the learning rate, since the gradient estimate will be more accurate. Similarly, if you decrease the batch size, you might need to decrease the learning rate to prevent instability due to the noisier gradient estimate.

19. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why is Adagrad sometimes favored in problems with sparse gradients?**

    Adagrad is favored in problems with sparse gradients as it adapts the learning rate for each weight in the model individually, which can be beneficial when dealing with sparse features where the importance of individual features might vary significantly.

20. **Adam vs. SGD**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **What can you say about the ability to converge and generalize of Adam vs. SGD?**

        Adam is generally faster to converge than SGD due to its adaptive learning rate for each parameter. However, some research suggests that SGD might generalize better than Adam, especially on larger datasets or deeper networks.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **What else can you say about the difference between these two optimizers?**

        Adam combines the benefits of two extensions of SGD—Adagrad and RMSprop. Unlike SGD, Adam uses the square gradients to scale the learning rate and it takes advantage of momentum by using moving average of the gradient instead of the gradient itself.

21. <span class="badge text-bg-secondary bg-warning">Medium</span> **With model parallelism, you might update your model weights using the gradients from each machine asynchronously or synchronously. What are the pros and cons of asynchronous SGD vs. synchronous SGD?**

    Asynchronous SGD allows for faster training as it doesn't have to wait for all workers to finish their computation before updating the model weights. However, it can lead to more stale gradients which can negatively impact convergence. Synchronous SGD can lead to better convergence as it uses up-to-date gradients, but it can be slower as it needs to wait for all workers to finish their computations.

22. <span class="badge text-bg-secondary bg-warning">Medium</span> **Why shouldn’t we have two consecutive linear layers in a neural network?**

    Two consecutive linear layers without a non-linear activation function in between are essentially equivalent to a single linear layer because the composition of two linear functions is another linear function. Therefore, having two consecutive linear layers does not increase the capacity of the model and is inefficient.

23. <span class="badge text-bg-secondary bg-warning">Medium</span> **Can a neural network with only RELU (non-linearity) act as a linear classifier?**

    A neural network with only ReLU activations can act as a non-linear classifier. ReLU introduces non-linearity into the model which allows it to learn more complex patterns.

24. <span class="badge text-bg-secondary bg-warning">Medium</span> **Design the smallest neural network that can function as an XOR gate.**

    The smallest neural network that can function as an XOR gate would have two inputs, a hidden layer with two neurons, and an output layer with one neuron. The activation function for the hidden layer neurons could be a non-linear function such as a sigmoid or ReLU.

25. <span class="badge text-bg-secondary bg-success">Easy</span> **Why don’t we just initialize all weights in a neural network to zero?**

    Initializing all weights to zero in a neural network is problematic because all neurons would become symmetric and learn the same features during training, which defeats the purpose of having multiple neurons in a layer. It essentially makes the network equivalent to a network with only one neuron per layer.

26. **Stochasticity**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **What are some sources of randomness in a neural network?**

        Some sources of randomness in a neural network include the initial weight initialization, the shuffling of data before each epoch, the usage of dropout during training, and the stochastic nature of the optimization algorithm (e.g., stochastic gradient descent).

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **Sometimes stochasticity is desirable when training neural networks. Why is that?**

        Stochasticity is desirable as it can help the model avoid getting stuck in poor local minima and achieve better generalization performance.

27. **Dead neuron**

    1. <span class="badge text-bg-secondary bg-success">Easy</span> **What’s a dead neuron?**

        A dead neuron is a neuron in a neural network that does not contribute to the final output, often because it always outputs the same value, usually zero in the context of ReLU activations.

    2. <span class="badge text-bg-secondary bg-success">Easy</span> **How do we detect them in our neural network?**

        Dead neurons can be detected by examining the outputs of the neurons. If a neuron's output is consistently zero for various inputs, it might be a dead neuron.

    3. <span class="badge text-bg-secondary bg-warning">Medium</span> **How to prevent them?**

        To prevent dead neurons, you can use leaky ReLU or parametric ReLU activation functions, which allow for small non-zero outputs when the input is negative. Good weight initialization strategies can also help prevent neurons from dying.

28. **Pruning**

    1. <span class="badge text-bg-secondary bg-warning">Medium</span> **Pruning is a popular technique where certain weights of a neural network are set to 0. Why is it desirable?**

        Pruning is desirable because it can reduce the size and complexity of a neural network without significantly impacting its performance. This makes the network more efficient to run, especially on hardware with limited resources.

    2. <span class="badge text-bg-secondary bg-warning">Medium</span> **How do you choose what to prune from a neural network?**

        To prune a neural network, you could start by identifying the weights that have the smallest magnitude or the ones that contribute the least to the output of the model according to some measure.

29. <span class="badge text-bg-secondary bg-danger">Hard</span> **Under what conditions would it be possible to recover training data from the weight checkpoints?**

    It may be possible to recover training data from the weight checkpoints if the model has been overfitted to the training data and the training dataset is small enough. However, this is generally very difficult and not a realistic concern for most deep learning models.

30. <span class="badge text-bg-secondary bg-danger">Hard</span> **Why do we try to reduce the size of a big trained model through techniques such as knowledge distillation instead of just training a small model from the beginning?**

    The reason to use techniques like knowledge distillation to reduce the size of a big trained model instead of training a smaller model from scratch is that the smaller model may not have the capacity to learn the same function as the larger model. The distillation process transfers knowledge from the larger model to the smaller one, allowing the smaller model to achieve similar performance with fewer parameters.