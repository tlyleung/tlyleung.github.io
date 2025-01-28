---
layout: note
title: Mathematics
description: High-level overview of the mathematics used in Machine Learning.
authors: [tlyleung]
x: 50
y: 60
---

<section class="relative mb-4 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-50 px-4 py-2 dark:bg-zinc-800" markdown="1">
<div class="absolute -top-2 right-4 h-16 w-16 text-zinc-200 dark:text-zinc-900">{% svg /assets/images/streamline/neural-swarm-1.svg width="100%" height="100%" %}</div>
# Mathematics

This cheatsheet attempts to give a high-level overview of the mathematics used in Machine Learning. Please [contact me](https://twitter.com/tlyleung) for corrections/omissions.

*Last updated: 1 January 2024*

</section>

<section class="relative mb-4 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-50 px-4 py-2 dark:bg-zinc-800" markdown="1">
<div class="absolute -top-2 right-4 h-16 w-16 text-zinc-200 dark:text-zinc-900">{% svg /assets/images/streamline/book-flip-page.svg width="100%" height="100%" %}</div>
# Contents

- [Probability](#probability)
    - [Conditional Probability](#probability-conditional-probability)
    - [Random Variables](#probability-random-variables)
- [Linear Algebra](#linear-algebra)
    - [Products](#linear-algebra-products)
    - [Operations & Properties](#linear-algebra-operations--properties)
</section>

<section class="relative mb-4 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-50 px-4 py-2 dark:bg-zinc-800" markdown="1">
# Probability[^prob2019]

Probabilities arise when we perform an experiment:

- The set of all possible outcomes of the experiment is the *sample space*.
- A subset of the sample space is an *event*.
- The *union* $$(A \cup B)$$ is the event that occurs if and only if *at least one* of $$A$$, $$B$$ occurs.
- The *intersection* $$(A \cap B)$$ is the event that occurs if and only if *both* $$A$$ and $$B$$ occur.
- The *complement* $$(A^C)$$ is the event that occurs if and only if $$A$$ *does not* occur.

---

## Combinatorics

- **Multiplication Rule:** The total outcomes of two sub-experiments with $$a$$ and $$b$$ outcomes are $$ab$$.

- **Sampling with Replacement:** Choosing $$k$$ times from $$n$$ objects *with replacement* gives $$n^k$$ outcomes.

- **Sampling without Replacement:** Choosing $$k$$ times from$$n$$ objects *without replacement* gives $$n(n-1)\cdots(n-k+1)$$ outcomes.

- **Permutations:** There are $$n!$$ permutations of $$n$$ distinct items, reflecting all possible orderings.

- **Binomial Coefficient Formula**: The binomial coefficient $$\binom{n}{k}$$ is $$\frac{n(n-1)\ldots(n-k+1)}{k!} = \frac{n!}{(n-k)!k!}$$ for $$k \leq n$$, and 0 otherwise.

---

## Axioms

1. For any event $$A$$, $$P(A) \geq 0$$ (non-negativity).
2. Probability of sample space $$S$$ is $$P(S) = 1$$ (completeness).
3. If $$A_1, A_2, \ldots$$ are disjoint events, then $$P(A_1 \cup A_2 \cup \ldots) = P(A_1) + P(A_2) + \ldots$$ (countable additivity).

---

## Consequences

### Monotonicity

If $$A \subseteq B$$, then $$P(A) \leq P(B)$$.

### Probability of the Empty Set

$$P(\emptyset) = 0$$

### Complement Rule

$$P(A^C) = 1 - P(A)$$

### Inclusion-Exclusion Formula $$(n=2)$$

$$P(A_i \cup A_j) = P(A_1) + P(A_2) - (A_i \cap A_j)$$

### Inclusion-Exclusion Formula

$$\begin{eqnarray}
P\left(\bigcup_{i=1}^{n} A_i\right) &=& \sum_{i} P(A_i) - \sum_{i<j} P(A_i \cap A_j) \\
                                    && + \sum_{i<j<k} P(A_i \cap A_j \cap A_k) - \cdots \\
                                    && + (-1)^{n+1} P(A_1 \cap \cdots \cap A_n)
\end{eqnarray}$$

</section>

<section class="relative mb-4 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-50 px-4 py-2 dark:bg-zinc-800" markdown="1">
# Probability: Conditional Probability[^prob2019]

The conditional probability of $$A$$ given $$B$$ is:

- $$P(A \vert B) = \frac{A \cap B}{P(B)}$$

- $$P(A)$$ is the prior probability of $$A$$

- $$P(A \vert B)$$ is the posterior probability of $$A$$

### Probability of the Intersection of Two Events

$$P(A \cap B) = P(B) P(A \vert B) = P(A) P(B \vert A)$$

### Probability of the Intersection of $$n$$ Events

$$P(A_1, A_2, \ldots) = P(A_1)P(A_2 \vert A_1)\ldots P(A_n \vert A_1, \ldots A_{n-1})$$

### Bayes' Rule

Update the probability of a hypothesis based on new evidence, balancing prior beliefs with the likelihood of observed data.

$$P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}$$

### Law of Total Probability

Calculate the overall probability of an event by considering all possible ways it can occur across different conditions.

$$P(B) = \sum^{n}_{i-1} P(B \vert A_i)P(A_i)$$

---

## Independent Events

Independent events provide no information about each other.

### Independence of Two Events

Events $$A$$ and $$B$$ are independent if $$P(A \cap B) = P(A)P(B)$$.

### Independence of $$n$$ Events

Events $$A_1, A_2, \ldots A_n$$ are independent if any pair satisfies $$P(A_i \cap A_j) = P(A_i)P(A_j)$$ for $$i \neq j$$, any triplet satisfies $$P(A_i \cap A_j \cap A_k) = P(A_i)P(A_j)P(A_k)$$ for $$i \neq  j \neq  k$$, and similarly for all quadruplets, quintuplets and so on.

---

## Common Mistakes

- Confusing the prior probability $$P(A)$$ with the posterior probability $$P(A \vert B)$$.
- Confusing $$P(A \vert B)$$ with $$P(B \vert A)$$ (the prosecutor's fallacy).
- Failing to condition on *all* the evidence (the defense attorney's fallacy).
- Overlooking a scenario where a trend in individual groups disappears or reverses when the groups are combined (Simpson's paradox).

</section>

<section class="relative mb-4 break-inside-avoid-column overflow-hidden rounded-md bg-zinc-50 px-4 py-2 dark:bg-zinc-800" markdown="1">
# Probability: Random Variables[^prob2019]

A random variable is a function assigning a real number to every possible outcome of an experiment.

### Probability Mass Function (PMF)

- The probability mass function (PMF) of a discrete random variable $$X$$ is the function $$p(x) = P(X=x)$$.
- For a PMF to be valid, it must be non-negative and sum to 1.

### Probability Density Function (PDF)

- The probability density function (PDF) of a continuous random variable $$X$$ is the derivative $$f(x) = F'(x)$$ of the CDF.
- For a PDF to be valid, it must be $$f(x) \geq 0$$ for all $$x$$ and $$\int_{-\infty}^{\infty} f(x)dx = 1$$.
- To obtain a probability, we need to integrate the PDF.

### Cumulative Distribution Function (CDF)

- The cumulative distribution function (CDF) of a random variable $$X$$ is the function $$F(x) = P(X \leq x)$$.
- For a CDF to be valid, it must be continuous, right-continuous, converge to 0 as $$x \rightarrow -\infty$$ and converge to 1 as $$x \rightarrow \infty$$.

---

## Independence

### Independence of two random variables

$$P(X \leq x, Y \leq y) = P(X \leq x)P(Y \leq y)$$

### Independence of $$n$$ random variables

$$P(X_1 \leq x_1, \ldots, X_n \leq x_n) = P(X_1 \leq x_1) \ldots P(X_n \leq x_n)$$

### Independent and identically distributed (IID) random variables

Random variables are *independent* if they provide no information about each other and are *identically distributed* if they have the same PMF or CDF.

---

## Expectation

### Expectation

Center of mass of a distribution.

$$E(X) = \sum_x x P(X=x)$$

### Variance

Spread of a distribution around its mean.

$$\textrm{Var}(X) = E(X - E(X))^2 = E(X^2) E^2(X)$$

### Standard Deviation

Average distance of each data point from the mean.

$$\textrm{Stdev}(X) = \sqrt{\textrm{Var}(X)}$$

### Law of the Unconscious Statistician (LOTUS)

Expectation of $$g(X)$$ can be calculated using only the PMF of $$X$$, without first having the find the distribution of $$g(x)$$.

$$E(g(X)) = \sum_x g(x)P(X=x)$$

### Covariance

Covariance is the tendency of two random variables to go up or down together, relative to their means.

$$\textrm{Cov}(X, Y) = E((X-E(X))(Y-E(Y))) = E(XY) - E(X)E(Y)$$

### Correlation

Like Covariance but scaled to be unitless with values always ranging from -1 to 1.

$$\textrm{Corr}(X, Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$$

### Law of Large Numbers

The mean of a large number of independent random samples converges to the true mean.

### Central Limit Theorem

The sum of a large number of IID random variables has an approximately Normal distribution, regardless of the distribution of the individual random variables.

---

## Discrete Random Variables

### Bernoulli Distribution, $$X \sim \textrm{Bern}(p)$$

Indicator of success in a Bernoulli trial with probability of success $$p$$.

$$P(X=0) = 1-p$$

$$P(X=1) = p$$

### Binomial Distribution, $$X \sim \textrm{Bin}(n, p)$$

Number of successes in $$n$$ independent Bernoulli trials, all with the same probability $$p$$ of success.

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$ for $$k = 0, 1, \ldots, n$$

### Hypergeometric Distribution, $$X \sim \textrm{HGeom}(w, b, n)$$

Number of white balls obtained in a sample of size $$n$$ drawn without replacement from an urn of $$w$$ white and $$b$$ black balls.

$$P(X=k) = \frac{\binom{w}{k}\binom{b}{n-k}}{\binom{w+b}{n}}$$

### Discrete Uniform Distribution, $$X \sim \textrm{DUnif}(C)$$

Obtained by randomly choosing an element of the finite set $$C$$, with equal probabilities for each element.

$$P(X=x) = \frac{1}{|C|}$$

### Geometric Distribution, $$X \sim \textrm{Geom}(p)$$

Number of failures before the first success in a sequence of independent Bernoulli trials with probability $$p$$ of success.

### Negative Binomial Distribution, $$X \sim \textrm{NBin}(r, p)$$

Number of failures before $$r$$ successes with replacement.

$$P(X=n) = \binom{n+r-1}{r-1} p^r (1-p)^{n}$$ for $$n = 0, 1, \ldots$$

### Negative Hypergeometric Distribution, $$X \sim \textrm{NHGeom}(w, b, r)$$

Number of failures before $$r$$ successes without replacement.

$$P(X=k) = \frac{\binom{w}{r-1}\binom{b}{k}}{\binom{w+b}{r+k-1}} \cdot \frac{w-r+1}{w+b-r-k+1}$$

### Poisson Distribution, $$X \sim \textrm{Pois}(\lambda)$$

Number of events occurring in a fixed interval of time, given arrival rate $$\lambda$$.

$$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$$ for $$k = 0, 1, \ldots, $$

---

## Continuous Random Variables

### Uniform Distribution, $$X \sim \textrm{Unif}(a, b)$$

Completely random number in the interval $$(a, b)$$.

$$f(x) = \frac{1}{b-a}$$

### Normal Distribution, $$X \sim \mathcal{N}(\mu, \sigma^2)$$

A symmetric bell-shaped distribution centered at $$\mu$$, with $$\sigma$$ determining the spread.

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp \left( -\frac{(x-\mu)^2}{2\sigma^2} \right)$$

### Exponential Distribution, $$X \sim \textrm{Expo}(\lambda)$$

Waiting time for the first success in continuous time, with arrival rate $$\lambda$$.

$$f(x) = \lambda e^{\lambda x}$$

Note: the exponential distribution has the memoryless property; given a certain wait time without success, the distribution of the remaining wait time is unchanged, i.e. $$P(X \geq s + t \vert X \geq s) = P (X \geq t)$$.

</section>

<section class="mb-4 px-4 py-2 break-inside-avoid-column rounded-md bg-zinc-50 dark:bg-zinc-800" markdown="1" style="/assets/images/streamline/rotate-angle.svg">

# Linear Algebra

## Vector Addition

$$u + v = \left( a_1 + b_1, a_2 + b_2, \ldots, a_n + b_n \right)$$

### Example

Given $$u = \begin{bmatrix} 3 \\ -5 \end{bmatrix}$$ and $$v = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$, $$u + v = \begin{bmatrix} 3 \\ -5 \end{bmatrix} + \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 + 2 \\ -5 + 1 \end{bmatrix} = \begin{bmatrix} 5 \\ -4 \end{bmatrix}$$.

## Scalar Multiplication

$$ku = \left( ka_1, ka_2, \ldots, ka_n \right)$$

### Example

Given $$u = \begin{bmatrix} 3 \\ -5 \end{bmatrix}$$, $$2u = 2 \begin{bmatrix} 3 \\ -5 \end{bmatrix} = \begin{bmatrix} 2 \cdot 3 \\ 2 \cdot -5 \end{bmatrix} = \begin{bmatrix} 6 \\ -10 \end{bmatrix}$$.

## Linear Combination

$$k_1 u_1 + k_2 u_2 + \ldots + k_n u_n$$

### Example

Given $$u = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$ and $$v = \begin{bmatrix} 2 \\ -1 \end{bmatrix}$$, $$2u - 3v =  2 \begin{bmatrix} 4 \\ 1 \end{bmatrix} - 3 \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \\ 5 \end{bmatrix}$$.

## Linear Independence

A set of vectors is linearly independent if no vector can be represented as a linear combination of the remaining vectors.

## Basis

Standard unit vectors $$\widehat{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$ and $$\widehat{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$ are basis vectors of the $$x\textrm{-}y$$ coordinate system since any 2D vector can be expressed as a linear combination of them.

## Span

The span of two vectors is the set of all vectors that can be formed from their linear combinations.

## Linear Transformation

A linear transformation is fully described by where the two standard basis vectors land. This transformation can be captured by a 2×2 matrix: the first column is where $$\widehat{i}$$ lands, and the second column is where $$\widehat{j}$$ lands. Visually, a linear transformation keeps grid lines parallel and evenly spaced and the origin fixed.

### Example

Rotate 90° counter-clockwise

$$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

### Example

Shear

$$\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

## Rank

- The column rank of a matrix is the largest number of columns that constitute a linearly independent set.
- The row rank of a matrix is the largest number of rows that constitute a linearly independent set.
- Since the column rank and row rank are always equal, the rank of a matrix is the largest number of rows or columns that constitute a linearly independent set.

### Properties

- $$\textrm{rank}(A) \leq min(m, n)$$
- $$\textrm{rank}(A) = \textrm{rank}(A^T)$$
- $$\textrm{rank}(AB) \leq min(rank(A), rank(B))$$
- $$\textrm{rank}(A+B) \leq rank(A) + rank(B)$$

</section>

<section class="mb-4 px-4 py-2 break-inside-avoid-column rounded-md bg-zinc-50 dark:bg-zinc-800" markdown="1" style="/assets/images/streamline/rotate-angle.svg">

# Linear Algebra: Products

## Dot (Inner) Product

$$u \cdot v = a_1 b_1 + a_2 b_2 + \ldots + a_n b_n$$

### Properties

- Orthogonality: $$u \cdot v = 0$$
- Angle between vectors: $$\cos \theta = \frac{v \cdot w}{\|v\| \|w\|}$$
- Projection: $$proj(u, v) = \frac{u \cdot v}{\|v\|^2} v = \frac{u \cdot v}{v \cdot v} v$$
- Schwarz inequality: $$\lvert v \cdot w \rvert \leq \|v\| \|w\|$$
- Triangle inequality: $$\|v + w\| \leq \|v\| + \|w\|$$

### Example

Given $$u = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$ and $$v = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$, $$u \cdot v = (1)(4) + (2)(6) = 4 + 12 = 16$$.

### Example

Given $$u = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$ and $$v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$, $$\cos \theta = \frac{u \cdot v}{\|u\| \|v\|} = \frac{1}{(1)(\sqrt{2})}$$, which means $$\theta = 45^\circ$$.

## Cross Product

The cross product between two vectors gives a third vector whose length is the parallelogram's area and points in a direction that is orthogonal to both vectors, following the right-hand rule.

$$u \times v = \begin{vmatrix} i & j & k \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}$$

### Properties

- $$\|u \times v\| = \|u\| \|v\| \sin \theta$$
- $$u \times u = 0$$
- $$u \times v = -(v \times u)$$
- $$u \times (v + w) = (u \times v) + (u \times w)$$

## Outer Product

Each element of $$u$$ is multiplied by each element of $$v$$.

$$u \otimes v = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \ldots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \ldots & a_2 b_n \\ \vdots & \vdots & \ddots & \vdots \\ a_m b_1 & a_m b_2 & \ldots & a_m b_n \end{bmatrix}$$

### Properties

- $$(u \otimes v)^T = (v \otimes u)$$
- $$(v + w) \otimes u = v \otimes u + w \otimes u$$
- $$u \otimes (v + w) = u \otimes v + u \otimes w$$
- $$c (v \otimes u) = (cv) \otimes u = v \otimes (cu)$$

</section>

<section class="mb-4 px-4 py-2 break-inside-avoid-column rounded-md bg-zinc-50 dark:bg-zinc-800" markdown="1" style="/assets/images/streamline/rotate-angle.svg">

# Linear Algebra: Operations & Properties[^cs229linear]

## Identity Matrix

A square matrix with ones on the diagonal and zeros everywhere else.

$$I_{ij} = \begin{cases}1 & \text{if } i = j \\ 0 & \text{if } i \neq j\end{cases}$$

### Example

$$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

---

## Diagonal Matrix

A square matrix where all non-diagonal elements are zero.

$$D_{ij} = \begin{cases}d_i & \text{if } i = j \\ 0 & \text{if } i \neq j\end{cases}$$

### Example

$$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}$$

---

## Transpose

Flip the rows and columns.

$$A_{ij}^{T} = A_{ji}$$

### Properties

- $$\left(A^T\right)^T = A$$
- $$\left(AB\right)^T = B^TA^T$$
- $$\left(A+B\right)^T = A^T + B^T$$

### Example

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} = \begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix}^T$$

---

## Symmetric Matrix

$$A = A^T$$

### Example

$$\begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix}^T = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix}$$

---

## Anti-symmetric Matrix

$$A = -A^T$$

### Example

$$\begin{bmatrix} 0 & 1 & -2 \\ -1 & 0 & -1 \\ 2 & 1 & 0 \end{bmatrix} = -\begin{bmatrix} 0 & 1 & -2 \\ -1 & 0 & -1 \\ 2 & 1 & 0 \end{bmatrix}^T$$

---

## Triangular Matrix

A square matrix is called lower triangular if all the entries above the main diagonal are zero. Similarly, a square matrix is called upper triangular if all the entries below the main diagonal are zero. 

### Example

$$L = \begin{bmatrix} 1 & 0 & 0 \\ 4 & 5 & 0 \\ 7 & 8 & 9 \end{bmatrix}$$

$$U = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 5 & 6 \\ 0 & 0 & 9 \end{bmatrix}$$

---

## Orthogonal Matrix

A square matrix $$Q$$ such that its transpose is equal to its inverse.

$$ Q^T Q = Q Q^T = I $$

### Example

$$ Q = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{bmatrix} $$

---

## Normal Matrix

A square matrix $$A$$ is called a normal matrix if it commutes with its conjugate transpose.

$$ AA^* = A^*A $$

### Example

Given the matrix:

$$A = \begin{bmatrix} 1 & i \\ -i & 1 \end{bmatrix}$$

The conjugate transpose $$A^*$$ is:

$$A^* = \begin{bmatrix} 1 & -i \\ i & 1 \end{bmatrix}$$

---

## Trace

Sum of diagonal elements in the matrix.

$$\textrm{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

### Properties

- $$\textrm{tr}(A) = \textrm{tr}(A)^T$$
- $$\textrm{tr}(A+B) = \textrm{tr}(A) + \textrm{tr}(B)$$
- $$\textrm{tr}(kA) = k \textrm{tr}(A)$$

### Example

$$\textrm{tr}\left(\begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}\right) = 1 + 2 + 3 = 6$$

---

## Norm

Length of a vector.

$$\|u\| = \sqrt{u \cdot u} = \sqrt{a_1^2 + a_2^2 + \ldots + a_n^2}$$

$$u$$ is a unit vector if it has length 1, i.e., $$\|u\| = 1$$

### Properties

1. For all $$x \in \mathbb{R}^n, f(x) \geq 0$$ (non-negativity).
2. $$f(x) = 0$$ if and only if $$x = 0$$ (definiteness).
3. For all $$x \in \mathbb{R}^n, t \in \mathbb{R}, f(tx) - \lvert t \rvert f(x)$$ (homogeneity).
4. For all $$x,y \in \mathbb{R}^n, f(x+y) \leq f(x) + f(y)$$ (triangle inequality).

### $$\ell_1$$ Norm

$$\|x\|_1 = \sum_{i=1}^{n}\lvert x_i \rvert$$

### $$\ell_2$$ Norm

$$\|x\|_2 = \sqrt{\sum_{i=1}^{n}x_i^2}$$

### $$\ell_\infty$$ Norm

$$\|x\|_\infty = \max_i \lvert x_i \rvert$$

### $$\ell_p$$ Norm

$$\|x\|_p = \left( \sum_{i=1}^{n} \lvert x_i \rvert^p \right)^{1/p}$$

---

## Inverse Matrix

The inverse of a square matrix $$A$$ is denoted $$A^{-1}$$ and is the unique matrix such that: $$A^{-1}A = I$$.

Not all matrices have inverses. $$A$$ is invertible or non-singular if the inverse exists and non-invertible or singular otherwise.

### Properties

- $$\left(A^{-1}\right)^{-1} = A$$
- $$(AB)^{-1} = B^{-1}A^{-1}$$
- $$\left(A^{-1}\right)^T = \left(A^T\right)^{-1}$$

---

## Determinant

The determinant of a square matrix can be understood as the volume scaling factor for the linear transformation described by the matrix.

For any matrix, the absolute value of the determinant gives the volume of the shape formed by its column vectors: a parallelogram in 2D, a parallelepiped in 3D, and an $$n$$-dimensional parallelotope in higher dimensions.

The absolute value of the determinant, combined with its sign, represents the oriented area (or volume in higher dimensions), indicating whether the transformation preserves or reverses orientation.

A determinant of zero means the matrix is not invertible, as the linear transformation collapses the original space into a lower dimension with zero volume (such as a flat plane, a line, or a point). Once space is compressed in this way, it cannot be expanded back to its original form. For example, you cannot "unsquish" a line back into a plane, or a plane into a volume.

### Properties

1. $$\| I \| = 1$$.
2. The determinant of a matrix, where one of its columns is a linear combination of other vectors, is also a linear combination of those vectors.
3. Whenever two columns of a matrix are identical, its determinant is 0.

### Example

$$\begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad-bc$$

### Example

$$\begin{vmatrix} a & b & c \\ d & e & f \\ g & h & i \end{vmatrix} = a \begin{vmatrix} e & f \\ h & i \end{vmatrix} - b \begin{vmatrix} d & f \\ g & i \end{vmatrix} + c \begin{vmatrix} d & e \\ g & h \end{vmatrix}$$

---

## Eigenvectors and Eigenvalues

Given a square matrix $$A$$, we say that $$\lambda$$ is an eigenvalue of $$A$$ and $$x$$ is the corresponding eigenvector if $$Ax = \lambda x$$. In other words, when the matrix $$A$$ is applied to the vector $$x$$, the result is a new vector that points in the same direction as $$x$$, but scaled by a factor of $$\lambda$$.

In a typical linear transformation, most vectors are "knocked off" the line they span, meaning they change direction. However, the vectors that remain along the same line after the transformation are called eigenvectors of that transformation. The scaling factor by which these vectors are stretched or compressed is the corresponding eigenvalue. In 3D, this can be visualized as an axis of rotation.

### Properties

- $$\textrm{tr}(A) = \sum_{i=1}^{n} \lambda_i$$
- $$\lvert A \rvert = \prod_{i=1}^{n} \lambda_i$$

</section>

[^3blue1brown]: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
[^cs229linear]: [Linear Algebra Review and Reference](https://cs229.stanford.edu/lectures-spring2022/cs229-linear_algebra_review.pdf)
[^cs229prob]: [Probability Theory Review and Reference](https://cs229.stanford.edu/lectures-spring2022/cs229-probability_review.pdf)
[^prob2019]: [Introduction to Probability, Second Edition.](http://probabilitybook.net)
