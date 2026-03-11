This document gives an overview of the following topics:
- Classification and logistic regression
- Entropy and cross-entropy
- Regularization
- Batch size 

---

<details markdown="block">
<summary> Binary classification and the logistic function </summary>

## Understanding how loss is defined in regression and classification 

Since linear regression predicts a continuous value (score) that is unbounded, it is not adequate for *classification* problems. Let's consider the simple case of a *binary classification problem*, e.g. predict the sex from other variables for the Penguin data set. 

In that case, the are just two classes *female* and *male*, and we only need to consider one of them in the model. What we intend to predict is the **probability** that a new example is of class *female* (if it is, say, 78% then the probability of being of class *male* will just be 22%). 

We need a function that converts the output of the linear regression $y$ into a probability. This is done by the logistic function:

$$\sigma(y)=\frac{1}{1+e^{-y}}.$$

For instance, for 3 predictor variables, the full  *logistic regression* prediction model for binary classification is the following:

$$\hat{p}= \frac{1}{1+e^{-(w_0 + w_1 \\, x_1 +  w_2 \\, x_2 + w_3 \\, x_3)}}.$$

where $\hat{p}$ is the estimated probability that the example belongs to the class. Let's consider the example above where the class is *female*. If $\hat{p} \ge 0.5$ then, the predicted class $\hat{y}$ is *female*. Otherwise, the predicted class $\hat{y}$ is *male*. To fit the model, one also solves the ML minimization problem where the loss function depends for each example on the true label $y \in \{0,1\}$ and on the predicted probability $\hat{p} \in [0,1]$. 

The loss function should be low if $\hat{p}$ is close to 1 and the true class is *female* or if $\hat{p}$ is close to 0 and the true class is *male*. This typical loss function for this kind of problem is known as *log loss*,  *binary cross-entropy* or *logistic loss* and it is defined by $L=-[y {\rm log}(\hat{p}) + (1-y) {\rm log}(1-\hat{p})]$ where $y$ is the true label.

</details>

---

<details markdown="block">
<summary> Entropy and cross-entropy </summary>

In machine learning, **Entropy** and **Cross-Entropy** are the tools we use to measure "uncertainty" and "error" and are used to guide the training of many ML models.

### Entropy: the measure of surprise

Entropy originates from Information Theory. It measures the amount of **disorder** or **uncertainty** in a dataset.

- **High Entropy:** If you have a bag with 50 blue marbles and 50 red marbles, the next pick is highly uncertain. This is high entropy.
- **Low Entropy:** If you have a bag with 99 blue marbles and 1 red marble, you are almost certain the next pick will be blue. This is low entropy. The amount of suprise is very low if one pick a blue marble and it is very high if we pick a red marble. But the overall surprise is still lower than in the 50-50 case.

In model training, we calculate entropy using the probability $p(x)$ of each class:

$$H(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)$$

### Cross-Entropy: the measure of distance between distributions

While entropy measures a single distribution, **Cross-Entropy** compares two distributions:

1. The **True** distribution (the actual labels, like $[1, 0]$ for "is a cat").
2. The **Predicted** distribution (what your model thinks, like $[0.8, 0.2]$).

It measures how many "bits" of information are needed to identify an event from the true distribution using the predicted distribution. The closer the prediction is to the truth, the lower the Cross-Entropy.

During training of classification models, cross-entropy is a typical choice for the **loss function**. The optimizer (like SGD) works to find the best model weights that minimize this value.

### Comparison

| Concept | What it answers | Formula Context |
| --- | --- | --- |
| **Entropy** | How much uncertainty for this variable? | Uses only $p(x)$ |
| **Cross-Entropy** | How far off is my prediction ($q$) from the truth ($p$)? | Uses both $p(x)$ and $q(x)$ |


One could consider using accuracy as the loss function for classification. However, 
accuracy is "all or nothing"—it doesn't care if the model was *almost* right or *wildly* wrong. Cross-Entropy is a continuous gradient; it heavily penalizes a model that is **confident and wrong**.

</details>

---

<details markdown="block">
<summary> Regularization to avoid overfitting</summary>

## Understanding Regularization in Machine Learning

At its core, **regularization** is a technique used to "tame" a machine learning model. When a model is too complex, it tends to memorize the noise in the training data rather than learning the underlying patterns—a phenomenon known as **overfitting**. By introducing a penalty for complexity, regularization ensures the model remains simple enough to generalize well to new, unseen data.

### Adding a penalty to the Loss Function

In standard regression, we aim to minimize the residual sum of squares. Regularization modifies this objective by adding a **penalty term**:

$$\text{Total Loss} = \text{Loss}(\text{Actual, Predicted}) + \lambda \times \text{Complexity Penalty}$$

The hyperparameter $\lambda$ (lambda) controls the strength of the regularization. A higher $\lambda$ increases the penalty, leading to simpler models.


### Primary Regularization Techniques

There are two main ways to penalize model coefficients, each with distinct effects on the final model:

#### L1 Regularization (Lasso Regression)

Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty equal to the **absolute value** of the magnitude of coefficients.

* **Effect:** It can force some coefficients to become exactly zero.
* **Use Case:** Ideal for **feature selection**, as it effectively removes irrelevant variables, resulting in a sparse and interpretable model.

#### L2 Regularization (Ridge Regression)

Ridge adds a penalty equal to the **square** of the magnitude of coefficients.

* **Effect:** It shrinks the coefficients toward zero but rarely makes them exactly zero.
* **Use Case:** Best when you have many features that all contribute slightly to the output, or when features are highly correlated (**multicollinearity**).

#### Elastic Net

Elastic Net is a middle-ground approach that combines both L1 and L2 penalties. This is particularly useful when you have a large dataset with multiple correlated features and you want the benefits of both feature selection and coefficient shrinkage.

Elastic Net combines the penalties of both L1 and L2. It adds both the absolute value and the square of the coefficients to the loss function.

$$\text{Loss} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$

By adjusting the ratio between these two penalties, Elastic Net can:

1. **Perform Feature Selection:** Like Lasso, it can set coefficients to zero.
2. **Handle Correlated Groups:** Unlike Lasso, it tends to include the entire group of correlated variables (or leave them all out) rather than picking one at random.

When implementing Elastic Net (for example, in Scikit-Learn), you usually deal with two main knobs:

1. **Alpha ($\alpha$):** The overall strength of the penalty.
2. **L1 Ratio:** This determines how much of the penalty is L1 vs L2.
* An L1 ratio of **1.0** is exactly Lasso.
* An L1 ratio of **0.0** is exactly Ridge.
* Anything in between (e.g., **0.5**) is a true Elastic Net


### Comparison Summary

| Feature | L1 (Lasso) | L2 (Ridge) |
| --- | --- | --- |
| **Penalty Term** | $\sum | w_i |
| **Coefficient Effect** | Can shrink to zero | Shrinks toward zero |
| **Feature Selection** | Yes | No |
| **Robustness** | Robust to outliers | Less robust than L1 |

</details>

---

<details markdown="block">
<summary> Batch size</summary>

In machine learning, **Batch Size** is a hyperparameter that defines the number of training samples propagated through the network before the model's internal parameters (weights) are updated.

### The Three Main Approaches

1. **Batch Gradient Descent (Batch Size = Total Dataset):**
* The model looks at *every* single example before making one update.
* **Pro:** Very stable updates.
* **Con:** Extremely slow and memory-intensive for large data.

2. **Stochastic Gradient Descent (Batch Size = 1):**
* The model updates its weights after *every* single example.
* **Pro:** Very fast and adds "noise" that can help skip over local minima.
* **Con:** The learning path is extremely erratic and zig-zags constantly.

3. **Mini-Batch Gradient Descent (Batch Size = 32, 64, 128, etc.):**
* The choice in almost all modern deep learning.
* **Pro:** Balances the stability of Batch GD with the speed and "noise" of Stochastic GD. It also allows for **GPU parallelization**.


### Comparison of small and large batch sizes

| Impact Area | Small Batch (e.g., 32) | Large Batch (e.g., 1024) |
| --- | --- | --- |
| **Memory** | Low memory footprint. | High; can lead to "Out of Memory" errors. |
| **Generalization** | Often generalizes better (adds helpful noise). | Can lead to "sharp minima" (overfitting). |
| **Speed** | More updates per epoch, but slower per update. | Fewer updates, but utilizes GPU cores better. |



</details>
