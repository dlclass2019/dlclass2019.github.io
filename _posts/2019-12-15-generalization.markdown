---
layout: post
title:  "Understanding Generalization Performance for Deep Classifiers"
date:   2019-12-02 09:00:09 -0500
categories: post
mathjax: true
---
<!-- Need to include this line to enable mathjax -->
{% include mathjax.html %}

Posted by Rahul Pandey, Angeela Acharya, Junxiang (Will) Wang and Jomana Bashatah. Presented by Zhengyang Fan, Zhenlong Jiang and Di Zhang.

### Classification in Machine Learning
A general problem of classification in machine learning can be defined as predicting an outcome y from some set Y of possible outcomes, on the basis of some observation x from a feature space X.

Mathematically, we can define a classification problem as to find a map $$ f: X \rightarrow Y $$ based on $$ n $$ observations $$ (x1, y1), ..., (xn, yn) $$ such
that $$ f(x) $$ is a good prediction of $$ y $$ for a new observation $$ (x, y) $$.

To measure the success of the map $$ f: X \rightarrow Y $$, we generally define loss function. Low loss function score implies better classification model.

### Deep Learning in Classification
Most recent approach to solve classification problem is to use Deep Learning. Deep Learning uses deep networks, which are statistically deep compositions of non-linear functions.

$$ f = f_L \circ f_{L_1} \circ ... \circ f_{1} $$

There are various non-learning function, which has been used in deep learning. Some of the popular ones are sigmoid function and ReLu (Rectified Linear Unit) function.

__Sigmoid__

$$ f_i \rightarrow \sigma(W_ix) $$, where

$$ \sigma(v_i) = \frac{1}{1+e^{-v_i}} $$

__ReLu__

$$ f_i \rightarrow \sigma(W_ix) $$, where

$$ \sigma(v_i) = max\{0,v_i\} $$

### Deep Learning and Generalization
Deep Networks Classifiers generally perform much better than the traditional machine learning. That means their function map has a very low loss function score not only on training data, but on test data as well. However, classifiers in deep learning are heavily parameterized due to the complex nature of deep networks. The reason for its optimal performance despite overfitting is the size of the function space, which is large enough to interpolate data points.

In this blog, we will investigate the generalization of deep networks and compare its performance.

Before jumping on generalization, let's talk about the probabilistic assumption of a deep learning model.

Given a probability distribution $$ \mathbb{P} $$ on $$ X \times Y $$, we define a function $$ f $$ such that the risk $$ R(f) $$ is minimized.

Risk is defined as

$$ R(f) = \mathbb{E}l(f(X), Y) $$

where $$ \mathbb{E} $$ is expectation and $$ l $$ is loss

over all $$ X \times Y $$ i.i.d. pairs.

NOTE i.i.d. $$ \rightarrow $$ independent and identically distributed random variables

Now, to measure the richness of function $$ \mathcal{F} $$ over all $$ X \times Y $$ i.i.d. pairs, we define Rademacher Complexity.

#### Rademacher Complexity
As stated above, Rademacher Complexity are used to find the richness of the learnt function in ML. Now, given a function $$ \mathcal {F} $$, the Rademacher Complexity is $$ \mathbb{E}{\left \| R_n \right \|}_{\mathcal{F}} $$, where the empirical process $$ R_n $$ is defined as

$$ R_n(f) = \frac{1}{n}\sum_{i=1}^{n} \epsilon_if(X_i) $$, where

$$ \epsilon_i $$ are Rademacher random variables, which are i.i.d. uniform on $$ \{-1, 1\} $$
