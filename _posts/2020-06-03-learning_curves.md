---
layout: post
title:  "Get Close With Learning Curves"
date:   2020-06-14
categories: optimizations  
author: William D'Souza
---

![Alt Text]({{ site.url }}/images/learning_curves/logo.jpg)

Learning curves are a simple yet effective visual that should be in your toolkit. We tend to forget to use them when building our models since it is so easy to evaluate performance by looking at the final scoring metrics. However, the visuals are a highly impactful way to learn about your model and improve your comprehension of it!

When describing data to a viewer, we typically represent any findings through visualizations for the fundamental reason that visuals speak largely to people. It gives the audience the ability to process the information faster than just going through sheets of numbers.  It's easy for us to forget that as providers of data, we need to rely on visuals just as much as someone who digests it.

## What Are Learning Curves? Why Are They Helpful?

Learning curves are a visual tool to diagnose your model's performance during its training stage. The visuals outline the relationship between your training data and any chosen metrics. It equips the practitioner with important information by outlining the changes in the model's learning performance in terms of experience. 

Learning curves are helpful for a few main reasons:

**1) Can help you evaluate the bias-variance trade-off**

The fundamental concept of the bias-variance tradeoff deals with prediction errors. There is always a tradeoff with a models ability to minimize bias and variance. An excess of bias in the model and it may ignore the data. An excess of variance in the model and it will "over-analyze" the data and have problems generalising. *Flexible models will have higher variance but less bias. The more flexible the model, the variance will increase and bias will decrease*

***2) Can help you understand the fit of your model**

Related to the bias-variance trade-off,  learning curves are an amazing way to evaluate the goodness of fit for the model. ***Undefitting*** happens when the model can not capture any patterns in the data, typically an indication of suffering from a high bias and low variance. ***Overfitting*** occurs when the model captures the patterns in the data extremely closely, typically an indication the model is suffering from high variance and loss bias. As you could guess a ***Well fit*** model is one that minimizes the trade-off between variance and bias, gains an understanding of the data and can generalize.

**3) You develop a more intimate understanding of how your model operates**

Machine learning can be a complicated process depending on the model and interpretability can always be an issue. If you look at neural networks, we are still researching how certain types of networks operate the way they do. We have a general understanding of how it works, but we don't have complete answers for certain things like choosing the number of layers and hidden units. With popular machine learning models (regressors and trees), the math is more explicit and a plentiful of research has been done to understand it. 

Visualising how your model is training over time brings the practitioner a little bit closer to the model, helping to develop an understanding with how the model is learning from the data. It gives you the ability to comprehend the underlying data and chosen model when comparing with other models, it will  help you intuitively understand why some models are either more interpretable or flexible and why certain datasets are better suited for certain models

## How do Learning Curves Work?

Learning curves are simple plots with experience on the x-axis and performance on the y-axis. To gain an intuitive understanding of what a learning curve is, imagine that you wanted to get better at typing both faster and accurate. Every day, you write a new paragraph and measure your words per minute and accuracy. After 100 days, you can see how your accuracy and words per minute have improved, and this is the general idea of what a learning curve is!

A machine learning model's state at each step is plotted with the training data and validation set. Evaluation of the training data informs you how well the model can learn and evaluation of the validation set will give you an understanding of your generalisation error. **It's a good practice to plot both the training and validation set on the same plot** 

Another good practice is to create 2 plots for multiple metrics. One plot to evaluate the model's parameters with a loss metric, and a second plot to evaluate performance with a scoring metric.

***The following learning curves were built using artificial data meant to exaggerate visuals. In real life, the curves for certain losses may not always be so stable as in the examples below.***

## Diagnosing the Model's Behaviour

### Overfit Model

Overfitting refers to when the model has learned the data too closely, which will ultimately result in poor performance when the model is used for predictions as it won't be able to generalise.

Below is an example of a model that suffers from overfitting. 

![Alt Text]({{ site.url }}/images/learning_curves/overfit_logloss.gif)

![Alt Text]({{ site.url }}/images/learning_curves/overfit_acc.gif)

To visually understand that overfitting is occurring, its best to look if over time there is a training loss that is consistently decreasing and a validation loss that begins to slowly increase. It can be seen when the validation loss follows the same pattern as the training loss but begins to deviate.


### Underfit Model

Underfitting refers to when the model struggles to learn from the training data, which ultimately will result in a poor model since any predictions attempted to being made with the model will be random.

Below is an example of a model that suffers from underfitting. 

![Alt Text]({{ site.url }}/images/learning_curves/underfit_logloss.gif)

![Alt Text]({{ site.url }}/images/learning_curves/underfit_acc.gif)

A model that is under fitted can be seen from visualising the training loss over time. it will look either extremely noisy or can be a straight line and you will notice that distance between the validation and training loss is large.

### Well Fit Model 

As you can imagine, a model that is fit well is one that is in between a underfit and overfit model, which will result in a model that is extremely useful for predictions as it can generalize well.

Below is an example of a model that is well fit. 

![Alt Text]({{ site.url }}/images/learning_curves/wellfit_logloss.gif)

![Alt Text]({{ site.url }}/images/learning_curves/wellfit_acc.gif)

A model that is fit well can be seen from visualising the training and validation loss over time, it should decrease to a point and stabilise from there, and the gap between the two lines should be as minimal. There is no perfect model and the gap in between won't always be the smallest. This is the generalization loss that is natural to see in all modelling.

### Unrepresentative Training Sets

Learning curves can also help paint a picture of the underlying dataset used to train a model. A dataset is considered representative if is similar in statistical properties to other data pulled from the same domain. Unrepresentative data causes a lot of problems for machine learning models. A model that is trained on an unrepresentative training set can be seen visually a learning curve with a training and validation sets. There may be a spike in improvement with both lines, however, a large gap between the two will exist.

## Learning Curves Should be Your "Go-To"

Learning curves should always be in your tool kit when building models. They are extremely easy to perform and add an enormous amount of value. They will even start helping you to understand techniques like ***early stopping***, which will go a long way when building neural networks! We provide visuals to others to help simplify and understand data better, so as easy as it is for us to fall into the trap of only relying on the numbers, we should practice always using visuals to understand what we create ourselves.



