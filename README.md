# Preimage
Structured prediction with string kernels. <br />
The next subsections give an overview of our work published at the International Conference on Machine Learning (ICML 2015). For more information, you can download our paper at : http://jmlr.org/proceedings/papers/v37/giguere15.pdf.

## Problem
This project aims at solving two problems involving the prediction of strings or sequences. The first one is the structured prediction problem, where the goal is to find the output (string) associated to a given input. The second one is the predictor maximization problem, where the goal is to find the string that maximizes the prediction function of a classifier or a regressor. <br />

We show an example of the structured prediction problem using the ocr-letter dataset, where the goal is to predict the handwritten word contained in a binary pixel image. We also show an example of the predictor maximization problem where we want to predict the peptides (small proteins) that can best achieve some desirable activity (like fighting some disease). 

## String kernels
Intuitively, a string kernel is a function that measures the similarity between two strings. By using a string kernel, we are able to treat the structured prediction problem as a joint problem. This means that when we want to predict the best possible string, the best character at some position in our string depends on what are the other characters in the string. Treating the structured prediction problem as a joint problem can improve the accuracy of our predictor but can make the inference phase (searching for the best string over all possible ones) a lot harder.<br />

Moreover, the best string kernel depends on the problem we are trying to solve. In this work, we use a kernel called the Generic String (GS) kernel which computes the similarity between two strings by comparing all their substrings of a specific length. There is also a parameter that controls how much we penalise when two substrings are at different positions, and another parameter controls how much we penalise when two substrings are different. We can obtain one of 8 known string kernels depending on the values we fix these parameters to. The interesting thing about this kernel is that we can choose the best parameters by cross-validation. That way, we don't make assumptions about the best kernel for a specific problem, but we let the problem decide for us.

## Method
Both the structured prediction and the predictor maximization problems involve a search phase. We can see the training phase as the problem of learning a function that predicts how good a string is. Once we have this function, we are interested in finding the string that maximizes it (the string with the highest predicted value over all possible strings).

For some parameters of the GS kernel (when we obtain the hamming or the weighted degree kernel), we can use a polynomial time algorithm to predict the best string. For more complex string kernels, we can use this algorithm to develop an upper bound that is used in a branch and bound algorithm. This upper bound is going to estimate the maximum value that can be obtained by a partial solution. For instance, a partial solution could be a string of length 5 starting with "a". By bounding the maximum value of different partial solutions, we can choose which search space (partial solution) should be explored first. We can then bound the value of its children ("aa" to "az") and choose the best one to be explored again. That way, we are able to reach a good solution rapidly and we can keep searching until we can prove that our best solution found has a higher value than all other partial solutions or set a time limit for the search. 

# Installation
Download the code. Install the packages listed in the requirements.txt file. Then go in the main preimage folder and compile the cython: <br />
```
python setup.py build_ext -i 
```
Run an example file: <br />
```
python preimage/examples/n_gram_ocr.py
```

# How to use it
Check the examples on the website: http://a-ro.github.io/preimage/ <br />
Or check the examples in preimage/examples.

# Authors and Contributors
Amélie Rolland and Sébastien Giguère
