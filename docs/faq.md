We have list some common troubles faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.
If the contents here do not cover your issue, please just create an issue [here](../../issues).

# Questions

<!-- TOC -->

- [Environment Installation](#environment-installation)
- [Training and Test](#training-and-test)
- [Paper Details](#paper-details)

<!-- TOC -->

## Environment Installation

1.  Q:

## Training and Test

1.  Q:

## Paper Details

1.  Q: Will the code be open sourced to MMDetection for wider spread? ([Issue #1](../../issues/1))

    A: MI-AOD is mainly for active learning, but MMDetection is more for object detection.
    It would be better for MI-AOD to open source to an active learning toolbox. 

2.  Q: There are differences on the order of maximizing/minimizing uncertainty and the fixed layers between paper and code. ([Issue #4](../../issues/4))

    A: Our experiments have shown that, if the order of max step and min step is reversed (including the fixed layers), the performance will change little.
        
3.  Q: The initial labeled experiment in Figure 5 of this paper should be similar in theory. Why not in experiments? ([Issue #4](../../issues/4))

    A: The reason can be summarized as:
    - Intentional use of unlabeled data
    - -> Better aligned instance distributions of the labeled and unlabeled set
    - -> Effective information (prediction discrepancy) of the unlabeled set
    - -> Naturally formed unsupervised learning procedure
    - -> Performance improvement

4.  Q: 

