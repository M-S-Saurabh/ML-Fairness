 SAURABH
---------
- Why is fp rate same in SVM and LR?
- Find new metrics to show bias on COMPAS

 ANUSHREE
---------
- Home loan data show Fairness metrics

 ESHA
-------
- Find Fairness studies/ features in Breast cancer data.
- Find some fallback datasets with clear bias

--------------------------------------------------------------------------------------------------------------------------------
Week: 25 Oct - 31 Oct
--------------------------------------------------------------------------------------------------------------------------------
Problem: 
If we're using race as a feature in our predictor (SVM), it will automatically have an effect on outcome.
Why are we surprised that it does?

Solution:
Maybe ignore all sensitive attributes informationin these algorithms?

Moritz Hardt: "This idea of fairness through blindness, however, fails due to the existence of redundant encodings. There are almost always ways of predicting unknown protected attributes from other seemingly innocuous features."
(http://blog.mrtz.org/2016/09/06/approaching-fairness.html)

Demographic Parity: decision is independent of protected attribute P(C=1|A=a) = P(C=1|A=b).
It might be the case that Y is naturally corelated with A. This is not cause for concern as interests differ between groups naturally. Demographic parity may force us to make good predictions for group a and essentially random guess for group b.
Setting up for failure.

Equalized odds: ()
P(C | A=a, y) = P(C | A=b, y)
Equality of opportunity: (weaker condition than equalized odds)
P(C=1 | A=a, y=1) = P(C=1 | A=b, y=1)
C = 1 is the advantaged class. TPR should be equalized.
(https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf)

Varying threshold over the score instead of direct prediction.

--------------------------------------------------------------------------------------------------------------------------------

Progress report - Plan
-----------------------
1. Introduction from proposal (1 para)
2. Datasets we are using (1 para)
3. What Bias they have examples: (mention discrimination bias, gender bias) (1 para)
4. False positive rate, demographic parity, equality of opportunity explain (1 para)
   Show metrics in dataset (FPR, Parity, TPR graphs)
5. Mitigating threshold: 3 methods explain (Threshold- Saurabh)
6. Tried so far: Threshold-based: explain method detailed, show graph (Saurabh)
7. Discuss results.
8. Future work: say we'll do the other 2 methods.
9. References



