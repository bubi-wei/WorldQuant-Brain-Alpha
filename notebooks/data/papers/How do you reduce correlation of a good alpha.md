This is a common problem many researchers face in their alpha research — you are not alone. First, let’s look at the good side of the problem. If the correlation between alphas is high, that means you have probably implemented similar ideas, so it is unlikely that you did something wrong. Your idea and implementation should be sound (assuming the original alpha had good performance).

So if you are new researcher, you should keep the idea around because it can be used for different alphas. Those alphas can be a variation of the current alpha using:

 Different data fields: You might try to use an equivalent data field first (such as “high,” “low” or “open” to replace “close”).
Different operator: Again start with something you find similar in practice, building your own library of similarity that could further help you reduce max correlation.
Different grouping: This is powerful approach, but don’t create an arbitrary group just for the sake of reducing correlation.
The reason to try something equivalent first is to reduce the chance that you distort the implementation of your original idea. Maintain the purity of the idea rather than complicate it unnecessarily for the sake of correlation fitting (which is considered bad practice).

Of course, the best way to reduce max correlation is to think outside of the box. That is true research.