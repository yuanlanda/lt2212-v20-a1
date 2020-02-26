# LT2212 V20 Assignment 1

## Part 1 - Convert the data into a dataframe
## Part 2 - Visualize

I split the txt files by spaces and lowercase all the words. In part 2, the default plot shows the top 10 term frequences sorted by class 'crude' in a descending order. You may change the counts of the terms by changing the second parameter(*m*) in the function part2_vis(df, m=10).


## Part 3 - tf-idf
## Part 4 - Part 3 Visualize

Compared to the chart in part2, the top 10 term frequences in part 3 has more specific terms(e.g. 'bpd','opec','crude' in class 'crude'), and tf-idf value also filters some function words(e.g. 'a','to','of'), which might be useless for classification.


## Part Bonus - classify

I chose SVM algorithms as my classifier modle. I read the instrustion [Implementing SVM and Kernel SVM with Python's Scikit-Learn](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/) as a reference to implement my classifier. The classification accuracy without tf-idf is around 0.9655, while the accuracy with tf-idf is reaching nearly 0.9957. The tf–idf value increases proportionally to the number of times a word appears in the document, therefore the transformed training dataset via tf–idf is better for model to classify.

