import os
import sys
from glob import glob
import pandas as pd
import numpy as np
import numpy.random as npr
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
# ADD ANY OTHER IMPORTS YOU LIKE

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

CLASS_NAME = 'classname'
FILE_NAME = 'filename'

def part1_load(folder1, folder2, n=1):
    file1_path = glob("{}/*.txt".format(folder1))
    file2_path = glob("{}/*.txt".format(folder2))

    #collect all the words from corpus
    file_path = file1_path + file2_path
    total_words = []
    for filename in file_path:
        with open(filename, "r") as thefile:
            for line in thefile:
                words = line.split()
                for word in words:
                    word = word.lower()
                    if word not in total_words:
                        total_words.append(word)

    # count number of words occurred in each document
    file_word_list = []
    for filename in file_path:
        word_count_dict = {}
        word_count_list = []
        with open(filename, "r") as thefile:
            for line in thefile:
                words = line.strip().split(" ")
                for word in words:
                    word = word.lower()
                    if word in word_count_dict:
                        word_count_dict[word]+=1
                    else:
                        word_count_dict[word]=1

        for word in total_words:
            if word in word_count_dict.keys():
                word_count_list.append(word_count_dict[word])
            else:
                word_count_list.append(0)

        word_count_list.insert(0, filename[0:len(folder1)])
        word_count_list.insert(0, filename[len(folder1)+1:])

        file_word_list.append(word_count_list)

    total_words.insert(0, CLASS_NAME)
    total_words.insert(0, FILE_NAME)

    df = pd.DataFrame(columns=total_words, data=file_word_list)
    return df

def part2_vis(df, m=10):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    class_list = df[CLASS_NAME].unique()

    df1 = df.loc[df[CLASS_NAME] == class_list[0]]
    df2 = df.loc[df[CLASS_NAME] == class_list[1]]

    df1_sum = df1.iloc[: , 2:].sum(axis = 0, skipna = True)
    df2_sum = df2.iloc[: , 2:].sum(axis = 0, skipna = True) 

    df_plot = pd.DataFrame({class_list[0]:df1_sum, class_list[1]:df2_sum}, columns=class_list)

    df_plot_sort = df_plot.sort_values(by=class_list[1], ascending=False).iloc[0:m , : ]
    return df_plot_sort.plot(kind='bar')


def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    not_zero_docs = df.iloc[: , 2:].astype(bool).sum(axis=0)
    idf = np.log((df.shape[0]/not_zero_docs))
    tf_idf_df = df.iloc[: , 2:].mul(idf, axis=1)

    tf_idf_df.insert(0, CLASS_NAME, df[CLASS_NAME])
    tf_idf_df.insert(0, FILE_NAME, df[FILE_NAME])

    return tf_idf_df


def part_bonus_classify(df):
    svclassifier = SVC(kernel="linear")
    x = df.drop([CLASS_NAME, FILE_NAME], 1)
    y = df[CLASS_NAME]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return accuracy_score(y_test, y_pred)


# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.
if __name__ == '__main__':
    df = part1_load('crude', 'grain')
    part2_vis(df)
    df_tfidf = part3_tfidf(df)
    part_bonus_classify(df)
    part_bonus_classify(df_tfidf)
