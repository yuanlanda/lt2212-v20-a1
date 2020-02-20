import os
import sys
from glob import glob
import pandas as pd
import numpy as np
import numpy.random as npr
# ADD ANY OTHER IMPORTS YOU LIKE

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

def part1_load(folder1, folder2, n=1):
    file1_path = glob("{}/*.txt".format(folder1))
    file2_path = glob("{}/*.txt".format(folder2))

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

    total_words.insert(0,'classname')
    total_words.insert(0,'file name')

    df = pd.DataFrame(columns=total_words, data=file_word_list)
    return df

def part2_vis(df, m):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    class_list = df['classname'].unique()

    df1 = df.loc[df['classname'] == class_list[0]]
    df2 = df.loc[df['classname'] == class_list[1]]

    df1_sum = df1.iloc[: , 2:].sum(axis = 0, skipna = True)
    df2_sum = df2.iloc[: , 2:].sum(axis = 0, skipna = True) 

    df_plot = pd.DataFrame({class_list[0]:df1_sum, class_list[1]:df2_sum}, columns=class_list)

    df_plot_sort = df_plot.sort_values(by=class_list[1], ascending=False).iloc[0:m , : ]
    return df_plot_sort.plot(kind='bar')

def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    return df #DUMMY RETURN


# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.
if __name__ == '__main__':
    df = part1_load('crude', 'grain')
    part2_vis(df)
