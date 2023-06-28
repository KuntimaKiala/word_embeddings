
import csv
import numpy as np

def read_data(filename) :
    data = []
    with open(filename, "r") as f:
        reader_obj = csv.reader(f)
        for word in reader_obj :
            data.append(word[0].lower())
    return data


def stop_words_removal(sentences, stop_words) :
    
    # make sure there is stop words 
    assert(len(stop_words) > 0)
    
    new_sentences = []
    for sentence in sentences :
        temps = [] # to hold the words temporarily
        for word in sentence.split() : # split() will to replace space with commas
            if word not in stop_words :
                temps.append(word)
        new_sentences.append(temps)
    
    # make sure there is new sentences
    assert(len(new_sentences) > 0)
    return new_sentences
    

def bigrams(sentences) :
    # each list in the bigram has the input word and the ouput word
    bigrams = []
    for words_list in sentences:
        for i in range(len(words_list) - 1):
            for j in range(i+1, len(words_list)): 
                
                bigrams.append([words_list[i], words_list[j]])
                bigrams.append([words_list[j], words_list[i]])
                
    return bigrams



def dictionary(bigrams) :
    
    all_words = []
    for bi in bigrams :
        all_words.extend(bi)
    all_words = sorted(list(set(all_words)))  
    
    words_dict = {}

    counter = 0
    for word in all_words:
        words_dict[word] = counter
        counter += 1
    return (words_dict)
    
def one_hot_vector(words_dict, printf=False) :
    one_hot_matrix = np.eye(len(words_dict), )
    one_hot_vector = {}
    
    counter = 0 
    for word in words_dict :
        one_hot_vector[word] = one_hot_matrix[counter, :]
        if printf :
            print(word, " :", one_hot_vector[word].tolist())
        counter += 1 

    return one_hot_vector



def data_process(bigrams, one_hot_vector ) :
    # a bigram is a list of list of words
    # each list in the bigram has the input word and the ouput word
    
    X, Y = [], []
    for bi in bigrams :
        X.append( one_hot_vector[bi[0]])
        Y.append( one_hot_vector[bi[1]])
    
    X = np.array(X) 
    Y = np.array(Y)   
    return X, Y
