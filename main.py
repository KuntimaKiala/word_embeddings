from utilities.protocols import  read_data, stop_words_removal, bigrams, dictionary, one_hot_vector, data_process

if __name__ == "__main__" :
    
    # 1 - extract data 
    data = read_data("/home/kuntima/workspace/github/word_embeddings/data/royal_data.txt")
    # 2 - remove stop words
    stopwords = ['the', 'is', 'will', 'be', 'a', 'only', 'can', 'their', 'now', 'and', 'at', 'it']
    sentences = stop_words_removal(sentences=data, stop_words=stopwords)
    # 3 - create bigrams
    bigrams = bigrams(sentences)
    # 4 - create a dictionary of each unique word
    dictionary_words = dictionary(bigrams)
    # 5 - create a one hot vector
    one_hot_v = one_hot_vector(dictionary_words, printf=False)
   
    X, Y = data_process(bigrams, one_hot_v)