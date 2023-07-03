import numpy as np
from utilities.protocols import  read_data, stop_words_removal, bigrams, dictionary, one_hot_vector, data_process
from trainer import SoftMax, MLP, DataHandler
import torch 


if __name__ == "__main__" :
    
    # 1 - extract data 
    data = read_data("/home/kuntima/workspace/github/word_embeddings/data/royal_data.txt")
    # 2 - remove stop words
    stopwords = ['the', 'is', 'will', 'be', 'a', 'only', 'can', 'their', 'now', 'and', 'at', 'it', 'to', 'go', 'have']
    sentences = stop_words_removal(sentences=data, stop_words=stopwords)
    # 3 - create bigrams
    bigrams = bigrams(sentences)
    # 4 - create a dictionary of each unique word
    dictionary_words = dictionary(bigrams)
    # 5 - create a one hot vector
    one_hot_v = one_hot_vector(dictionary_words, printf=False)
   
    # 6 - Extract Input and Output
    X, Y = data_process(bigrams, one_hot_v)
    
    # 7 - convert data to pythorch readable
    train_dataset = DataHandler(X, Y)
    dataset = train_dataset.DataLoader()

    # 8 - Model    
    device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')
    epoch = 1000
    embed_size = 2
    output_size = Y.shape[1]
    input_size = output_size
    model = SoftMax(input_size, embed_size, output_size)
    model.to(device=device)
    
    #9-  training
    trainer = MLP(model=model, epochs=epoch, learning_rate=0.01)
    trainer.run(dataset)
    
  
    #10- Get the weigths
    word_embeddings = {}
    for word in dictionary_words.keys():
        word_embeddings[word] = model.softmax_head[1].weight[dictionary_words[word]].detach().numpy()
        
        
        
    import matplotlib.pyplot as plt

    # plt.figure(figsize = (10, 10))
    for word in list(dictionary_words.keys()):
        coord = word_embeddings.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))

    plt.savefig('img.jpg')

   