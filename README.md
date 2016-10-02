# sentiment_analysis

# Goal
The task here is to create a classifier to predict sentiment using training data
from Kaggle https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data.
Assess model repformance using 10-fold validation.

# Approach

After some research on the topic I found an approach that is most interesting to me 
because I have some experience in applying CNN to images. 
I followed the schema outlined in Kim (2014). I also found helpful following posts:

        Very nice overview on using CNN for NLP with a lot of useful links at the end
 
	http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/#more-348
        
        Thorough explanation of the Kim's approach and implementation in Tensor Flow

        http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

# Basic Idea

Here I will brefly summarize the approach. I will just repeat what is described in the KIm's paper and links above. For more information please refer to the links and Kim (2014) paper.

In order to use CNNs for sentence classification Kim suggested to use fixed lenght word embedding and 
create a matrix of the size (number of words in sentence) x (embedding dimensionality). This is already similar to a single channel image matrix. One can also add more channels by utilizing different types of embeddings. The different sentence lenght are handled by just adding a padding and fixing the maximum length of the sentence. This will allow batch training.
Once sentente translated to an image like representation convolutional operations can be used. The main difference between standard convolutional kernels for image processing and text processing is that kernels for text processing should be as wide as embeddings. For example Kim suggested to use kernals of size 3,4,5x(embedding dimensionality). This type of kernels will convolve phrases of 3, 4, 5 word length. After applying convolutions and ReLu activations max-pooling is used to create a feature vector and finaly fully connected layer with drop out and SoftMax for classification. For the experiments here I used word2vec 300 dimentional embeddings. 

# Data Preprocessing

I wrote a simple notebook to convert data from a Kaggel train tsv to a format that can be digested by
the code from this repositories:
    [sent-conv-torch] (https://github.com/harvardnlp/sent-conv-torch)
    [CNN_sentence] (https://github.com/yoonkim/CNN_sentence)

At first I used original Kim's python/Theano implementation. While I was able to reploduce Kim's results on a RT data det (provided with the repo) I had a performance issue when I swithed to a Kaggle data set which is order of magnitude lardger. 10-fold cross validation for this data set was intracable on my local machine, so I had to use EC2 g2.2xlarge with GPU acceleration. Setting up this machine for Theano and Torch is yet another story wich took me a while because of the nvidia driver installation problems. I foud this page is a life saver http://tleyden.github.io/blog/2015/11/22/cuda-7-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/. 
After setting up the machine I tried to run original Theano implementation and encountered the problem of not having enough memory on GPU for "Kaggle" datasert. I decided to swith to a torch implementation [code] (https://github.com/harvardnlp/sent-conv-torch) and install torch on my EC2 machine. I had to fix some minor problems since initial version wasn't working for me. I suggest to use my fork of the [code] (https://github.com/harvardnlp/sent-conv-torch) if you want to reproduce the results. Torch implementation much faster then Theanos (IMHO).

# Creating Data for sent-conv-torch
    I use my nootebook to create a training data for sent-conv-torch (see coments in the nootebook).
    You will also need to get wor2vec embeddings file.
    After that I use the preprocessing util from sent-conv-torch to create a custom hdf5 file:
      
        python preprocess.py custom /path/to/word2vec.bin /path/to/train.data

    The script will output custom.hdf5 input file for torch training tool.

#Results.

      

## Relevant publications

This code is based on Kim (2014) and the corresponding Theano [code](https://github.com/yoonkim/CNN_sentence/).

    Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746–1751, Doha, Qatar. Association for Computational Linguistics.
