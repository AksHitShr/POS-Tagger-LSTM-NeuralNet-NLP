## Introduction to NLP

This assignment is based on analysis of performance of Neural Network and LSTM models, trained on the Part of Speech (POS) tagging task.

#### Files Included:
#### pos_tagger.py (contains code to load pretrained models, take input sentence and give pos tag corresponding to each word of the sentence. Also contains code to get evaluation metrics and plots used in Report.)
##### To run (-f flag for FFNN and -r flag for RNN):
> python3 pos_tagger.py -f
#### To run code for getting metrics and plots for both models (flag -fm for FFNN and -rm for LSTM):
> python3 pos_tagger.py -fm
#### en_atis-ud-dev.conllu, en_atis-ud-test.conllu, en_atis-ud-train.conllu (dataset used)
#### LSTM_POS_Tagger.pt, FFNN_POS_Tagger.pt (pre-trained models)
##### Load in files using:
> torch.load('LSTM_POS_Tagger.py)
#### 2 ipynb files containg code for FFNN and LSTM. They also contain code used for printing evaluation metrics, plotting graphs, training, etc.
#### Report.pdf containg the required evaluation metrics, plots and analysis.
