from nltk.tokenize import RegexpTokenizer
from conllu import parse
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator,Vocab
from collections import Counter
import sys
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
EMBEDDING_SIZE=256
HIDDEN_DIM=256
NUM_STACKS=2
BATCH_SIZE=256
lrate=0.001
EPOCHS=20
p=4
s=4

class FeedForwardNN(torch.nn.Module):
  def __init__(self, p, s, out_size, vocabulary_size: int, embedding_size: int, hidden_size: int):
    super().__init__()
    self.embedding_module = torch.nn.Embedding(vocabulary_size, embedding_size)
    self.entity_predictor = torch.nn.Sequential(
                                    torch.nn.Linear(embedding_size*(p+s+1), hidden_size),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, out_size))
  def forward(self, word_seq: torch.Tensor):
    embedding = self.embedding_module(word_seq)
    embedding=embedding.reshape(embedding.shape[0],-1)
    return self.entity_predictor(embedding)

class LSTMModel(torch.nn.Module):
  def __init__(self, embedding_dim: int, hidden_dim: int, vocabulary_size: int, tagset_size: int, stacks: int):
    super().__init__()
    self.embedding_module = torch.nn.Embedding(vocabulary_size, embedding_dim)
    self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, stacks)
    self.hidden_to_tag = torch.nn.Linear(hidden_dim, tagset_size)
  def forward(self, sentence: torch.Tensor):
    embeddings = self.embedding_module(sentence)
    lstm_out, _ = self.lstm(embeddings)
    return self.hidden_to_tag(lstm_out)

def filter_sentences_by_tag(sentences, pos_tags, tag_to_exclude='SYM'): # removing SYM tag sentences
    filtered_sentences = []
    filtered_pos_tags = []
    for sentence, tags in zip(sentences, pos_tags):
        if tag_to_exclude not in tags:
            filtered_sentences.append(sentence)
            filtered_pos_tags.append(tags)
    return filtered_sentences, filtered_pos_tags
def extract_tokens_and_tags(sentences):
    token_sequences = []
    tag_sequences = []
    for sentence in sentences:
        tokens = []
        tags = []
        for token in sentence:
            tokens.append(token["form"])
            tags.append(token["upos"])
        token_sequences.append(tokens)
        tag_sequences.append(tags)
    return token_sequences, tag_sequences
def replace_low_frequency_words(sentences, threshold=3):
    word_counts = Counter(word for sentence in sentences for word in sentence)
    replaced_sentences = [
        [UNKNOWN_TOKEN if word_counts[word] < threshold else word for word in sentence]
        for sentence in sentences
    ]
    return replaced_sentences
def append_tokens(p, s, sentences):
    for sentence in sentences:
        for _ in range(p):
            sentence.insert(0,START_TOKEN)
        for _ in range(s):
            sentence.append(END_TOKEN)
    return sentences
def append_labels(p, s, sentences):
    for sentence in sentences:
        for _ in range(p):
            sentence.insert(0,0)
        for _ in range(s):
            sentence.append(0)
    return sentences

class EntityDataset_LSTM(Dataset):
  def __init__(self, sent, labs, vocabulary:Vocab|None=None):
    """Initialize the dataset. Setup Code goes here"""
    self.sentences = sent
    self.labels = labs
    if vocabulary is None:
      self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[UNKNOWN_TOKEN, PAD_TOKEN])
      self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
    else:
      self.vocabulary = vocabulary

  def __len__(self) -> int:
    """Returns number of datapoints."""
    return len(self.sentences)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the datapoint at `index`."""
    return torch.tensor(self.vocabulary.lookup_indices(self.sentences[index])), torch.tensor(self.labels[index])

  def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of datapoints, batch them together"""
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(0))
    return padded_sentences, padded_labels

class EntityDataset(Dataset):
  def __init__(self, sent, tok,p,s, vocabulary:Vocab|None=None):
    """Initialize the dataset. Setup Code goes here"""
    self.sentences = sent
    self.labels = tok
    words = []
    labels = []
    self.p=p
    self.s=s
    for sentence, label in zip(self.sentences, self.labels):
      for word, label in zip(sentence, label):
        if word!=START_TOKEN and word!=END_TOKEN:
          words.append(word)
          labels.append(label)
    self.words=words
    self.labels=labels
    if vocabulary is None:
      self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[UNKNOWN_TOKEN])
      self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
    else:
      self.vocabulary = vocabulary
    inp_dt=[]
    inp_labels=[]
    for sen,lab in zip(sent, tok):
      for i in range(self.p,len(sen)-self.s):
        toks_to_add=sen[i-self.p:i+self.s+1]
        inp_dt.append(toks_to_add)
        inp_labels.append(lab[i])
    self.inp_dt=inp_dt
    self.inp_labels=inp_labels
  def __len__(self) -> int:
    """Returns number of datapoints."""
    return len(self.words)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the datapoint at `index`."""
    return torch.tensor(self.vocabulary.lookup_indices(self.inp_dt[index])), torch.tensor(self.inp_labels[index])

  def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of datapoints, batch them together"""
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
    labels=torch.tensor(labels)
    labels=labels.view(-1,1)
    return padded_sentences,labels
  
def get_datasets_LSTM():
  with open('en_atis-ud-train.conllu') as f:
    train_sentences = parse(f.read())
  tok_seq,tag_seq=extract_tokens_and_tags(train_sentences)
  tok_seq=replace_low_frequency_words(tok_seq)
  unique_tags = set(tag for tags in tag_seq for tag in tags)
  tag_to_id = {tag: idx+1 for idx, tag in enumerate(sorted(unique_tags))}
  new_tag_seq=[[tag_to_id[tag] for tag in tags] for tags in tag_seq]
  train_dataset=EntityDataset_LSTM(tok_seq,new_tag_seq)
  with open('en_atis-ud-test.conllu') as f:
    test_sentences = parse(f.read())
  with open('en_atis-ud-dev.conllu') as f:
    val_sentences = parse(f.read())
  test_toks,test_tags=extract_tokens_and_tags(test_sentences)
  val_toks,val_tags=extract_tokens_and_tags(val_sentences)
  val_toks,val_tags=filter_sentences_by_tag(val_toks,val_tags)
  test_tags=[[tag_to_id[tag] for tag in tags] for tags in test_tags]
  val_tags=[[tag_to_id[tag] for tag in tags] for tags in val_tags]
  val_dataset=EntityDataset_LSTM(val_toks,val_tags,vocabulary=train_dataset.vocabulary)
  test_dataset=EntityDataset_LSTM(test_toks,test_tags,vocabulary=train_dataset.vocabulary)
  return train_dataset,val_dataset,test_dataset,tag_to_id

def get_datasets_ffNN(p,s):
  with open('en_atis-ud-train.conllu') as f:
    train_sentences = parse(f.read())
  tok_seq,tag_seq=extract_tokens_and_tags(train_sentences)
  tok_seq=replace_low_frequency_words(tok_seq)
  tok_seq=append_tokens(p,s,tok_seq)
  unique_tags = set(tag for tags in tag_seq for tag in tags)
  tag_to_id = {tag: idx+1 for idx, tag in enumerate(sorted(unique_tags))}
  new_tag_seq=[[tag_to_id[tag] for tag in tags] for tags in tag_seq]
  new_tag_seq=append_labels(p,s,new_tag_seq)
  train_dataset=EntityDataset(tok_seq,new_tag_seq,p,s)
  with open('en_atis-ud-test.conllu') as f:
    test_sentences = parse(f.read())
  with open('en_atis-ud-dev.conllu') as f:
    val_sentences = parse(f.read())
  test_toks,test_tags=extract_tokens_and_tags(test_sentences)
  val_toks,val_tags=extract_tokens_and_tags(val_sentences)
  val_toks,val_tags=filter_sentences_by_tag(val_toks,val_tags)
  test_tags=[[tag_to_id[tag] for tag in tags] for tags in test_tags]
  val_tags=[[tag_to_id[tag] for tag in tags] for tags in val_tags]
  val_toks=append_tokens(p,s,val_toks)
  test_toks=append_tokens(p,s,test_toks)
  val_tags=append_labels(p,s,val_tags)
  test_tags=append_labels(p,s,test_tags)
  val_dataset=EntityDataset(val_toks,val_tags,p,s,vocabulary=train_dataset.vocabulary)
  test_dataset=EntityDataset(test_toks,test_tags,p,s,vocabulary=train_dataset.vocabulary)
  return train_dataset,val_dataset,test_dataset,tag_to_id

def reverse_dict(original_dict):
    reverse_dict = {v: k for k, v in original_dict.items()}
    return reverse_dict

def tokenise(text):
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(text)
    return tokens


def run_FFNN():
   train_dataset,val_dataset,test_dataset,tag_to_id=get_datasets_ffNN(p,s)
   entity_predictor=torch.load('FFNN_POS_Tagger.pt')
   rev_tag_to_id=reverse_dict(tag_to_id)
   user_inp=input()
   tok_sent=tokenise(user_inp)
   orig_sent=[]
   for x in tok_sent:
    orig_sent.append(x)
   ntok_sent=(append_tokens(p,s,[tok_sent]))[0]
   inp_data=[]
   for i in range(p,len(ntok_sent)-s):
    window=ntok_sent[i-p:i+s+1]
    encoded_inp=train_dataset.vocabulary.lookup_indices(window)
    inp_data.append(encoded_inp)
   with torch.no_grad():
    temp=[]
    for x in inp_data:
       temp.append(torch.tensor(x))
    inp_data=torch.stack(temp)
    res=entity_predictor(inp_data)
    pred_maxcol=np.argmax(res, axis=1)+1
    for a,t in zip(orig_sent,pred_maxcol):
        print(a+" "+rev_tag_to_id[int(t)])

def run_LSTM():
   train_dataset,val_dataset,test_dataset,tag_to_id=get_datasets_LSTM()
   entity_predictor=torch.load('LSTM_POS_Tagger.pt')
   rev_tag_to_id=reverse_dict(tag_to_id)
   user_inp=input()
   tok_sent=tokenise(user_inp)
   orig_sent=[]
   for x in tok_sent:
    orig_sent.append(x)
   ntok_sent=tok_sent
   with torch.no_grad():
    temp=[train_dataset.vocabulary.lookup_indices(ntok_sent)]
    temp=torch.tensor(temp)
    res=entity_predictor(temp)
    pred_max_index = torch.argmax(res, dim=2)
    mask = (pred_max_index == 0)
    next_max_indices = torch.argsort(res, dim=2)[:, :, -2]
    pred_max_index = torch.where(mask, next_max_indices, pred_max_index)
    for a,t in zip(orig_sent,pred_max_index[0]):
       print(a+" "+rev_tag_to_id[int(t)])

def LSTM_metrics():
   device = "mps" if torch.cuda.is_available() else "cpu"
   train_dataset,val_dataset,test_dataset,tag_to_id=get_datasets_LSTM()
   train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=train_dataset.collate)
   val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,collate_fn=val_dataset.collate)
   test_dataloader=DataLoader(test_dataset, batch_size=BATCH_SIZE,collate_fn=test_dataset.collate)
   entity_predictor=LSTMModel(EMBEDDING_SIZE,HIDDEN_DIM,len(train_dataset.vocabulary),len(tag_to_id)+1,NUM_STACKS)
   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(entity_predictor.parameters(),lr=lrate)
   entity_predictor = entity_predictor.to(device)
   dev_set_acc=[]
   for epoch_num in range(EPOCHS):
     entity_predictor.train()
     for batch_num, (words, tags) in enumerate(train_dataloader):
       (words, tags) = (words.to(device), tags.to(device))
       one_hot_tags=(torch.nn.functional.one_hot(tags, num_classes=len(tag_to_id)+1)).float()
       pred = entity_predictor(words)
       loss = loss_fn(pred, one_hot_tags)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
     entity_predictor.eval()
     with torch.no_grad():
       test_loss = 0
       for batch_num, (words, tags) in enumerate(val_dataloader):
         (words, tags) = (words.to(device), tags.to(device))
         one_hot_tags=(torch.nn.functional.one_hot(tags, num_classes=len(tag_to_id)+1)).float()
         pred = entity_predictor(words)
         test_loss += (loss_fn(pred, one_hot_tags)).item()
     print(f"Validation error: {test_loss/len(val_dataloader)}")
     entity_predictor.eval()
     predictions=[]
     true_vals=[]
     with torch.no_grad():
       for batch_num, (words, tags) in enumerate(val_dataloader):
         (words, tags) = (words.to(device), tags.to(device))
         pred = entity_predictor(words)
         pred_max_index = torch.argmax(pred, dim=2)
         true_vals.extend(tags.flatten().cpu())
         predictions.extend(pred_max_index.flatten().cpu())
     predictions=torch.stack(predictions).numpy()
     true_vals=torch.stack(true_vals).numpy()
     dev_set_acc.append(accuracy_score(true_vals,predictions))
   plt.bar(range(1,EPOCHS+1), dev_set_acc)
   plt.xlabel('epoch')
   plt.ylabel('Accuracy on dev set')
   plt.title('epoch number vs dev set accuracy for LSTM POS Tagger')
   plt.show()
   entity_predictor.eval()
   true_vals=[]
   predictions=[]
   with torch.no_grad():
     for batch_num, (words, tags) in enumerate(val_dataloader):
       (words, tags) = (words.to(device), tags.to(device))
       pred = entity_predictor(words)
       pred_max_index = torch.argmax(pred, dim=2)
       true_vals.extend(tags.flatten().cpu())
       predictions.extend(pred_max_index.flatten().cpu())
   predictions=torch.stack(predictions).numpy()
   true_vals=torch.stack(true_vals).numpy()
   f1_micro=f1_score(true_vals,predictions,average='micro')
   f1_macro=f1_score(true_vals,predictions,average='macro')
   rec_micro=recall_score(true_vals,predictions,average='micro')
   rec_macro=recall_score(true_vals,predictions,average='macro')
   pre_micro=precision_score(true_vals,predictions,average='micro')
   pre_macro=precision_score(true_vals,predictions,average='macro')
   print(f'Accuracy Score on dev set: {accuracy_score(true_vals,predictions)}')
   print(f'F1-Score(micro) on dev set: {f1_micro}')
   print(f'F1-Score(macro) on dev set: {f1_macro}')
   print(f'Recall(micro) Score on dev set: {rec_micro}')
   print(f'Recall(macro) Score on dev set: {rec_macro}')
   print(f'Precision(micro) Score on dev set: {pre_micro}')
   print(f'Precision(macro) Score on dev set: {pre_macro}')
   print(f'Confusion matrix for dev set:\n {confusion_matrix(true_vals,predictions)}')
   entity_predictor.eval()
   predictions=[]
   true_vals=[]
   with torch.no_grad():
     for batch_num, (words, tags) in enumerate(test_dataloader):
       (words, tags) = (words.to(device), tags.to(device))
       pred = entity_predictor(words)
       pred_max_index = torch.argmax(pred, dim=2)
       true_vals.extend(tags.flatten().cpu())
       predictions.extend(pred_max_index.flatten().cpu())
   predictions=torch.stack(predictions).numpy()
   true_vals=torch.stack(true_vals).numpy()
   f1_micro=f1_score(true_vals,predictions,average='micro')
   f1_macro=f1_score(true_vals,predictions,average='macro')
   rec_micro=recall_score(true_vals,predictions,average='micro')
   rec_macro=recall_score(true_vals,predictions,average='macro')
   pre_micro=precision_score(true_vals,predictions,average='micro')
   pre_macro=precision_score(true_vals,predictions,average='macro')
   print(f'Accuracy Score on test set: {accuracy_score(true_vals,predictions)}')
   print(f'F1-Score(micro) on test set: {f1_micro}')
   print(f'F1-Score(macro) on test set: {f1_macro}')
   print(f'Recall(micro) Score on test set: {rec_micro}')
   print(f'Recall(macro) Score on test set: {rec_macro}')
   print(f'Precision(micro) Score on test set: {pre_micro}')
   print(f'Precision(macro) Score on test set: {pre_macro}')
   print(f'Confusion matrix for test set:\n {confusion_matrix(true_vals,predictions)}')

def FFNN_metrics():
   epochs=10
   EMBEDDING_SIZE=20
   HIDDEN_SIZE=120
   lrate=1e-2
   BATCH_SIZE=10
   p=4
   s=4
   train_dataset,val_dataset,test_dataset,tag_to_id=get_datasets_ffNN(p,s)
   train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
   val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
   test_dataloader=DataLoader(test_dataset, batch_size=BATCH_SIZE)
   device = "mps" if torch.cuda.is_available() else "cpu"
   entity_predictor=FeedForwardNN(p,s,len(tag_to_id),len(train_dataset.vocabulary),EMBEDDING_SIZE,HIDDEN_SIZE)
   loss_fn = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(entity_predictor.parameters(), lr=lrate)
   entity_predictor = entity_predictor.to(device)
   for epoch_num in range(epochs):
     entity_predictor.train()
     for batch_num, (words, tags) in enumerate(train_dataloader):
       (words, tags) = (words.to(device), tags.to(device))
       one_hot_tags=(torch.nn.functional.one_hot(tags - 1, num_classes=len(tag_to_id))).float()
       one_hot_tags=torch.squeeze(one_hot_tags,dim=1)
       pred = entity_predictor(words)
       loss = loss_fn(pred, one_hot_tags)
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
     entity_predictor.eval()
     with torch.no_grad():
       test_loss = 0
       for batch_num, (words, tags) in enumerate(val_dataloader):
         (words, tags) = (words.to(device), tags.to(device))
         one_hot_tags=(torch.nn.functional.one_hot(tags - 1, num_classes=len(tag_to_id))).float()
         one_hot_tags=torch.squeeze(one_hot_tags,dim=1)
         pred = entity_predictor(words)
         test_loss += loss_fn(pred, one_hot_tags)
     print(f"Validation error: {test_loss/len(val_dataloader)}")   
   entity_predictor.eval()
   predictions=[]
   true_vals=[]
   with torch.no_grad():
     for batch_num, (words, tags) in enumerate(val_dataloader):
       (words, tags) = (words.to(device), tags.to(device))
       one_hot_tags=(torch.nn.functional.one_hot(tags - 1, num_classes=len(tag_to_id))).float()
       one_hot_tags=torch.squeeze(one_hot_tags,dim=1)
       pred = entity_predictor(words)
       pred_max_index = torch.argmax(pred, dim=1) + 1
       if tags.dim()>1:
         tags=tags.squeeze(dim=1)
       true_vals.extend(tags.flatten().cpu())
       predictions.extend(pred_max_index.flatten().cpu())
   predictions=torch.stack(predictions).numpy()
   true_vals=torch.stack(true_vals).numpy()
   f1_micro=f1_score(true_vals,predictions,average='micro')
   f1_macro=f1_score(true_vals,predictions,average='macro')
   rec_micro=recall_score(true_vals,predictions,average='micro')
   rec_macro=recall_score(true_vals,predictions,average='macro')
   pre_micro=precision_score(true_vals,predictions,average='micro')
   pre_macro=precision_score(true_vals,predictions,average='macro')
   print(f'Accuracy Score on dev set: {accuracy_score(true_vals,predictions)}')
   print(f'F1-Score(micro) on dev set: {f1_micro}')
   print(f'F1-Score(macro) on dev set: {f1_macro}')
   print(f'Recall(micro) Score on dev set: {rec_micro}')
   print(f'Recall(macro) Score on dev set: {rec_macro}')
   print(f'Precision(micro) Score on dev set: {pre_micro}')
   print(f'Precision(macro) Score on dev set: {pre_macro}')
   print(f'Confusion matrix for dev set:\n {confusion_matrix(true_vals,predictions)}')
   entity_predictor.eval()
   predictions=[]
   true_vals=[]
   with torch.no_grad():
     for batch_num, (words, tags) in enumerate(test_dataloader):
       (words, tags) = (words.to(device), tags.to(device))
       one_hot_tags=(torch.nn.functional.one_hot(tags - 1, num_classes=len(tag_to_id))).float()
       one_hot_tags=torch.squeeze(one_hot_tags,dim=1)
       pred = entity_predictor(words)
       pred_max_index = torch.argmax(pred, dim=1) + 1
       if tags.dim()>1:
         tags=tags.squeeze(dim=1)
       true_vals.extend(tags.flatten().cpu())
       predictions.extend(pred_max_index.flatten().cpu())
   predictions=torch.stack(predictions).numpy()
   true_vals=torch.stack(true_vals).numpy()
   f1_micro=f1_score(true_vals,predictions,average='micro')
   f1_macro=f1_score(true_vals,predictions,average='macro')
   rec_micro=recall_score(true_vals,predictions,average='micro')
   rec_macro=recall_score(true_vals,predictions,average='macro')
   pre_micro=precision_score(true_vals,predictions,average='micro')
   pre_macro=precision_score(true_vals,predictions,average='macro')
   print(f'Accuracy Score on test set: {accuracy_score(true_vals,predictions)}')
   print(f'F1-Score(micro) on test set: {f1_micro}')
   print(f'F1-Score(macro) on test set: {f1_macro}')
   print(f'Recall(micro) Score on test set: {rec_micro}')
   print(f'Recall(macro) Score on test set: {rec_macro}')
   print(f'Precision(micro) Score on test set: {pre_micro}')
   print(f'Precision(macro) Score on test set: {pre_macro}')
   print(f'Confusion matrix for test set:\n {confusion_matrix(true_vals,predictions)}')
   pred_vals_graph=[]
   for val in range(0,5):
     p=val
     s=val
     train_dataset,val_dataset,test_dataset,tag_to_id=get_datasets_ffNN(p,s)
     train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
     val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
     test_dataloader=DataLoader(test_dataset, batch_size=BATCH_SIZE)
     entity_predictor=FeedForwardNN(p,s,len(tag_to_id),len(train_dataset.vocabulary),EMBEDDING_SIZE,HIDDEN_SIZE)
     loss_fn = torch.nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(entity_predictor.parameters(), lr=lrate)
     entity_predictor = entity_predictor.to(device)
     # train loop
     for epoch_num in range(epochs):
       entity_predictor.train()
       for batch_num, (words, tags) in enumerate(train_dataloader):
         (words, tags) = (words.to(device), tags.to(device))
         one_hot_tags=(torch.nn.functional.one_hot(tags - 1, num_classes=len(tag_to_id))).float()
         one_hot_tags=torch.squeeze(one_hot_tags,dim=1)
         pred = entity_predictor(words)
         loss = loss_fn(pred, one_hot_tags)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()
     # accuracy on val set
     entity_predictor.eval()
     predictions=[]
     true_vals=[]
     with torch.no_grad():
       for batch_num, (words, tags) in enumerate(val_dataloader):
         (words, tags) = (words.to(device), tags.to(device))
         one_hot_tags=(torch.nn.functional.one_hot(tags - 1, num_classes=len(tag_to_id))).float()
         one_hot_tags=torch.squeeze(one_hot_tags,dim=1)
         pred = entity_predictor(words)
         pred_max_index = torch.argmax(pred, dim=1) + 1
         if tags.dim()>1:
           tags=tags.squeeze(dim=1)
         true_vals.extend(tags.flatten().cpu())
         predictions.extend(pred_max_index.flatten().cpu())
     predictions=torch.stack(predictions).numpy()
     true_vals=torch.stack(true_vals).numpy()
     pred_vals_graph.append(accuracy_score(true_vals,predictions))
   plt.bar(range(0,5), pred_vals_graph)
   plt.ylim(0.9, 1)
   plt.xlabel('context_window')
   plt.ylabel('Accuracy on dev set')
   plt.title('context_window vs dev set accuracy for FFNN')
   plt.show()


if sys.argv[1]=='-f':
   run_FFNN()
elif sys.argv[1]=='-r':
   run_LSTM()
elif sys.argv[1]=='-fm':
   FFNN_metrics()
elif sys.argv[1]=='-rm':
   LSTM_metrics()