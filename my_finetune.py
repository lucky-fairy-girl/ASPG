import os, argparse, h5py, codecs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk import ParentedTree
from subwordnmt.apply_bpe import BPE, read_vocabulary
from model import SynPG
from utils import Timer, make_path, load_embedding, load_dictionary, deleaf, sent2str, synt2str,reverse_bpe
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default="./model_new/", 
                       help="directory to save models")
parser.add_argument('--model_path', type=str, default="./model/pretrained_synpg.pt",#"./model_finetune/synpg_epoch140.pt",#"./model/pretrained_synpg.pt", 
                       help="initialized model path")
parser.add_argument('--output_dir', type=str, default="./output_finetune/",
                       help="directory to save outputs")
parser.add_argument('--bpe_codes_path', type=str, default='./data/bpe.codes',
                       help="bpe codes file")
parser.add_argument('--bpe_vocab_path', type=str, default='./data/vocab.txt',
                       help="bpe vcocabulary file")
parser.add_argument('--bpe_vocab_thresh', type=int, default=50, 
                       help="bpe threshold")
parser.add_argument('--dictionary_path', type=str, default="./data/dictionary.pkl", 
                       help="dictionary file")
parser.add_argument('--train_data_path', type=str, default="./data/glue_sst_train.h5",
                       help="training data")
parser.add_argument('--valid_data_path', type=str, default="./data/glue_sst_dev.h5",
                       help="validation data")
parser.add_argument('--max_sent_len', type=int, default=50,
                       help="max length of sentences")
parser.add_argument('--max_synt_len', type=int, default=234,
                       help="max length of syntax")
parser.add_argument('--word_dropout', type=float, default=0.4,
                       help="word dropout ratio")
parser.add_argument('--n_epoch', type=int, default=40,
                       help="number of epoches")
parser.add_argument('--batch_size', type=int, default=24,
                       help="batch size")
parser.add_argument('--lr', type=float, default=1e-4,
                       help="learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help="weight decay for adam")
parser.add_argument('--log_interval', type=int, default=250,
                       help="print log and validation loss evry 250 iterations")
parser.add_argument('--gen_interval', type=int, default=5000,
                       help="generate outputs every 500 iterations")
parser.add_argument('--save_interval', type=int, default=10000,
                       help="save model every 10000 iterations")
parser.add_argument('--temp', type=float, default=0.5,
                       help="temperature for generating outputs")
parser.add_argument('--seed', type=int, default=0, 
                       help="random seed")
args = parser.parse_args()
pprint(vars(args))
print()


#tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2",cache_dir = './huggingface/token')
#modelclass = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2",cache_dir = './huggingface/model')

#tokenizer = AutoTokenizer.from_pretrained("textattack/distilbert-base-cased-SST-2",cache_dir = './huggingface/token')
#modelclass = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-cased-SST-2",cache_dir = './huggingface/model')

tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2",cache_dir = './huggingface/token')
modelclass = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2",cache_dir = './huggingface/model')

def load_data(name):
    h5f = h5py.File(name, "r")
    data = (h5f["sent1"],h5f["synt1"])
    return data

def train(epoch, model1, train_data, valid_data, train_loader, valid_loader, optimizer, criterion, dictionary, bpe, args):
    
    timer = Timer()
    n_it = len(train_loader)
    attack=0
    
    for it, data_idxs in enumerate(train_loader):
        model1.train()
        
        data_idxs = np.sort(data_idxs.numpy())
        
        # get batch of raw sentences and raw syntax
        labels_ =train_data[0][data_idxs]
        print(labels_)
        sents_ = train_data[0][data_idxs]
        synts_ = train_data[1][data_idxs]
        batch_size =len(sents_)
        
        # initialize tensors
        sents = np.zeros((batch_size, args.max_sent_len), dtype=np.long)    # words without position
        synts = np.zeros((batch_size, args.max_synt_len+2), dtype=np.long)  # syntax
        targs = np.zeros((batch_size, args.max_sent_len+2), dtype=np.long)  # target output
        
        for i in range(batch_size):
            
            # bpe segment and convert to tensor
            sent_ = sents_[i].decode('utf-8')
            sent_ = bpe.segment(sent_).split()
            sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
            sents[i, :len(sent_)] = sent_
            
            # add <sos> and <eos> for target output
            targ_ = [dictionary.word2idx["<sos>"]] + sent_ + [dictionary.word2idx["<eos>"]]
            targs[i, :len(targ_)] = targ_
            
            # parse syntax and convert to tensor
            synt_ = synts_[i].decode('utf-8')
            synt_ = ParentedTree.fromstring(synt_)
            synt_ = deleaf(synt_)
            synt_ = [dictionary.word2idx[f"<{w}>"] for w in synt_ if f"<{w}>" in dictionary.word2idx]
            synt_ = [dictionary.word2idx["<sos>"]] + synt_ + [dictionary.word2idx["<eos>"]]
            synts[i, :len(synt_)] = synt_
            
        sents = torch.from_numpy(sents).cuda()
        synts = torch.from_numpy(synts).cuda()
        targs = torch.from_numpy(targs).cuda()
        sample = True  
        temp = 0.5       
      
        # forward
        torch.backends.cudnn.enabled = False
        outputs = model1(sents, synts, targs)

        if not sample:
            target2 = torch.max(outputs,2)[1]
        else:
            probs = F.softmax(outputs/temp,dim=2)
            target2=torch.zeros([batch_size,51],dtype=torch.int64).cuda()
            for i in range(batch_size):               
                a= torch.multinomial(probs[i],1).squeeze(1)
                target2[i,:]= a

        # calculate loss
        targs_ = targs[:, 1:].contiguous().view(-1)
        outputs_ = outputs.contiguous().view(-1, outputs.size(-1))
        optimizer.zero_grad()
        loss1 = criterion(outputs_, targs_)
        # calculate reward and loss2
        classgroup=np.zeros(batch_size)
        loss2=0
        for i in range(batch_size):
            label=labels_[i]
            idx=target2[i,:].detach().cpu().numpy()
            targ=targs[i,:]
            a=reverse_bpe(synt2str(idx, dictionary).split())+ '\n'
            detect= '<pad>' in a or '<unk>' in a
            paraphrase = tokenizer(a, return_tensors="pt")

            result = modelclass(**paraphrase)[0]#.logits
            result=result.detach().numpy()
            classout = np.argmax(result)
            classgroup[i]=classout
            if detect ==True:
                reward=-0.5
            elif classout != int(label):
                reward=1
                attack=attack+1
            else:
                reward=-1
            loss22=criterion(outputs[i],target2[i])
            loss2=loss2+reward*loss22
        print(classgroup)
        loss2=loss2/batch_size
     
        print('loss1/:{:.4f}|loss2/:{:.4f}'.format(loss1,loss2))
        loss =0.1*loss1-loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
        optimizer.step()

print("==== loading data ====")

# load bpe codes
bpe_codes = codecs.open(args.bpe_codes_path, encoding='utf-8')
bpe_vocab = codecs.open(args.bpe_vocab_path, encoding='utf-8')
bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

# load dictionary and data
dictionary = load_dictionary(args.dictionary_path)
train_data = load_data(args.train_data_path)
valid_data = load_data(args.valid_data_path)

train_idxs = np.arange(len(train_data[0]))
valid_idxs = np.arange(len(valid_data[0]))
print(f"number of train examples: {len(train_data[0])}")
print(f"number of valid examples: {len(valid_data[0])}")

train_loader = DataLoader(train_idxs, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_idxs, batch_size=args.batch_size, shuffle=False)

# load model
model1 = SynPG(len(dictionary), 300, word_dropout=args.word_dropout)
model1.load_state_dict(torch.load(args.model_path))

optimizer = torch.optim.Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss(ignore_index=dictionary.word2idx["<pad>"])

model1 = model1.cuda()
criterion = criterion.cuda()

# create folders
#make_path(args.model_dir)
#make_path(args.output_dir)

print("==== start training ====")
for epoch in range(1, args.n_epoch+1):
    # training
    print('epoch')
    print(epoch)

    train(epoch, model1, train_data, valid_data, train_loader, valid_loader, optimizer, criterion, dictionary, bpe, args)
    # save model
    if epoch%10==0:
        torch.save(model1.state_dict(), os.path.join(args.model_dir, "sst-albert{:02d}.pt".format(epoch)))
    # shuffle training data
