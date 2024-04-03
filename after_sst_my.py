import os, argparse, codecs,h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk import ParentedTree
from subwordnmt.apply_bpe import BPE, read_vocabulary
from model import SynPG
from utils import Timer, make_path, load_data, load_embedding, load_dictionary, tree2tmpl, getleaf, synt2str, reverse_bpe
from tqdm import tqdm
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--synpg_model_path', type=str, default="./model_new/sst-albert30.pt",#"finetune_full/synpg_epoch-135.pt",#"./mrpc_finetune/synpg_epoch240.pt", #"./model/pretrained_synpg.pt", 
                       help="prtrained SynPG")
parser.add_argument('--pg_model_path', type=str, default="./model_parse/parse2_sst40.pt",#"./model/pretrained_parse_generator.pt",#
                       help="prtrained parse generator")
parser.add_argument('--train_data_path', type=str, default="./data/sst_albert_right.h5",
                       help="input file")
parser.add_argument('--output_path', type=str, default="./sstafter/",
                       help="output file")
parser.add_argument('--bpe_codes_path', type=str, default='./data/bpe.codes',
                       help="bpe codes file")
parser.add_argument('--bpe_vocab_path', type=str, default='./data/vocab.txt',
                       help="bpe vcocabulary file")
parser.add_argument('--bpe_vocab_thresh', type=int, default=50, 
                       help="bpe threshold")
parser.add_argument('--dictionary_path', type=str, default="./data/dictionary.pkl", 
                       help="dictionary file")
parser.add_argument('--max_sent_len', type=int, default=50,
                       help="max length of sentences")
parser.add_argument('--max_tmpl_len', type=int, default=100,
                       help="max length of tempalte")
parser.add_argument('--max_synt_len', type=int, default=234,
                       help="max length of syntax")
parser.add_argument('--temp', type=float, default=0.5,
                       help="temperature for generating outputs")
parser.add_argument('--batch_size', type=int, default=64,
                       help="batch size")
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
modelclass.load_state_dict(torch.load('./class/class_sst-albert08.pt'),strict=False)


class mymodel(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,pairs):
        out = modelclass(**pairs)
        return out
mymodel=mymodel()


def load_data(name):
    h5f = h5py.File(name, "r")
    data = (h5f["label"],h5f["sent1"],h5f["synt1"])
    return data
def cal_bleu(hyp, ref, n):
    hyp = hyp.strip().split(' ')
    ref = ref.strip().split(' ')
    
    if n == 0:
        return sentence_bleu([ref], hyp)
    elif n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (0, 1, 0, 0)
    elif n == 3:
        weights = (0, 0, 1, 0)
    elif n == 4:
        weights = (0, 0, 0, 1)

    return sentence_bleu([ref], hyp, weights=weights)



templates = [
    "( ROOT ( S ( NP ) ( VP ) ( . ) ) )",
    "( ROOT ( S ( VP ) ) )",
    "( ROOT ( SINV ( VP ) ( NP ) ) )",
    "( ROOT ( NP ( NP ) ( PP ) ) )",
    "( ROOT ( S ( VP ) ( . ) ) )",
]

def template2tensor(templates, max_tmpl_len, dictionary):
    tmpls = np.zeros((len(templates), max_tmpl_len+2), dtype=np.long)
    for i, tp in enumerate(templates):
        tmpl_ = ParentedTree.fromstring(tp)
        tree2tmpl(tmpl_, 1, 2)
        tmpl_ = str(tmpl_).replace(")", " )").replace("(", "( ").split(" ")
        tmpl_ = [dictionary.word2idx[f"<{w}>"] for w in tmpl_ if f"<{w}>" in dictionary.word2idx]
        tmpl_ = [dictionary.word2idx["<sos>"]] + tmpl_ + [dictionary.word2idx["<eos>"]]
        tmpls[i, :len(tmpl_)] = tmpl_
    
    tmpls = torch.from_numpy(tmpls).cuda()
    
    return tmpls

def generate(sent, synt, tmpls, synpg_model, pg_model, args):
    with torch.no_grad():
        
        # convert syntax to tag sequence
        tagss = np.zeros((len(tmpls), args.max_sent_len), dtype=np.long)
        tags_ = ParentedTree.fromstring(synt)
        tags_ = getleaf(tags_)
        tags_ = [dictionary.word2idx[f"<{w}>"] for w in tags_ if f"<{w}>" in dictionary.word2idx]
        tagss[:, :len(tags_)] = tags_[:args.max_sent_len]
        
        tagss = torch.from_numpy(tagss).cuda()
        # generate parses from tag sequence and templates
        parse_idxs = pg_model.generate(tagss, tmpls, args.max_synt_len, temp=args.temp)
        
        # add <sos> and remove tokens after <eos>
        synts = np.zeros((len(tmpls), args.max_synt_len+2), dtype=np.long)
        synts[:, 0] = 1
        for i in range((len(tmpls))):
            parse_idx = parse_idxs[i].cpu().numpy()
            eos_pos = np.where(parse_idx==dictionary.word2idx["<eos>"])[0]
            eos_pos = eos_pos[0]+1 if len(eos_pos) > 0 else len(parse_idx)
            synts[i, 1:eos_pos+1] = parse_idx[:eos_pos]
            
        synts = torch.from_numpy(synts).cuda()
        # bpe segment and convert sentence to tensor
        sents = np.zeros((len(tmpls), args.max_sent_len), dtype=np.long)
        sent_ = bpe.segment(sent).split()
        sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
        sents[:, :len(sent_)] = sent_[:args.max_sent_len]
        sents = torch.from_numpy(sents).cuda()
        
        # generate paraphrases from sentence and generated parses
        output_idxs = synpg_model.generate(sents, synts, args.max_sent_len, temp=args.temp)
        output_idxs = output_idxs.cpu().numpy()
        
        paraphrases = [reverse_bpe(synt2str(output_idxs[i], dictionary).split()) for i in range(len(tmpls))]
        return paraphrases


print("==== loading models ====")

# load bpe codes
bpe_codes = codecs.open(args.bpe_codes_path, encoding='utf-8')
bpe_vocab = codecs.open(args.bpe_vocab_path, encoding='utf-8')
bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

# load dictionary and models
dictionary = load_dictionary(args.dictionary_path)

synpg_model = SynPG(len(dictionary), 300, word_dropout=0.0)
synpg_model.load_state_dict(torch.load(args.synpg_model_path))
synpg_model = synpg_model.cuda()
synpg_model.eval()

train_data = load_data(args.train_data_path)
train_idxs = np.arange(len(train_data[0]))
train_loader = DataLoader(train_idxs, batch_size=args.batch_size, shuffle=False)
print(f"number of train examples: {len(train_data[0])}")


pg_model = SynPG(len(dictionary), 300, word_dropout=0.0)
pg_model.load_state_dict(torch.load(args.pg_model_path))
pg_model = pg_model.cuda()
pg_model.eval()

print("==== generate paraphrases ====")

# convert template strings to tensors
tmpls = template2tensor(templates, args.max_tmpl_len, dictionary)

with open(os.path.join(args.output_path, f"output_my_albertattack-parse2.txt"), "w") as fp1, \
    open(os.path.join(args.output_path, f"output_my_albertsen-parse2.txt"), "w") as fp2,\
    open(os.path.join(args.output_path, f"output_my_sst-albert2.txt"), "w") as fp3:
    


    attack = 0
    for it, data_idxs in enumerate(train_loader):
        data_idxs = np.sort(data_idxs.numpy())

        sents = train_data[1][data_idxs]
        synts = train_data[2][data_idxs]
        labels = train_data[0][data_idxs]

        batch_size=len(sents)
        print(f"the it number:{it}")

        for i in range(batch_size):
            sent=sents[i].decode('utf-8')
            synt=synts[i].decode('utf-8')
            label=labels[i]#.decode('utf-8')

            # generate paraphrases
            paraphrases = generate(sent, synt, tmpls, synpg_model, pg_model, args)
            att=0
            for j in range(len(tmpls)):
                # write to output file
                results = tokenizer(paraphrases[j], return_tensors="pt")
                result = mymodel(results)[0]#.logits
                result=result.detach().numpy()
                result=np.argmax(result)
                if int(result)!=int(label):
                    att+=1
                    fp1.write(paraphrases[j]+'\n')
                    fp2.write(sent+'\n')
                    fp3.write(str(label)+'	'+paraphrases[j]+ '\n')
            if att!=0:
                attack+=1
                print('attack')

        print(attack)
print(attack) 