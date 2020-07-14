import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import data
import model
import time
import genotypes
from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='ptb/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=850,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=850,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=850,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.2,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--model_path', type=str, default='weight/model.pt',
                    help='path to load the pretrained model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--n_experts', type=int, default=1,
                    help='number of experts')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--arch', type=str, default='DARTS_A2', help='which architecture to use')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes rnn layers (0 = no dropout)')
args = parser.parse_args()


def logging(s, print_=True, log_=True):
    print(s)


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

corpus = data.Corpus(args.data)

# word_2_id
word = corpus.dictionary.word2idx
word_2_id = dict(zip(word.values(), word.keys()))

test_batch_size = 1
test_data = batchify(corpus.test, test_batch_size, args)
########################################################
# my_data = batchify(corpus.mytxt, test_batch_size, args)

ntokens = len(corpus.dictionary)


def evaluate_my(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        # print(i, data_source.size(0)-1)
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        ########################################
        data_ = data.data.cpu().numpy()
        txt_data = []
        for t in data_:
            tt = word_2_id[t[0]]
            txt_data.append(tt)
        # print(txt_data)
        ########################################
        target = targets.data.cpu().numpy()
        txt_tar = []
        for t in target:
            tt = word_2_id[t]
            txt_tar.append(tt)
        # print(txt_tar)
        out = log_prob.view(-1, log_prob.size(2))
        output = []
        for i in range(out.shape[0]):  # i表示第i个单词的索引
            i_vetor = out[i, :].data.cpu().numpy()  # ii表示第i个单词的词向量
            i_vetor_list = []  # i_vetor_list
            for n in i_vetor:
                i_vetor_list.append(n)
            max_i = sorted(i_vetor_list)[-1]
            index = i_vetor_list.index(max_i)
            index_word = word_2_id[index]
            output.append(index_word)
            # 输出最后一位的十种可能情况
            if i == int(out.shape[0]) - 1:
                word_pre_end = []
                possible_end_word = sorted(i_vetor_list, reverse=True)[:10]
                for end_word in possible_end_word:
                    end_word_index = i_vetor_list.index(end_word)
                    possible_word = word_2_id[end_word_index]
                    word_pre_end.append(possible_word)
        word_pre = [i for i in output]
        # word_pre_1 = [output[i] for i in range(len(output))]
        return word_pre_end

        ##################################


def new_txt(txt_path, data):
    result2txt = str(data)  # data是前面运行出的数据，先将其转为字符串才能写入
    with open(txt_path + 'myself.txt', 'w') as file_handle:  # .txt可以不自己新建,代码会自动新建
        file_handle.truncate()
        file_handle.write(result2txt)  # 写入
        # file_handle.write('\n')         # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
        data_path = txt_path + 'myself.txt'
    with open(data_path, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
    return data


# Load the best saved model.

model = torch.load(args.model_path, map_location='cpu')
parallel_model = model
# os.remove(args.data + 'myself.txt')
#################################################################
all_txt = []


def get_next(text):
    try:
        all_txt.append(text)
        pick = []
        for i in all_txt:
            num = word[i]
            pick.append(num)
        pick.append(24)
        bbb = torch.tensor(pick, dtype=torch.long)
        length = len(pick)
        my_data = bbb.view(length, 1)

        pre_word = evaluate_my(my_data, test_batch_size)
        return pre_word

    except:
        print('很抱歉，暂时未收录该句子中的单词')
        all_txt.pop(-1)
        return None

