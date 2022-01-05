import torch
import torch.nn as nn
import spacy
import sentencepiece
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd  # for lookup in annotation file

import math
import time
import madgrad
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from torch import einsum
from einops import rearrange


data_path_log = '/content/drive/My Drive/Colab Notebooks'

spacy_tr = spacy.load("xx_ent_wiki_sm")
FOLDNUM=5

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        rel_pos_num_buckets = 32
        rel_pos_max_distance = 256
        rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'
        self.rel_pos = RelativePositionBias(scale=64 ** 0.5, causal=False, heads=n_heads,
                                            num_buckets=rel_pos_num_buckets,
                                            max_distance=rel_pos_max_distance)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        energy = self.rel_pos(energy)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention

class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x
class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        # output = [batch size, trg len, output dim]

        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w',encoding="utf-8")
    file.write(data)
    file.close()

def remove_end_spaces(string):
    return "".join(string.rstrip())

def remove_beginning_spaces(string):
    return "".join(string.lstrip())


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.memory_summary(device=None, abbreviated=False)
print("Using device:", device)


class TextDataset(Dataset):
    def __init__(self, captions_file, freq_threshold=1):
        dfSentences = pd.read_csv(captions_file, encoding='utf-8')

        result = dfSentences.drop_duplicates(subset=["sentences"])
        print(result.size)
        result['wordlen'] = result['sentences'].str.split().str.len()
        # result = result.sort_values(by=['wordlen'], ascending=False)

        result = result[result['wordlen'] < 50]
        # result = result[result['wordlen'] > 2]
        result['anlen'] = result['analysis'].str.split().str.len()
        # result = result.sort_values(by=['anlen'], ascending=False)

        result = result[result['anlen'] < 300]
        print(result.size)
        Total = result['wordlen'].sum()
        print(Total)
        result = result.drop(['wordlen'], axis=1)
        result = result.drop(['anlen'], axis=1)
        result = result.dropna()
        result = result.reset_index(drop=True)
        # result = result.sort_values(by=['anlen'], ascending=False)
        self.df = result
        self.sentences = self.df["sentences"]
        self.analysis = self.df["analysis"]
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        train_analyst_list = self.analysis.tolist()
        self.vocab.build_vocabulary(self.analysis.tolist())  # train_analyst_list[3339:len(train_analyst_list)-1])
        self.sp = self.vocab.build_sentencepiece_vocabulary()

    def __len__(self):
        return len(self.df)

    def lenVal(self):
        return len(self.dfVal)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        analysis = self.analysis[index]
        numericalized_sentence = self.vocab.numericalizeWithSentencePiece(sentence)

        numericalized_analysis = [self.vocab.stoi["<SOS>"]]
        numericalized_analysis += self.vocab.numericalize(analysis)
        numericalized_analysis.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_sentence), torch.tensor(numericalized_analysis), sentence, analysis

    def getValidationItem(self, index):
        sentence = self.valSentences[index]
        analysis = self.valAnalysis[index]

        numericalized_sentence = self.vocab.numericalizeWithSentencePiece(sentence)

        numericalized_analysis = [self.vocab.stoi["<SOS>"]]
        numericalized_analysis += self.vocab.numericalize(analysis)
        numericalized_analysis.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_sentence), torch.tensor(numericalized_analysis)


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "Eow", 5: "Eor"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "Eow": 4, "Eor": 5}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_tr(text):
        return [tok.text for tok in spacy_tr.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 6

        for sentence in sentence_list:
            wordAnalysList = sentence.split('Eow')
            for x in range(0, len(wordAnalysList) - 1):
                elements = wordAnalysList[x].split('Eor ')
                root = remove_beginning_spaces(elements[0])
                if len(root) > 0:
                    for char in root:
                        if char not in frequencies:
                            frequencies[char] = 1
                        else:
                            frequencies[char] += 1

                        if frequencies[char] == self.freq_threshold:
                            self.stoi[char] = idx
                            self.itos[idx] = char
                            idx += 1
                remaining = remove_end_spaces(elements[1])
                for word in remaining.split():
                    if word not in frequencies:
                        frequencies[word] = 1

                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def build_sentencepiece_vocabulary(self):
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(data_path_log + "/m_user5M16KChar.model")
        return self.sp

    def numericalize(self, text):
        tokenized_text = self.tokenizer_tr(text)
        wordAnalysList = text.split('Eow')
        result = []
        for x in range(0, len(wordAnalysList) - 1):
            elements = wordAnalysList[x].split('Eor ')
            root = remove_beginning_spaces(elements[0])
            if len(root) > 0:
                root_numericalized = [self.stoi[char] if char in self.stoi else self.stoi["<UNK>"]
                                      for char in root]
                result.extend(root_numericalized)
                result.append(self.stoi['Eor'])
            morphemes = remove_end_spaces(elements[1])
            morphemes = morphemes.split()
            morphemes_numericalized = [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"]
                                       for word in morphemes]
            result.extend(morphemes_numericalized)
            result.append(self.stoi['Eow'])

        return result

    def numericalizeWithSentencePiece(self, text):
        indices = self.sp.EncodeAsIds(text)
        return indices


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        source = [item[0] for item in batch]
        source = pad_sequence(source, batch_first=True, padding_value=self.pad_idx)
        target = [item[1] for item in batch]
        target = pad_sequence(target, batch_first=True, padding_value=self.pad_idx)
        sentence = [item[2] for item in batch]
        analysis = [item[3] for item in batch]
        return source, target, sentence, analysis


def get_loader(
    annotation_file,
    fold_number,
    batch_size=8,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    test_split=.1,
):

    dataset = TextDataset( annotation_file)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    # Creating data indices for training and validation splits:
    random_seed=42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    arr_train = []
    arr_test = []
    kf10 = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in kf10.split(indices):
      arr_train.append(train_index)
      arr_test.append(test_index)
      print(train_index, test_index)
    '''split = int(np.floor(test_split * dataset_size))
    train_split = dataset_size-split
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)'''
    #train_indices test_indices = indices[split:], indices[:split]
    train_indices=arr_train[fold_number]
    test_indices = arr_test[fold_number]
    print(train_indices)
    print("*******")
    print(test_indices)
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)


    test_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        sampler=test_sampler
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        sampler=train_sampler
    )

    return train_loader,test_loader, dataset

data_file =data_path_log + '/trMor2018csv2.csv'
train_iterator,test_iterator, dataset = get_loader(annotation_file=data_file,fold_number=FOLDNUM)

max=0
maxa=0
for idx, (sentences, analysis,x,y) in enumerate(train_iterator):
    if sentences.shape[1]>max:
        max=sentences.shape[1]
    if analysis.shape[1]>maxa:
        maxa=analysis.shape[1]

print(max)
print(maxa)

for idx, (sentences, analysis,x,y) in enumerate(test_iterator):
    if sentences.shape[1]>max:
        max=sentences.shape[1]
    if analysis.shape[1]>maxa:
        maxa=analysis.shape[1]

print(max)
print(maxa)

INPUT_DIM = dataset.vocab.sp.GetPieceSize()
OUTPUT_DIM = dataset.vocab.__len__()
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 4
DEC_PF_DIM = 4
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
MAX_ENCODER = max+1
MAX_DECODER=maxa+1
enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device,
              MAX_ENCODER)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device,
              MAX_DECODER)

SRC_PAD_IDX = dataset.vocab.stoi["<PAD>"]
TRG_PAD_IDX = dataset.vocab.stoi["<PAD>"]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);

LEARNING_RATE = 0.0003

optimizer = madgrad.MADGRAD(model.parameters(), lr=LEARNING_RATE)#torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)#
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, (sentences, analysis,x,y) in enumerate(iterator):
        src = sentences.to(device)
        trg = analysis.to(device)

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, (sentences,analysis,x,y) in enumerate(iterator):
            src = sentences.to(device)
            trg = analysis.to(device)

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def create_analysis(model, sentence, dataset, device, max_length=525):
    model.eval()
    numericalized_sentence = dataset.vocab.numericalizeWithSentencePiece(sentence)
    src_tensor = torch.tensor(numericalized_sentence).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes  = [dataset.vocab.stoi["<SOS>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == dataset.vocab.stoi["<EOS>"]:
            break

    trg_tokens = [dataset.vocab.itos[i] for i in trg_indexes]
    # remove start token
    return trg_tokens[1:], attention


def eval_analyzer(data, model, dataset, device, prt, dataSize, foldNum):
    targets = []
    outputs = []
    total_words_count = 0
    total_correct_count = 0
    total_correct_count_re = 0
    problem_sentences = []
    total_correct_root_count = 0
    total_correct_morphemes_count = 0

    for i, (sentences, analysis, x, y) in enumerate(data):

        for j, k in zip(x, y):

            srcPrint = j
            prediction, attention = create_analysis(model, j, dataset, device, max_length=525)
            prediction = prediction[:-1]  # remove <eos> token
            wordAnalysList = k.split('Eow')

            trgStrArr = []
            trgPrtStr = ''
            trgRootArr = []
            trgMorphemeArr = []
            for x in range(0, len(wordAnalysList) - 1):
                elements = wordAnalysList[x].split('Eor ')
                rootTrg = remove_beginning_spaces(elements[0])
                morphemes = remove_end_spaces(elements[1])
                trgRootArr.append(rootTrg.lower())
                wordStr = rootTrg
                morphemes = morphemes.split()
                mrphmesTrg = ""
                for word in morphemes:
                    wordStr = wordStr + word
                    mrphmesTrg = mrphmesTrg + word
                trgStrArr.append(wordStr.lower())
                trgMorphemeArr.append(mrphmesTrg)
                trgPrtStr = trgPrtStr + ' ' + wordStr

            predictedList = []
            predStr = ''
            for x in prediction:
                predStr = predStr + x

            predPrtStr = ''
            predMorphemeArr = []
            predRootArr = []
            predWordList = predStr.split('Eow')
            for x in range(0, len(predWordList) - 1):
                elements = predWordList[x].split('Eor')
                if len(elements) == 2:
                    predRootArr.append(elements[0].lower())
                    predMorphemeArr.append(elements[1])
                    predicted = elements[0] + elements[1]
                    predPrtStr = predPrtStr + ' ' + predicted
                    predictedList.append(predicted.lower())

            correct_count_re = 0
            trgStrArrRe = []
            predStrArrRe = []
            for i in trgStrArr:
                i = i.replace("prop", "")
                trgStrArrRe.append(i)

            for i in predictedList:
                i = i.replace("prop", "")
                predStrArrRe.append(i)

            for i in trgStrArrRe:
                if i in predStrArrRe:
                    correct_count_re += 1

            correct_count = 0
            for i in trgStrArr:
                if i in predictedList:
                    correct_count += 1
            correct_root_count = 0
            correct_morphemes_count = 0
            for i in trgRootArr:
                if i in predRootArr:
                    correct_root_count += 1

            for i in trgMorphemeArr:
                if i in predMorphemeArr:
                    correct_morphemes_count += 1

            problem_sentences.append(srcPrint + "\n" + trgPrtStr + "\n" + predPrtStr + "\n")
            total_count = len(trgStrArr)

            total_words_count += total_count
            total_correct_count += correct_count
            total_correct_count_re += correct_count_re
            total_correct_root_count += correct_root_count
            total_correct_morphemes_count += correct_morphemes_count
            # problem_sentences.append(srcPrint+"\n"+trgPrtStr+"\n"+predPrtStr+"\n")

            # print("Sonuc: " + str(total_correct_root_count) + "/ " + str(total_correct_morphemes_count) + "/ " + str(
            #    total_correct_count) + "/ " + str(total_words_count))
            torch.cuda.empty_cache()
            # save_doc(problem_sentences, data_path_log + '/problemsentences20KSeperated.txt')
    acc_re = total_correct_count_re / total_words_count
    print(f"Accuracy : {acc_re:.4f}")
    save_doc(problem_sentences, data_path_log + '/finalproblemsentencesCharFoldC' + str(foldNum) + '.txt')

    return total_correct_count, total_words_count, total_correct_root_count, total_correct_morphemes_count


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

PATH = data_path_log+'/MorphCharFold'+str(FOLDNUM)+'loss.pt'

#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#loaded_epoch = checkpoint['epoch']
#loss = checkpoint['loss']
N_EPOCHS = 351
CLIP = 1
writer = SummaryWriter(data_path_log + "/runsmorphChar" + str(FOLDNUM) + "_plot")
step = 1
best_valid_loss = float('inf')
best_acc = 0.00000
for epoch in range( N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if (epoch % 10 == 0):
        correct, total, correct_root, correct_morphemes = eval_analyzer(test_iterator, model, dataset, device, False, 99,
                                                               FOLDNUM)
        acc = correct / total
        acc_root = correct_root / total
        acc_morph = correct_morphemes / total
        print(f"Accuracy : {acc:.4f}")
        print(f"Accuracy root : {acc_root:.4f}")
        print(f"Accuracy morphemes : {acc_morph:.4f}")
        writer.add_scalar("Test Accuracy", round(acc, 3), global_step=step)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), data_path_log + '/morphbestAccLoss' + str(FOLDNUM) + '.pt')
            print("BEST ACCURACY Model Saved loss down line")
            # import sys
            # sys.exit()

    scheduler.step(train_loss)
    step += 1
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, PATH)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')