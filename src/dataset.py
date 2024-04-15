import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        # đọc dữ liệu từ file csv chuyển vào texts và labels
        texts, labels = [], []
        with open(data_path, encoding='utf-8-sig') as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx.lower()
                    text += " "
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        
        # đọc từ điển từ tệp csv, từ điển dùng để mã hóa các từ trong văn bản
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        
        # chia văn bản thành các câu, chia các câu thành các từ, mỗi từ được mã hóa theo từ điển, không nó được mã hóa bằng -1
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        # nếu độ dài câu nhỏ hơn max_length_word, thêm các từ giả (-1) vào cuối cho đủ
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        # nếu số lượng câu nhỏ hơn max_length_sentences, thêm các từ giả (-1) vào câu và thêm câu giả (-1,-1,...,-1) vào đoạn cho đủ
        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        # cắt đoạn văn bản đã mã hóa để nó không dài quá max
        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label
    