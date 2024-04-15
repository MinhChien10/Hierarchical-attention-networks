import torch
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np

# đánh giá hiệu suất của 1 mô hình phân loại dựa trên tập hơp các chỉ số được chỉ định
def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

# nhân ma trận giữa feature trong 1 batch và 1 trọng số, thêm bias nếu có và áp dụng hàm tanh
def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight) # nhân ma trận giữa feature và weigh
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0) # dùng hàm tanh lên feature
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze() # nối các feature trong list theo chiều đầu tiên

# nhân từng phần tử giữa 2 tensor đầu vào
def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1) # thêm 1 chiều vào feature_2 tại vị trí 1 và mở rộng để cùng kích thước feature_1
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

# tính độ dài tối đa của câu và từ
def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path, encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        # sắp xếp độ dài từ và câu theo thứ tự tăng dần
        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]
