import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import pickle
from tqdm import tqdm

# 加载保存的词汇表
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 定义LSTM模型（需要与训练时的定义一致）
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim * 3 + 2, 256)  # 3个LSTM输出和2个数值输入（ups, sentiment）
        self.fc2 = nn.Linear(256, 3)  # 输出3个类别

    def forward(self, comment, title, topic, ups, sentiment):
        comment_embed = self.embedding(comment)
        title_embed = self.embedding(title)
        topic_embed = self.embedding(topic)

        # LSTM的前向传播
        _, (h_n_comment, _) = self.lstm(comment_embed)
        _, (h_n_title, _) = self.lstm(title_embed)
        _, (h_n_topic, _) = self.lstm(topic_embed)

        # 获取LSTM的最后一个隐藏状态，形状为 [batch_size, lstm_hidden_dim]
        h_n_comment = h_n_comment[-1]
        h_n_title = h_n_title[-1]
        h_n_topic = h_n_topic[-1]

        # 现在ups和sentiment的形状是 [batch_size, 1]，我们只需要消除第2维度
        ups = ups.squeeze(1)  # [batch_size]
        sentiment = sentiment.squeeze(1)  # [batch_size]

        # 扩展ups和sentiment的维度到 [batch_size, 1]
        ups = ups.unsqueeze(1)  # 从 [batch_size] 变为 [batch_size, 1]
        sentiment = sentiment.unsqueeze(1)  # 从 [batch_size] 变为 [batch_size, 1]

        # 拼接所有的输入到一起，形成一个大的输入特征向量
        combined = torch.cat((h_n_comment, h_n_title, h_n_topic, ups, sentiment), dim=1)

        # 全连接层
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x

# 加载训练好的模型
vocab_size = len(vocab)
embed_dim = 512
lstm_hidden_dim = 256

model = LSTMModel(vocab_size=vocab_size, embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim)
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()  # 设置模型为评估模式

# 文本预处理函数
tokenizer = get_tokenizer('basic_english')

def text_pipeline(x):
    return vocab(tokenizer(x))

def preprocess_data(df):
    texts = list(df['self_text'].apply(lambda x: "nan" if pd.isna(x) else x))
    titles = list(df['post_title'].apply(lambda x: "nan" if pd.isna(x) else x))
    topics = list(df['subreddit'].apply(lambda x: "nan" if pd.isna(x) else x))
    comment_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in texts]
    title_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in titles]
    topic_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in topics]
    ups_data = torch.tensor(df[['post_thumbs_ups']].values, dtype=torch.float32).unsqueeze(1)
    sentiment_data = torch.tensor(df['overall_score'].values, dtype=torch.float32).unsqueeze(1)
    # 新的数据集可能没有标签，因此设置为0
    labels = torch.zeros(len(df), dtype=torch.long)
    return comment_data, title_data, topic_data, ups_data, sentiment_data, labels

# 自定义数据集类
class ReviewDataset(Dataset):
    def __init__(self, comment_data, title_data, topic_data, ups_data, sentiment_data, labels):
        self.comment_data = comment_data
        self.title_data = title_data
        self.topic_data = topic_data
        self.ups_data = ups_data
        self.sentiment_data = sentiment_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.comment_data[idx], self.title_data[idx], self.topic_data[idx], self.ups_data[idx], self.sentiment_data[idx], self.labels[idx]

# 数据加载器的collate函数
def collate_fn(batch):
    comment_data, title_data, topic_data, ups_data, sentiment_data, labels = zip(*batch)
    comment_data = pad_sequence(comment_data, batch_first=True, padding_value=0)
    title_data = pad_sequence(title_data, batch_first=True, padding_value=0)
    topic_data = pad_sequence(topic_data, batch_first=True, padding_value=0)
    ups_data = torch.stack(ups_data).squeeze(-1)
    sentiment_data = torch.stack(sentiment_data)
    labels = torch.stack(labels)
    return comment_data, title_data, topic_data, ups_data, sentiment_data, labels

# 读取新的数据集 先测试一下
new_data = pd.read_csv("../processed_batch_from_20/protest_batch_30.csv")  # 请替换为新数据集的路径

# 预处理新的数据集
comment_data_new, title_data_new, topic_data_new, ups_data_new, sentiment_data_new, labels_new = preprocess_data(new_data)

# 创建数据集和数据加载器
new_dataset = ReviewDataset(comment_data_new, title_data_new, topic_data_new, ups_data_new, sentiment_data_new, labels_new)
new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 进行预测
all_new_preds = []

with torch.no_grad():
    for batch in tqdm(new_loader):
        comment, title, topic, ups, sentiment, labels = batch
        outputs = model(comment, title, topic, ups, sentiment)
        _, preds = torch.max(outputs, 1)
        all_new_preds.extend(preds.tolist())

# 将预测结果添加到新的数据集中
new_data['predicted_protest_probability'] = all_new_preds

# 保存预测结果
#new_data.to_csv("../processed_batch_from_20/protest_batch_20.csv", index=False)

#调试一下看一下结果
from sklearn.metrics import f1_score
print("f1", f1_score(list(new_data['protest_probability']), all_new_preds, average="weighted"))
print("Predictions saved to 'predicted_new_dataset.csv'")
