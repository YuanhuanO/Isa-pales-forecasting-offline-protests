import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import torchtext
import pickle
torchtext.disable_torchtext_deprecation_warning()

# train, val, test，分别包含了整理好的训练、验证和测试数据集
train = pd.read_csv("data/train_df.csv")
val = pd.read_csv("data/val_df.csv")
test = pd.read_csv("data/test_df.csv")

# # 标签映射函数
# def map_labels(labels):
#     return labels.map({0.0: 0, 0.5: 1, 1.0: 2})

# # 应用标签映射
# train['protest_probability'] = map_labels(train['protest_probability'])
# val['protest_probability'] = map_labels(val['protest_probability'])
# test['protest_probability'] = map_labels(test['protest_probability'])

# 文本预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train['self_text']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 保存词汇表
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)


def text_pipeline(x):
    return vocab(tokenizer(x))

# 将文本数据转换为整数序列
def preprocess_data(df):
    texts = list(df['self_text'].apply(lambda x: "nan" if pd.isna(x) else x))
    titles = list(df['post_title'].apply(lambda x: "nan" if pd.isna(x) else x))
    topics = list(df['subreddit'].apply(lambda x: "nan" if pd.isna(x) else x))
    comment_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in texts]
    title_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in titles]
    topic_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in topics]
    ups_data = torch.tensor(df[['post_thumbs_ups']].values, dtype=torch.float32).unsqueeze(1)
    sentiment_data = torch.tensor(df['overall_score'].values, dtype=torch.float32).unsqueeze(1)  # 添加维度
    labels = torch.tensor(df['protest_probability'].values, dtype=torch.long)
    return comment_data, title_data, topic_data, ups_data, sentiment_data, labels

comment_data_train, title_data_train, topic_data_train, ups_data_train, sentiment_data_train, labels_train = preprocess_data(train)
comment_data_val, title_data_val, topic_data_val, ups_data_val, sentiment_data_val, labels_val = preprocess_data(val)
comment_data_test, title_data_test, topic_data_test, ups_data_test, sentiment_data_test, labels_test = preprocess_data(test)

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
    # ups_data = torch.stack(ups_data)
    # sentiment_data = torch.stack(sentiment_data)
    # 修改这里确保ups_data和sentiment_data维度是[batch_size, 1]
    ups_data = torch.stack(ups_data).squeeze(-1)  # 改为 .squeeze(-1) 将维度从 [batch_size, 1, 1] -> [batch_size, 1]
    sentiment_data = torch.stack(sentiment_data)  # 已经是 [batch_size, 1]，这里加上squeeze去掉多余的维度
    
    labels = torch.stack(labels)
    return comment_data, title_data, topic_data, ups_data, sentiment_data, labels

# 创建数据集和数据加载器
train_dataset = ReviewDataset(comment_data_train, title_data_train, topic_data_train, ups_data_train, sentiment_data_train, labels_train)
val_dataset = ReviewDataset(comment_data_val, title_data_val, topic_data_val, ups_data_val, sentiment_data_val, labels_val)
test_dataset = ReviewDataset(comment_data_test, title_data_test, topic_data_test, ups_data_test, sentiment_data_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

# 构建LSTM模型
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




# 模型训练设置
vocab_size = len(vocab)
embed_dim = 512
lstm_hidden_dim = 256


# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型放到所有可用的GPU上
model = LSTMModel(vocab_size, embed_dim, lstm_hidden_dim).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# 训练模型并保存最佳模型
num_epochs = 20
best_val_loss = float('inf')
best_model_path = 'best_lstm_model.pth'

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    all_train_labels = []
    all_train_preds = []

    for batch in tqdm(train_loader):
        comment, title, topic, ups, sentiment, labels = batch

        # 将数据移动到GPU
        comment, title, topic, ups, sentiment, labels = comment.to(device), title.to(device), topic.to(device), ups.to(device), sentiment.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(comment, title, topic, ups, sentiment)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
        # 获取预测结果
        _, preds = torch.max(outputs, 1)
        all_train_labels.extend(labels.cpu().tolist())  # 移动到CPU
        all_train_preds.extend(preds.cpu().tolist())

    avg_train_loss = total_train_loss / len(train_loader)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train F1 Score: {train_f1:.4f}')
    
    # 验证模型
    model.eval()
    total_val_loss = 0
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            comment, title, topic, ups, sentiment, labels = batch
            
            # 将数据移动到GPU
            comment, title, topic, ups, sentiment, labels = comment.to(device), title.to(device), topic.to(device), ups.to(device), sentiment.to(device), labels.to(device)
            
            outputs = model(comment, title, topic, ups, sentiment)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            all_val_labels.extend(labels.cpu().tolist())
            all_val_preds.extend(preds.cpu().tolist())

    avg_val_loss = total_val_loss / len(val_loader)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation F1 Score: {val_f1:.4f}')
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

# 测试集评估
model.eval()
total_test_loss = 0
all_test_labels = []
all_test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        comment, title, topic, ups, sentiment, labels = batch
        
        # 将数据移动到GPU
        comment, title, topic, ups, sentiment, labels = comment.to(device), title.to(device), topic.to(device), ups.to(device), sentiment.to(device), labels.to(device)
        
        outputs = model(comment, title, topic, ups, sentiment)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        
        # 获取预测结果
        _, preds = torch.max(outputs, 1)
        all_test_labels.extend(labels.cpu().tolist())
        all_test_preds.extend(preds.cpu().tolist())

avg_test_loss = total_test_loss / len(test_loader)
test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
print(f'Test Loss: {avg_test_loss:.4f}, Test F1 Score: {test_f1:.4f}')

def map_labels(labels):
    return labels.map({0.0: 0, 0.5: 1, 1.0: 2})

# 应用标签映射
train['protest_probability'] = map_labels(train['protest_probability'])
val['protest_probability'] = map_labels(val['protest_probability'])
test['protest_probability'] = map_labels(test['protest_probability'])

# 文本预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train['self_text']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 保存词汇表
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)


def text_pipeline(x):
    return vocab(tokenizer(x))

# 将文本数据转换为整数序列
def preprocess_data(df):
    texts = list(df['self_text'].apply(lambda x: "nan" if pd.isna(x) else x))
    titles = list(df['post_title'].apply(lambda x: "nan" if pd.isna(x) else x))
    topics = list(df['subreddit'].apply(lambda x: "nan" if pd.isna(x) else x))
    comment_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in texts]
    title_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in titles]
    topic_data = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in topics]
    ups_data = torch.tensor(df[['post_thumbs_ups']].values, dtype=torch.float32).unsqueeze(1)
    sentiment_data = torch.tensor(df['overall_score'].values, dtype=torch.float32).unsqueeze(1)  # 添加维度
    labels = torch.tensor(df['protest_probability'].values, dtype=torch.long)
    return comment_data, title_data, topic_data, ups_data, sentiment_data, labels

comment_data_train, title_data_train, topic_data_train, ups_data_train, sentiment_data_train, labels_train = preprocess_data(train)
comment_data_val, title_data_val, topic_data_val, ups_data_val, sentiment_data_val, labels_val = preprocess_data(val)
comment_data_test, title_data_test, topic_data_test, ups_data_test, sentiment_data_test, labels_test = preprocess_data(test)

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
    # ups_data = torch.stack(ups_data)
    # sentiment_data = torch.stack(sentiment_data)
    # 修改这里确保ups_data和sentiment_data维度是[batch_size, 1]
    ups_data = torch.stack(ups_data).squeeze(-1)  # 改为 .squeeze(-1) 将维度从 [batch_size, 1, 1] -> [batch_size, 1]
    sentiment_data = torch.stack(sentiment_data)  # 已经是 [batch_size, 1]，这里加上squeeze去掉多余的维度
    
    labels = torch.stack(labels)
    return comment_data, title_data, topic_data, ups_data, sentiment_data, labels

# 创建数据集和数据加载器
train_dataset = ReviewDataset(comment_data_train, title_data_train, topic_data_train, ups_data_train, sentiment_data_train, labels_train)
val_dataset = ReviewDataset(comment_data_val, title_data_val, topic_data_val, ups_data_val, sentiment_data_val, labels_val)
test_dataset = ReviewDataset(comment_data_test, title_data_test, topic_data_test, ups_data_test, sentiment_data_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

# 构建LSTM模型
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

# 检查一下维度特征
for batch in train_loader:
    comment, title, topic, ups, sentiment, labels = batch
    print("comment shape:", comment.shape)
    print("title shape:", title.shape)
    print("topic shape:", topic.shape)
    print("ups shape:", ups.shape)
    print("sentiment shape:", sentiment.shape)
    break  # 检查一次就可以了




# 模型训练设置
vocab_size = len(vocab)
embed_dim = 512
lstm_hidden_dim = 256

model = LSTMModel(vocab_size, embed_dim, lstm_hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# 训练模型并保存最佳模型
num_epochs = 20
best_val_loss = float('inf')
best_model_path = 'best_lstm_model.pth'

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    all_train_labels = []
    all_train_preds = []

    for batch in tqdm(train_loader):
        comment, title, topic, ups, sentiment, labels = batch

        optimizer.zero_grad()
        outputs = model(comment, title, topic, ups, sentiment)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
        # 获取预测结果
        _, preds = torch.max(outputs, 1)
        all_train_labels.extend(labels.tolist())
        all_train_preds.extend(preds.tolist())

    avg_train_loss = total_train_loss / len(train_loader)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train F1 Score: {train_f1:.4f}')
    
    # 验证模型
    model.eval()
    total_val_loss = 0
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            comment, title, topic, ups, sentiment, labels = batch
            outputs = model(comment, title, topic, ups, sentiment)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            all_val_labels.extend(labels.tolist())
            all_val_preds.extend(preds.tolist())

    avg_val_loss = total_val_loss / len(val_loader)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation F1 Score: {val_f1:.4f}')
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

# 测试集评估
model.eval()
total_test_loss = 0
all_test_labels = []
all_test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        comment, title, topic, ups, sentiment, labels = batch
        outputs = model(comment, title, topic, ups, sentiment)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        
        # 获取预测结果
        _, preds = torch.max(outputs, 1)
        all_test_labels.extend(labels.tolist())
        all_test_preds.extend(preds.tolist())

avg_test_loss = total_test_loss / len(test_loader)
test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
print(f'Test Loss: {avg_test_loss:.4f}, Test F1 Score: {test_f1:.4f}')

