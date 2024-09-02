import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments
import torch.nn as nn
import numpy as np
import pandas as pd

# 1. 准备数据
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 文本数据和标签, 读取整理好的数据 （0-19）
train = pd.read_csv("train_df.csv")
val = pd.read_csv("val_df.csv")
test = pd.read_csv("test_df.csv")

train_texts = list(train.self_text)
val_texts = list(val.self_text)
test_texts = list(test.self_text)

train_labels = list(train.overall_score)
val_labels = list(val.overall_score)
test_labels = list(test.overall_score)

# 数据归一化
train_labels = np.clip(train_labels, -1, 1)
val_labels = np.clip(val_labels, -1, 1)
test_labels = np.clip(test_labels, -1, 1)

# 初始化分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 128

# 创建训练和验证数据集
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

# 2. 定义RoBERTa回归模型
class RobertaForRegression(nn.Module):
    def __init__(self):
        super(RobertaForRegression, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记的表示
        logits = self.regressor(sequence_output).squeeze(-1)  # 去除最后一个维度

        # 计算 MSE 损失
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits

model = RobertaForRegression()


# 3. 设置训练参数和损失函数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    load_best_model_at_end=True,
    metric_for_best_model='mse',  # 确保是根据 MSE 选择最佳模型
    greater_is_better=False,      # 因为 MSE 越小越好
    label_names=["labels"],       # 确保 labels 被传递给模型
)

# 自定义损失函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    mse = ((preds - labels) ** 2).mean()
    return {'mse': mse, 'eval_mse': mse}

# 4. 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(f'best_model')

# 评估模型
results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test MSE: {results['eval_mse']}")
