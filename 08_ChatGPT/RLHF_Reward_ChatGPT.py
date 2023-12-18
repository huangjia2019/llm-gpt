import torch # 导入torch
from transformers import GPT2Tokenizer # 导入GPT2分词器
from transformers import GPT2LMHeadModel # 导入GPT2语言模型

model_name = "gpt2"  # 也可以选择其他模型，如"gpt2-medium"、"gpt2-large"等
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # 加载分词器
device = "cuda" if torch.cuda.is_available() else "cpu" # 判断是否有可用GPU
model = GPT2LMHeadModel.from_pretrained(model_name).to(device) # 将模型加载到设备上（CPU或GPU）
vocab = tokenizer.get_vocab() # 获取词汇表

# 示例RLHF数据集
data = [
    {
        "User": "What is the capital of France?",
        # "AI": "The capital of France is Paris.",
        "AI": "Paris.",
        "score": 5
    },
    {
        "User": "What is the capital of France?",
        "AI": "Rome.",
        "score": 1
    },
    {
        "User": "How to cook pasta?",
        # "AI": "To cook pasta, first boil water and then add pasta.",
        "AI": "first boil water.",
        "score": 4
    },
    {
        "User": "How to cook pasta?",
        # "AI": "First, turn on the microwave and put the pasta inside.",
        "AI": "microwave.",
        "score": 2
    }
]


from torch.utils.data import Dataset  # 导入Pytorch的Dataset
class RLHFDataset(Dataset):
    def __init__(self, data, tokenizer, vocab):
        self.tokenizer = tokenizer  # 分词器
        self.vocab = vocab  # 词汇表
        self.input_data, self.target_data, self.scores = self.process_data(data)
        
    def process_data(self, data):        
        input_data, target_data, scores = [], [], []       
        for conversation in data:
            user_question = conversation["User"]
            model_answer = conversation["AI"]
            score = conversation["score"]

            input_tokens = self.tokenizer(f"{user_question}", return_tensors="pt")["input_ids"].tolist()[0]
            input_tokens = input_tokens + [tokenizer.eos_token_id]
            input_data.append(torch.tensor(input_tokens, dtype=torch.long))

            target_tokens = self.tokenizer(model_answer, return_tensors="pt")["input_ids"].tolist()[0]
            target_tokens = target_tokens + [tokenizer.eos_token_id]
            target_data.append(torch.tensor(target_tokens, dtype=torch.long))

            scores.append(score)

        return input_data, target_data, scores
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx], self.scores[idx]

rlhf_dataset = RLHFDataset(data, tokenizer, vocab) # 创建ChatDataset对象，传入文件、分词器和词汇表
# 打印数据集中前2个数据示例
for i in range(2):
    input_example, target_example, _ = rlhf_dataset[i]
    print(f"Example {i + 1}:")
    print("Input:", tokenizer.decode(input_example))
    print("Target:", tokenizer.decode(target_example))

from torch.utils.data import DataLoader # 导入Dataloader
tokenizer.pad_token = '<pad>' # 为分词器添加pad token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
# 定义pad_sequence函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if length is None else length    
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)    
    # 遍历序列，将每个序列的内容复制到结果张量中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

# 定义collate_fn函数，用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批次中分离源序列、目标序列和分数
    sources, targets, scores = zip(*batch)    
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))    
    # 使用 pad_sequence 函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=tokenizer.pad_token_id, length=max_length)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id, length=max_length)
    # 将分数转换为张量
    scores = torch.tensor(scores, dtype=torch.float)
    # 返回补齐后的源序列、目标序列和分数
    return sources, targets, scores

# 创建Dataloader
batch_size = 2
chat_dataloader = DataLoader(rlhf_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 检查Dataloader输出
for input_batch, target_batch, score_batch in chat_dataloader:
    print("Input batch tensor size:", input_batch.size())
    print("Target batch tensor size:", target_batch.size())
    print("Score batch tensor size:", score_batch.size())
    break


# 奖励函数
# def reward_function(predictions, targets, scores):
#     correct = (predictions == targets).float() * scores.unsqueeze(1)
#     reward = correct.sum(dim=-1) / (targets != tokenizer.pad_token_id).sum(dim=-1).float()
#     return reward

# def reward_function(predictions, targets, scores):
#     correct = (predictions == targets).float()
#     num_correct = correct.sum(dim=-1)
#     num_total = (targets != tokenizer.pad_token_id).sum(dim=-1).float()
#     match_ratio = num_correct / num_total
#     reward = match_ratio * scores
#     return reward

def reward_function(predictions, targets, scores):
    correct = (predictions == targets).float() * scores.unsqueeze(1)
    reward = correct.sum(dim=-1) / (targets != tokenizer.pad_token_id).sum(dim=-1).float()
    return reward / scores.max()


import numpy as np
import torch.nn as nn
import torch.optim as optim
# 训练过程
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
for epoch in range(num_epochs):
    epoch_rewards = []
    
    for batch_idx, (input_batch, target_batch, score_batch) in enumerate(chat_dataloader):
        optimizer.zero_grad()
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        score_batch = score_batch.to(device)
        
        outputs = model(input_batch)
        logits = outputs.logits
        
        _, predicted_tokens = torch.max(logits, dim=-1)
        
        # 计算奖励
        rewards = reward_function(predicted_tokens, target_batch, score_batch)
        
        # 计算损失
        loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        
        # 计算加权损失
        weighted_loss = torch.sum(loss * (1 - rewards)) / rewards.numel()
        
        # 反向传播和优化
        weighted_loss.backward()
        # loss.backward()
        optimizer.step()
        
        epoch_rewards.append(rewards.cpu().numpy())
    
    avg_reward = np.mean(np.concatenate(epoch_rewards))
    if (epoch + 1) % 20 == 0:
        print(f'Epoch: {epoch + 1:04d}, cost = {weighted_loss:.6f}, avg_reward = {avg_reward:.6f}')



def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval()  # 将模型设置为评估模式（不计算梯度）
    # 对输入字符串进行编码，并将其转换为 PyTorch 张量，然后将其移动到相应的设备上（例如 GPU）
    input_tokens = tokenizer.encode(input_str, return_tensors="pt").to(device)    
    # 初始化候选序列列表，包含当前输入序列和其对数概率得分（我们从0开始）
    candidates = [(input_tokens, 0.0)]    
    # 禁用梯度计算，以加速预测过程
    with torch.no_grad():
        # 迭代生成最大长度的序列
        for _ in range(max_len):
            new_candidates = []            
            # 对于每个候选序列
            for candidate, candidate_score in candidates:
                # 使用模型进行预测
                outputs = model(candidate)
                # 获取输出 logits
                logits = outputs.logits[:, -1, :]
                # 获取对数概率得分的 top-k 值（即 beam_width）及其对应的 token
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                final_results = []
                # 遍历 top-k token 及其对应的得分
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    # 在当前候选序列中添加新的 token
                    new_candidate = torch.cat((candidate, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    # 更新候选序列的得分
                    new_score = candidate_score - score.item()                    
                    # 如果新的 token 是结束符（eos_token），则将该候选序列添加到最终结果中
                    if next_token.item() == tokenizer.eos_token_id:
                        final_results.append((new_candidate, new_score))
                    # 否则，将新的候选序列添加到新候选序列列表中
                    else:
                        new_candidates.append((new_candidate, new_score))            
            # 从新候选序列列表中选择得分最高的 top-k 个序列
            candidates = sorted(new_candidates, key=lambda x: x[1])[:beam_width]    
    # 选择得分最高的候选序列
    best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]    
    # 将输出 token 转换回文本字符串
    output_str = tokenizer.decode(best_candidate[0])    
    # 移除输入字符串并修复空格问题
    input_len = len(tokenizer.encode(input_str))
    output_str = tokenizer.decode(best_candidate.squeeze()[input_len:])    
    return output_str

test_inputs = [
    "What is the capital of France?",
    "How to cook pasta?",
    "hi , what is your name?"
]

for i, input_str in enumerate(test_inputs, start=1):
    generated_text = generate_text_beam_search(model, input_str)
    print(f"Test {i}:")
    print(f"User: {input_str}")
    print(f"AI: {generated_text}")
    print()