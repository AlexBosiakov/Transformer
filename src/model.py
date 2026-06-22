import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        # Превращает id в векторы размерности d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Позиционные эмбеддинги, каждая позиция свой вектор
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()# Размер батча и длина последоватеьности в токенах
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # Сложение токенных и позиционных эмбеддингов
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        return x # Получаем готовые эмбеддинги: (batch_size, seq_len, d_model)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Проверка: размер эмбеддинга поделится поровну между всеми головами внимания
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # Размерность одной головы

        # линейные проекции для Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Линейная проекция для объединения всех голов
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Меняем форму у матриц с (batch, seq_len, d_model) на (batch, seq_len, self.num_heads, self.head_dim)
        # Меняем оси для матричного умножения (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Транспонируем K (batch, heads, head_dim, seq_len)
        # Вычисляем матрицу внимания Q @ K размерности (batch, heads, seq_len, seq_len)
        # Делим на корень размерности одной головы, для нормализации
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # От оценок получаем распределение вероятностей
        attn = scores.softmax(dim=-1)

        # Превращаем внимание в новое представление токенов (batch, heads, seq_len, head_dim)
        # Берем взвешенную сумму всех Value, где веса это вероятности внимания
        out = attn @ V

        # (batch, seq_len, heads, head_dim)
        # После транспонирования тензор становится непрерывным в памяти из-за этого нельзя применить .view()
        # Поэтому сначала создаем новый вектор, который лежит в памяти линейно
        # Получаем исходную размерность эмбеддинга d_model

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)

# Attention отвечает за взаимодействие между токенами
# FFN отвечает за локальную обработку информации внутри каждого токена

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim) # расширение
        self.fc2 = nn.Linear(hidden_dim, d_model) # сужение
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Residual connection
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_heads, ff_hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        cls = x[:, 0]
        logits = self.classifier(cls)
        return logits