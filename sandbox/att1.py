import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.nn import softmax, gelu
from datasets import load_dataset

# ----------------------------
# 1. Model Components
# ----------------------------

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta

def multi_head_attention(Q, K, V, Wq, Wk, Wv, Wo, mask=None):
    # Project inputs
    q = jnp.dot(Q, Wq)  # (seq, d_model)
    k = jnp.dot(K, Wk)
    v = jnp.dot(V, Wv)

    # Scaled dot-product attention
    d_k = q.shape[-1]
    scores = jnp.dot(q, k.T) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    attn_weights = softmax(scores, axis=-1)
    output = jnp.dot(attn_weights, v)
    return jnp.dot(output, Wo)

def feed_forward(x, W1, b1, W2, b2):
    return jnp.dot(gelu(jnp.dot(x, W1) + b1), W2) + b2

def transformer_block(x, params, mask=None):
    # Unpack params
    Wq, Wk, Wv, Wo = params['attn']
    gamma1, beta1 = params['ln1']
    W1, b1, W2, b2 = params['ff']
    gamma2, beta2 = params['ln2']

    # Self-attention
    attn_out = multi_head_attention(x, x, x, Wq, Wk, Wv, Wo, mask)
    x = layer_norm(x + attn_out, gamma1, beta1)

    # Feed-forward
    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_out, gamma2, beta2)
    return x

def model_forward(params, x, mask=None):
    x = jnp.dot(x, params['embedding'])  # Token embedding
    for block_params in params['blocks']:
        x = transformer_block(x, block_params, mask)
    # Global average pooling + classification
    x = jnp.mean(x, axis=0)  # (d_model,)
    logits = jnp.dot(x, params['clf_w']) + params['clf_b']
    return logits

# ----------------------------
# 2. Loss & Accuracy
# ----------------------------

def loss_fn(params, x, y, mask):
    logits = model_forward(params, x, mask)
    # Binary cross-entropy
    probs = jax.nn.sigmoid(logits)
    loss = -y * jnp.log(probs + 1e-8) - (1 - y) * jnp.log(1 - probs + 1e-8)
    return loss

@jit
def update(params, x, y, mask, lr=1e-3):
    grads = jax.grad(lambda p: loss_fn(p, x, y, mask))(params)
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

@jit
def predict(params, x, mask):
    logits = model_forward(params, x, mask)
    return jax.nn.sigmoid(logits) > 0.5

# ----------------------------
# 3. Parameter Initialization
# ----------------------------

def init_params(key, vocab_size, d_model, d_ff, n_blocks):
    k1, k2, k3 = random.split(key, 3)
    params = {
        'embedding': random.normal(k1, (vocab_size, d_model)) * 0.02,
        'blocks': [],
        'clf_w': random.normal(k2, (d_model,)) * 0.02,
        'clf_b': jnp.zeros(()),
    }
    for _ in range(n_blocks):
        kq, kk, kv, ko, k1, k2 = random.split(k3, 6)
        block = {
            'attn': [
                random.normal(kq, (d_model, d_model)) * 0.02,  # Wq
                random.normal(kk, (d_model, d_model)) * 0.02,  # Wk
                random.normal(kv, (d_model, d_model)) * 0.02,  # Wv
                random.normal(ko, (d_model, d_model)) * 0.02,  # Wo
            ],
            'ln1': [jnp.ones((d_model,)), jnp.zeros((d_model,))],
            'ff': [
                random.normal(k1, (d_model, d_ff)) * 0.02,  # W1
                jnp.zeros((d_ff,)),                           # b1
                random.normal(k2, (d_ff, d_model)) * 0.02,  # W2
                jnp.zeros((d_model,)),                        # b2
            ],
            'ln2': [jnp.ones((d_model,)), jnp.zeros((d_model,))],
        }
        params['blocks'].append(block)
    return params

# ----------------------------
# 4. Data Loading & Preprocessing
# ----------------------------

def tokenize_and_pad(texts, vocab, max_len=128):
    tokens = []
    for text in texts:
        ids = [vocab.get(w, 0) for w in text.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        tokens.append(ids)
    return jnp.array(tokens)

# Load IMDB dataset
dataset = load_dataset("imdb")
train_texts = dataset["train"]["text"][:1000]  # Small subset for demo
train_labels = jnp.array(dataset["train"]["label"][:1000])
test_texts = dataset["test"]["text"][:200]
test_labels = jnp.array(dataset["test"]["label"][:200])

# Build tiny vocabulary
vocab = {"<PAD>": 0}
for text in train_texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

# Tokenize
max_len = 64
X_train = tokenize_and_pad(train_texts, vocab, max_len)
X_test = tokenize_and_pad(test_texts, vocab, max_len)

# Create attention masks (1 for real tokens, 0 for padding)
train_mask = X_train != 0
test_mask = X_test != 0

# One-hot not needed; we use raw token IDs as indices for embedding lookup
# But for simplicity, we'll use raw IDs and embed via matrix multiplication

# ----------------------------
# 5. Training Loop
# ----------------------------

# Hyperparameters
vocab_size = len(vocab)
d_model = 32
d_ff = 64
n_blocks = 2
batch_size = 32
epochs = 10

# Initialize
key = random.PRNGKey(0)
params = init_params(key, vocab_size, d_model, d_ff, n_blocks)

# Embedding lookup helper
def embed_lookup(token_ids, embedding_matrix):
    return embedding_matrix[token_ids]

# Training
for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        y_batch = train_labels[i:i+batch_size]
        mask_batch = train_mask[i:i+batch_size]

        # Embed tokens
        x_emb = embed_lookup(x_batch, params['embedding'])  # (B, T, d_model)

        # Update params for each sample (simple loop; could vmap)
        for j in range(len(x_batch)):
            params = update(params, x_emb[j], y_batch[j], mask_batch[j])

        # Compute loss for monitoring
        losses = vmap(loss_fn, in_axes=(None, 0, 0, 0))(params, x_emb, y_batch, mask_batch)
        total_loss += jnp.sum(losses)

    # Evaluate
    test_emb = embed_lookup(X_test, params['embedding'])
    preds = vmap(predict, in_axes=(None, 0, 0))(params, test_emb, test_mask)
    acc = jnp.mean(preds == test_labels)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}, Test Acc: {acc:.2%}")