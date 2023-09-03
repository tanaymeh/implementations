import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def scaled_dot_product_attn(q, k, v, mask=None):
    """
    Computes Scaled Dot Product attention for a single head
    """
    d_model = q.shape[-1]
    attention_weights = softmax(np.dot(q, k.T) / np.sqrt(d_model))
    if mask is not None:
        attention_weights += mask * -1e9
    return attention_weights, np.dot(attention_weights, v)


def parallel_multihead_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_heads=8, mask=None
):
    """
    Extend the scaled dot production attention to Parrallelized Multi-headed attention
    """
    d_model = Q.shape[-1]
    seq_len = Q.shape[1]
    d_head = d_model // num_heads
    attention_weights = []
    outputs = []
    # Reshape the Q, K, V matrices
    # This operation basically divides the Q, K, V into 'num_heads' number of heads with
    # 'd_head' dimensionality each
    # Normal Multi-headed attention has the dimensions to be: [batch_size, num_heads, seq_len, d_head]
    # In the below op, the dim of Q, K, V will be [batch_size*seq_len, num_heads, d_head]
    # This is the parallelized version of multi-headed attention
    Q = np.reshape(Q, (-1, num_heads, seq_len, d_head))
    K = np.reshape(K, (-1, num_heads, seq_len, d_head))
    V = np.reshape(V, (-1, num_heads, seq_len, d_head))

    # Do scaled dot product attention for each head
    for i in range(num_heads):
        # Get the values for each head for Q, K, V
        head_Q = Q[:, i]
        head_K = K[:, i]
        head_V = V[:, i]

        # Perform scaled dot product attention
        head_attn_weights, head_outputs = scaled_dot_product_attn(
            head_Q, head_K, head_V, mask=mask
        )

        attention_weights.append(head_attn_weights)
        outputs.append(head_outputs)

    # Stack attention weights and concatenate output for all heads
    attention_weight = np.stack(attention_weights, axis=1)
    output = np.concatenate(outputs, axis=1)

    # Reshape the output into our desired form
    output = output.reshape(-1, num_heads * d_head)

    return output, attention_weight
