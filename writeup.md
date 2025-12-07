### Problem (unicode1): Understanding Unicode (1 point)

**(a)** What Unicode character does `chr(0)` return?
**Deliverable:** A one-sentence response.

```python
In [1]: chr(0)
Out[1]: '\x00'
```

**(b)** How does this character’s string representation (`__repr__()`) differ from its printed representation?
**Deliverable:** A one-sentence response.

```python
In [2]: chr(0).__repr__()  # shows the escaped form '\x00'
Out[2]: "'\\x00'"

In [3]: print(chr(0))  # no visible glyph

```

**(c)** What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**Deliverable:** A one-sentence response.

```python
In [7]: "this is a test" + chr(0) + "string"
Out[7]: 'this is a test\x00string'

In [8]: print("this is a test" + chr(0) + "string")
this is a teststring
```

---

### Problem (unicode2): Unicode Encodings (3 points)

**(a)** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
**Deliverable:** A one-to-two sentence response.

> **UTF-8**
>
> - 以 **1～4 字节** 表示一个 Unicode 码点（变长编码）。
> - ASCII 字符（U+0000 ~ U+007F，比如英文、数字、常见符号）只占 **1 字节**。
>
> **UTF-16**
>
> - 以 **2 或 4 字节** 表示一个码点（变长编码）。
> - 常用字符（BMP 区：U+0000 ~ U+FFFF）占 **2 字节**，超出 BMP 的字符用 **4 字节（代理对）**。
>
> **UTF-32**
>
> - 固定 **4 字节** 表示一个码点（定长编码）。
> - 所有字符一律 4 字节，没有变长。

Space-efficient, ASCII-compatible.

**(b)** Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**Deliverable:** An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.

```python
In [11]: decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
Cell In[11], line 1
----> 1 decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))

Cell In[9], line 2, in decode_utf8_bytes_to_str_wrong(bytestring)
      1 def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
----> 2     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

This function cannot decode the multi-bytes characters.

**(c)** Give a two byte sequence that does not decode to any Unicode character(s).
**Deliverable:** An example, with a one-sentence explanation.

```python
In [21]: bytes([228, 189, 160]).decode()
Out[21]: '你'

In [22]: bytes([228, 189]).decode()
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
Cell In[22], line 1
----> 1 bytes([228, 189]).decode()

UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
```



---

### Problem (train_bpe): BPE Tokenizer Training (15 points)

**Deliverable:** Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:

*   `input_path`: `str` Path to a text file with BPE tokenizer training data.
*   `vocab_size`: `int` A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
*   `special_tokens`: `list[str]` A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.

Your BPE training function should return the resulting vocabulary and merges:
*   `vocab`: `dict[int, bytes]` The tokenizer vocabulary, a mapping from `int` (token ID in the vocabulary) to `bytes` (token bytes).
*   `merges`: `list[tuple[bytes, bytes]]` A list of BPE merges produced from training. Each list item is a tuple of bytes (`<token1>`, `<token2>`), representing that `<token1>` was merged with `<token2>`. The merges should be ordered by order of creation.

---

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

**(a)** Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?
**Resource requirements:** ≤ 30 minutes (no GPUs), ≤ 30GB RAM
**Deliverable:** A one-to-two sentence response.

**(b)** Profile your code. What part of the tokenizer training process takes the most time?
**Deliverable:** A one-to-two sentence response.

---

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a)** Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
**Resource requirements:** ≤ 12 hours (no GPUs), ≤ 100GB RAM
**Deliverable:** A one-to-two sentence response.

**(b)** Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
**Deliverable:** A one-to-two sentence response.

---

### Problem (tokenizer): Implementing the tokenizer (15 points)

**Deliverable:** Implement a `Tokenizer` class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface:

```python
def __init__(self, vocab, merges, special_tokens=None)
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
def encode(self, text: str) -> list[int]
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
def decode(self, ids: list[int]) -> str
```

---

### Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

**(a)** Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
**Deliverable:** A one-to-two sentence response.

**(b)** What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
**Deliverable:** A one-to-two sentence response.

**(c)** Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
**Deliverable:** A one-to-two sentence response.

**(d)** Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype `uint16`. Why is `uint16` an appropriate choice?
**Deliverable:** A one-to-two sentence response.

---

### Problem (linear): Implementing the linear module (1 point)

**Deliverable:** Implement a `Linear` class that inherits from `torch.nn.Module` and performs a linear transformation. Your implementation should follow the interface of PyTorch’s built-in `nn.Linear` module, except for not having a `bias` argument or parameter.

---

### Problem (embedding): Implement the embedding module (1 point)

**Deliverable:** Implement the `Embedding` class that inherits from `torch.nn.Module` and performs an embedding lookup. Your implementation should follow the interface of PyTorch’s built-in `nn.Embedding` module.

---

### Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

**Deliverable:** Implement `RMSNorm` as a `torch.nn.Module`.
**Note:** Remember to upcast your input to `torch.float32` before performing the normalization (and later downcast to the original dtype).

---

### Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)

**Deliverable:** Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.
**Note:** in this particular case, you should feel free to use `torch.sigmoid` in your implementation for numerical stability.

---

### Problem (rope): Implement RoPE (2 points)

**Deliverable:** Implement a class `RotaryPositionalEmbedding` that applies RoPE to the input tensor.

---

### Problem (softmax): Implement softmax (1 point)

**Deliverable:** Write a function to apply the softmax operation on a tensor. Your function should take two parameters: a tensor and a dimension `i`, and apply softmax to the `i`-th dimension of the input tensor. Use the trick of subtracting the maximum value for numerical stability.

---

### Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)

**Deliverable:** Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape `(batch_size, ..., seq_len, d_k)` and values of shape `(batch_size, ..., seq_len, d_v)`.
Your implementation should also support an optional user-provided boolean mask of shape `(seq_len, seq_len)`.

---

### Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)

**Deliverable:** Implement causal multi-head self-attention as a `torch.nn.Module`. Your implementation should accept `d_model` and `num_heads`.
Applying RoPE: RoPE should be applied to the query and key vectors, but not the value vectors.
Causal masking: Your implementation should prevent the model from attending to future tokens.

---

### Problem (transformer_block): Implement the Transformer block (3 points)

**Deliverable:** Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your Transformer block code should pass the provided tests.

---

### Problem (transformer_lm): Implementing the Transformer LM (3 points)

**Deliverable:** Implement the Transformer language model as described in §3.1 and illustrated in Figure 1. Your implementation should accept all construction parameters for the Transformer block, as well as `vocab_size`, `context_length`, and `num_layers`.

---

### Problem (transformer_accounting): Transformer LM resource accounting (5 points)

**(a)** Consider GPT-2 XL, which has the following configuration:
*   `vocab_size`: 50,257
*   `context_length`: 1,024
*   `num_layers`: 48
*   `d_model`: 1,600
*   `num_heads`: 25
*   `d_ff`: 6,400

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
**Deliverable:** A one-to-two sentence response.

**(b)** Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.
**Deliverable:** A list of matrix multiplies (with descriptions), and the total number of FLOPs required.

**(c)** Based on your analysis above, which parts of the model require the most FLOPs?
**Deliverable:** A one-to-two sentence response.

**(d)** Repeat your analysis with GPT-2 small (12 layers, 768 `d_model`, 12 heads), GPT-2 medium (24 layers, 1024 `d_model`, 16 heads), and GPT-2 large (36 layers, 1280 `d_model`, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
**Deliverable:** For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

**(e)** Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?
**Deliverable:** A one-to-two sentence response.

---

### Problem (cross_entropy): Implement Cross entropy

**Deliverable:** Write a function to compute the cross entropy loss, which takes in predicted logits (`o_i`) and targets (`x_{i+1}`) and computes the cross entropy `l_i = -log softmax(o_i)[x_{i+1}]`. Your function should handle numerical stability, cancel out log and exp where possible, and return the average across the batch.

---

### Problem (learning_rate_tuning): Tuning the learning rate (1 point)

Run the SGD example (provided in text) with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge?
**Deliverable:** A one-two sentence response with the behaviors you observed.

---

### Problem (adamw): Implement AdamW (2 points)

**Deliverable:** Implement the AdamW optimizer as a subclass of `torch.optim.Optimizer`. Your class should take the learning rate `alpha` in `__init__`, as well as the `beta`, `epsilon` and `lambda` hyperparameters.

---

### Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)

Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

**(a)** How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the `batch_size` and the model hyperparameters. Assume `d_ff = 4 * d_model`.
For simplicity, when calculating memory usage of activations, consider only the following components:
*   Multi-head self-attention sublayer: QKV projections, Q^TK matrix multiply, softmax, weighted sum of values, output projection.
*   Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
*   final RMSNorm
*   output embedding
*   cross-entropy on logits
**Deliverable:** An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

**(b)** Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the `batch_size`. What is the maximum batch size you can use and still fit within 80GB memory?
**Deliverable:** An expression that looks like `a * batch_size + b` for numerical values `a`, `b`, and a number representing the maximum batch size.

**(c)** How many FLOPs does running one step of AdamW take?
**Deliverable:** An algebraic expression, with a brief justification.

**(d)** Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput. An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single A100? Assume that the backward pass has twice the FLOPs of the forward pass.
**Deliverable:** The number of days training would take, with a brief justification.

---

### Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup

**Deliverable:** Write a function that takes `t`, `alpha_max`, `alpha_min`, `T_w` and `T_c`, and returns the learning rate `alpha_t` according to the scheduler defined in the text.

---

### Problem (gradient_clipping): Implement gradient clipping (1 point)

**Deliverable:** Write a function that implements gradient clipping. Your function should take a list of parameters and a maximum l2-norm. It should modify each parameter gradient in place. Use `epsilon = 10^-6`.

---

### Problem (data_loading): Implement data loading (2 points)

**Deliverable:** Write a function that takes a numpy array `x` (integer array with token IDs), a `batch_size`, a `context_length` and a PyTorch device string, and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets.

---

### Problem (checkpointing): Implement model checkpointing (1 point)

**Deliverable:** Implement `save_checkpoint` and `load_checkpoint` functions to load and save checkpoints (model state, optimizer state, iteration number).

---

### Problem (training_together): Put it together (4 points)

**Deliverable:** Write a script that runs a training loop to train your model on user-provided input. In particular, we recommend that your training script allow for (at least) the following:
*   Ability to configure and control the various model and optimizer hyperparameters.
*   Memory-efficient loading of training and validation large datasets with `np.memmap`.
*   Serializing checkpoints to a user-provided path.
*   Periodically logging training and validation performance.

---

### Problem (decoding): Decoding (3 points)

**Deliverable:** Implement a function to decode from your language model. We recommend that you support the following features:
*   Generate completions for a user-provided prompt.
*   Allow the user to control the maximum number of generated tokens.
*   Given a desired temperature value, apply softmax temperature scaling.
*   Top-p sampling (nucleus sampling), given a user-specified threshold value.

---

### Problem (experiment_log): Experiment logging (3 points)

For your training and evaluation code, create experiment tracking infrastructure that allows you to track your experiments and loss curves with respect to gradient steps and wallclock time.
**Deliverable:** Logging infrastructure code for your experiments and an experiment log (a document of all the things you tried) for the assignment problems below in this section.

---

### Problem (learning_rate): Tune the learning rate (3 points)

The learning rate is one of the most important hyperparameters to tune. Taking the base model you’ve trained (TinyStories), answer the following questions:

**(a)** Perform a hyperparameter sweep over the learning rates and report the final losses (or note divergence if the optimizer diverges).
**Deliverable:** Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.
**Deliverable:** A model with validation loss (per-token) on TinyStories of at most 1.45

**(b)** Folk wisdom is that the best learning rate is “at the edge of stability.” Investigate how the point at which learning rates diverge is related to your best learning rate.
**Deliverable:** Learning curves of increasing learning rate which include at least one divergent run and an analysis of how this relates to convergence rates.

---

### Problem (batch_size_experiment): Batch size variations (1 point)

Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.
**Deliverable:** Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.
**Deliverable:** A few sentences discussing of your findings on batch sizes and their impacts on training.

---

### Problem (generate): Generate text (1 point)

Using your decoder and your trained checkpoint, report the text generated by your model. You may need to manipulate decoder parameters (temperature, top-p, etc.) to get fluent outputs.
**Deliverable:** Text dump of at least 256 tokens of text (or until the first `<|endoftext|>` token), and a brief comment on the fluency of this output and at least two factors which affect how good or bad this output is.

---

### Problem (layer_norm_ablation): Remove RMSNorm and train (1 point)

Remove all of the RMSNorms from your Transformer and train. What happens at the previous optimal learning rate? Can you get stability by using a lower learning rate?
**Deliverable:** A learning curve for when you remove RMSNorms and train, as well as a learning curve for the best learning rate.
**Deliverable:** A few sentence commentary on the impact of RMSNorm.

---

### Problem (pre_norm_ablation): Implement post-norm and train (1 point)

Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.
**Deliverable:** A learning curve for a post-norm transformer, compared to the pre-norm one.

---

### Problem (no_pos_emb): Implement NoPE (1 point)

Modify your Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.
**Deliverable:** A learning curve comparing the performance of RoPE and NoPE.

---

### Problem (swiglu_ablation): SwiGLU vs. SiLU (1 point)

**Deliverable:** A learning curve comparing the performance of SwiGLU and SiLU feed-forward networks, with approximately matched parameter counts.

---

### Problem (main_experiment): Experiment on OWT (2 points)

Train your language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does this model do?
**Deliverable:** A learning curve of your language model on OpenWebText. Describe the difference in losses from TinyStories – how should we interpret these losses?
**Deliverable:** Generated text from OpenWebText LM, in the same format as the TinyStories outputs. How is the fluency of this text? Why is the output quality worse even though we have the same model and compute budget as TinyStories?

---

### Problem (leaderboard): Leaderboard (6 points)

You will train a model under the leaderboard rules with the goal of minimizing the validation loss of your language model within 1.5 H100-hour.
**Deliverable:** The final validation loss that was recorded, an associated learning curve that clearly shows a wallclock-time x-axis that is less than 1.5 hours and a description of what you did. We expect a leaderboard submission to beat at least the naive baseline of a 5.0 loss.