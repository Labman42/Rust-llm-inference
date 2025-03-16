# Rust-llm-inference

A high-performance inference engine built in Rust for running Llama series models. This project leverages Rust's speed and safety to provide efficient text generation while offering a modular design that supports a range of neural network operators.

## Overview

The inference system implements the core functionalities needed to run Llama models, including:

- **Grouped Query Attention (GQA)**
- **RMSNorm (Root Mean Square Normalization)**
- **Masked Softmax**
- **SiLU (Swiglu variant) Activation**
- **RoPE (Rotary Positional Encoding)**

Future enhancements will add support for operators such as GELU, Layer Norm, Multi-Query Attention (MQA), Multi-Linear Attention (MLA), and more.

## Features

- **Efficient Inference Pipeline:**  
  Utilizes a key-value cache for attention states and reuses pre-allocated buffers to optimize runtime performance.

- **Core Neural Network Operators:**  
  Implements essential operations for transformer models:
  - **GQA:** Efficient handling of grouped attention heads.
  - **RMSNorm:** Stabilizes activations using RMS normalization.
  - **Masked Softmax:** Customized softmax for attention with masking support.
  - **SiLU (Swiglu):** Applies a variant of the SiLU activation function.
  - **RoPE:** Implements rotary positional encoding to capture token order.

- **Extensible Architecture:**  
  Designed to easily incorporate additional operators and features as the project evolves.


## Getting Started

### Prerequisites

- **Rust and Cargo:**  
  Install Rust using the instructions from [rust-lang.org](https://www.rust-lang.org/tools/install).

- **Dependencies:**  
  The project uses several external crates:
  - [SafeTensors](https://github.com/huggingface/safetensors-rust) for loading model weights.
  - [Tokenizers](https://github.com/huggingface/tokenizers) for text tokenization.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Build the project in release mode:**
    ```bash
    cargo build --release
    ```

## Future Enhancements

  -	Additional Operator Support:
    - Integrate operators such as GELU, Layer Norm, MQA, MLA, and others.
  -	Performance Optimizations:
    - Improve speed and memory efficiency for handling larger models and longer sequences.
  -	Extended Model Compatibility:
    - Adapt the system for other transformer-based models and configurations.