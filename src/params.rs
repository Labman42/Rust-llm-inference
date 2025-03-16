use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();
            let shape = tensor.shape().to_vec();

            let data: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::F32 => data.chunks(4).
                map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                _ => panic!("Unsupported data type"),
            };

            Tensor::new(data, &shape)
        };
        
        let layers = config.num_hidden_layers;

        let mut rms_att_w = Vec::with_capacity(layers);
        let mut wq = Vec::with_capacity(layers);
        let mut wk = Vec::with_capacity(layers);
        let mut wv = Vec::with_capacity(layers);
        let mut wo = Vec::with_capacity(layers);
        let mut rms_ffn_w = Vec::with_capacity(layers);
        let mut w_up = Vec::with_capacity(layers);
        let mut w_gate = Vec::with_capacity(layers);
        let mut w_down = Vec::with_capacity(layers);


        for layer in 0..layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", layer)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", layer)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", layer)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", layer)));
        }


        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight")
        }

    }
}
