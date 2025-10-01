import torch
from torch import nn

from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer 
from datasets import load_dataset 
import tqdm 

from functools import partial 

# MODEL_NAME = 'facebook/opt-125m'
# MODEL_NAME = 'facebook/opt-350m'
# MODEL_NAME = 'facebook/opt-1.3b'
MODEL_NAME = 'facebook/opt-2.7b'
# MODEL_NAME = 'facebook/opt-6.7b'

W_BITS = 2
A_BITS = 8

W_QUANT = 'custom' # [per_channel, per_tensor, custom]
A_QUANT = 'per_tensor' # [per_token, per_tensor] 



################################ <MSB_LSB_Quant_Functions> ################################ 

def split_msb_lsb_quantize(w, 
                           n_bits_total=16, #16
                           n_msb=2, 
                           n_bits_lsb_quant=8):
    """
    w: torch.Tensor of floats
    n_bits_total: total bits you first quantize into (default 16)
    n_msb: number of MSB bits to keep aside
    n_bits_lsb_quant: how many bits you quantize the remaining LSB part into (e.g., 8 or 6)
    """
    assert n_bits_total > n_msb, "MSB bits must be < total bits"
    n_lsb = n_bits_total - n_msb

    # Step 1: normal symmetric quantization to n_bits_total
    qmax_total = 2 ** (n_bits_total - 1) - 1
    scale_total = w.abs().max().clamp_min(1e-5) / qmax_total

    q_int16 = torch.round(w / scale_total).to(torch.int16)  # quantized int values

    # Step 2: split into MSB and LSB parts
    lsb_mask = (1 << n_lsb) - 1
    lsb_part = (q_int16 & lsb_mask).to(torch.int16)
    msb_part = (q_int16 >> n_lsb).to(torch.int16)  # top bits

    # Step 3: re-quantize the LSB part into n_bits_lsb_quant
    qmax_lsb = 2 ** n_bits_lsb_quant - 1
    # Compute scale for lsb part separately
    scale_lsb = lsb_part.abs().max().clamp_min(1e-5) / qmax_lsb
    q_lsb_requant = torch.round(lsb_part / scale_lsb).to(torch.int16)

    # Step 4: dequantize lsb back to its original range
    lsb_dequant = (q_lsb_requant * scale_lsb).to(torch.int16)

    # Step 5: merge back msb + dequantized lsb
    merged_int = (msb_part << n_lsb) | lsb_dequant

    # Step 6: dequantize back to float16 using original scale
    w_fp16 = (merged_int.to(torch.float16) * scale_total).to(torch.float16)

    return w_fp16#, (scale_total, scale_lsb)


################################ <Quant_Functions> ################################ 

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=W_BITS):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=W_BITS):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=A_BITS):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=A_BITS):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t
################################ </Quant_Functions> ################################ 

################################ <Quant_Class> ################################ 
class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=A_BITS)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=A_BITS)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=W_BITS
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=W_BITS
            )
        elif weight_quant == "custom":
            new_module.weight = split_msb_lsb_quantize(
                module.weight, n_bits_lsb_quant=W_BITS
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"

################################ </Quant_Class> ################################ 

################################ <OPT_Quant> ################################ 
def quantize_opt(
    model, weight_quant=W_QUANT, act_quant=A_QUANT, quantize_bmm_input=True
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model

################################ </OPT_Quant> ################################ 

################################ <Evaluator_Last_Token> ################################ 

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            # print(f"input_ids.shape: {input_ids.shape}")
            outputs = model(input_ids)
            # print(f"output.shape: {outputs.shape}")
            # print(f"output.logits.shape: {outputs.logits.shape}")
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
################################ </Evaluator_Last_Token> ################################ 

################################ <Evaluator_Perplexity> ################################ 

class EvaluatorPerplexity:
    def __init__(self, dataset, tokenizer, device, n_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

################################ </Evaluator_Perplexity> ################################ 


print(f"initializing tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

print(f"initializing dataset")
# dataset = load_dataset("lambada", split="validation[:100]")
dataset = load_dataset("cimec/lambada", split="test")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test") 
dataset = dataset.filter(lambda example: example['text'].strip() != '')

# count = 0 
# for i in range(len(dataset)):
    
#     # print(f"token {i}: {dataset['text'][i]}")
#     # print(f"len(token({i})) : {len(dataset['text'][i])}")
#     if len(dataset['text'][i]) == 0: 
#         count += 1
# print(f"len(dataset): {len(dataset)}") 
# print(f"count: {count}") 

evaluator_last_token = Evaluator(dataset, tokenizer, "cuda")
evaluator_perplexity = EvaluatorPerplexity(dataset, tokenizer, "cuda")

print("loading model")
model_fp16 = OPTForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto") 

# print("running eval of FP16 model")
# fp16_acc = evaluator_last_token.evaluate(model_fp16) 
# fp16_prep = evaluator_perplexity.evaluate(model_fp16) 


print("quantizing model")
model_w8a8 = quantize_opt(model_fp16) 
print("running eval of quantized model")
w8a8_acc = evaluator_last_token.evaluate(model_w8a8) 
w8a8_prep = evaluator_perplexity.evaluate(model_w8a8) 

# print(f"original model (fp16) accuracy: {fp16_acc}")
# print(f"original model (fp16) perplexity: {fp16_prep}")

print(f"quantized model (fp16) accuracy: {w8a8_acc}")
print(f"quantized model (fp16) perplexity: {w8a8_prep}")



## notebooks ##
# per channel, tensor quantization for weights
# per token, tensor quantization for activations
# hooks
# W8A8 linear class