from turtle import position
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

ACT2FN = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
}


class TransposeLinear(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class TfmrAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            # define the bias term for constructing the causal mask (i.e., seeing only prefix tokens).
            # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L128
            torch.tril(torch.ones(max_positions, max_positions, dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.c_attn = TransposeLinear(3 * self.embed_dim, self.embed_dim)
        self.c_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value):
        # query, key: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        attn_weights = query @ key.transpose(-2, -1)  # (..., query_len, key_len)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        query_seq_len, key_seq_len = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            ..., key_seq_len - query_seq_len : key_seq_len, :key_seq_len
        ]
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ value

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        # Input Size: (batch_size, sequence_length, hidden_size)
        # Output Size: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        bs, seq_len, _ = tensor.size()
        tensor = tensor.view(bs, seq_len, num_heads, attn_head_size)
        return tensor.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        # Input Size: (batch_size, num_attn_heads, sequence_length, attn_head_size)
        # Output Size: (batch_size, sequence_length, hidden_size)
        bs, _, seq_len, _ = tensor.size()
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(bs, seq_len, -1)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TfmrMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = TransposeLinear(intermediate_size, embed_dim)
        self.c_proj = TransposeLinear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TfmrBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TfmrAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TfmrMLP(inner_dim, config)
        self.layer_norm = "PreNorm"

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        if self.layer_norm == "PreNorm":
            return self._forward_pre_norm(hidden_states, layer_past, use_cache)
        elif self.layer_norm == "PostNorm":
            return self._forward_post_norm(hidden_states, layer_past, use_cache)
        else:
            raise ValueError(
                f"Unrecognized layer normalization type: {self.layer_norm}"
            )

    def _forward_pre_norm(self, hidden_states, layer_past, use_cache):
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        hidden_states = attn_output + residual
        residual = hidden_states

        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        return self._finalize_outputs(hidden_states, outputs, use_cache)

    def _forward_post_norm(self, hidden_states, layer_past, use_cache):
        residual = hidden_states

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        hidden_states = residual + attn_output
        hidden_states = self.ln_1(hidden_states)
        residual = hidden_states

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states
        hidden_states = self.ln_2(hidden_states)

        return self._finalize_outputs(hidden_states, outputs, use_cache)

    def _finalize_outputs(self, hidden_states, outputs, use_cache):
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class TfmrModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [TfmrBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        inputs_embeds = self.wte(input_ids)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        position_ids = torch.arange(
            past_length, input_shape[-1] + past_length, device=device
        ).view(-1, input_shape[-1])
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1],
            )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions,
        }


class TfmrLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TfmrModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        labels=None,
        use_cache=None,
        PAD_ID=None,
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            ce_loss_fct = CrossEntropyLoss(reduction="none")
            # We set the loss to 0 where [PAD] token is the label, except for the last token,
            # where [PAD] token worked as the "eod of sentence" token.
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            per_token_loss = ce_loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            per_token_loss = per_token_loss.view_as(shift_labels)
            # y1_hat   y2_hat   ...   yn_hat   [PAD]   [PAD]   [PAD]
            # ------------------------------------------------------
            # |                Transformer                         |
            # ------------------------------------------------------
            #  [PAD]     y1      y2     ...      yn    [PAD]   [PAD]
            #     1       1       1     ...       1      0       0  <- mask
            #     1      (         shift_labels != PAD_ID         ) <- mask
            mask = (shift_labels != PAD_ID).roll(shifts=1, dims=1).float()
            mask[:, 0] = 1.0
            per_token_loss *= mask

            seq_loss = per_token_loss.sum(dim=1) / mask.sum(dim=1)
            loss = seq_loss.mean()

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "cross_attentions": transformer_outputs["cross_attentions"],
        }

    def inference(
        self,
        device,
        PAD_ID,
        batch_size,
        maxlen,
        decode_strategy,
        temperature,
        top_p=1.0,
    ):
        self.eval()
        allgen = []
        with torch.no_grad():
            for i in range(0, int(5000 / batch_size) + 1):
                input_ids = torch.tensor([[PAD_ID] for _ in range(batch_size)]).to(
                    device
                )
                past_key_values = None
                output_ids = input_ids
                for _ in range(maxlen):
                    outputs = self(
                        input_ids, past_key_values=past_key_values, use_cache=True
                    )
                    logits = outputs["logits"]
                    past_key_values = outputs["past_key_values"]
                    logits = logits[:, -1, :] / temperature

                    if decode_strategy == "top-p":
                        # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L469
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=False
                        )
                        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits = logits.masked_fill(indices_to_remove, float("-inf"))
                    prob = logits.softmax(dim=-1)  # shape: (batch_size, num_vocabs)
                    now_token = torch.multinomial(prob, 1)[:, :1]  # shape: (batch_size)

                    output_ids = torch.cat([output_ids, now_token], 1)
                    input_ids = now_token
                allgen += output_ids.cpu().numpy().tolist()
        pro_allgen = []
        for gen in allgen[:5000]:
            pro_allgen.append([])
            for idx in gen[1:]:
                if idx == PAD_ID:
                    break
                pro_allgen[-1].append(idx)
        self.train()  # return to training mode
        return pro_allgen
