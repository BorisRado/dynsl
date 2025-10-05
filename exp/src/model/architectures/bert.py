import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from slbd.torch.sequential_model import FlexibleSequential
from slbd.torch.model_utils import extract_submodel

from exp.src.common.stochasticity import set_seed


def get_extended_attention_mask(attention_mask, dtype=torch.float32):
    extended_attention_mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - extended_attention_mask) * torch.finfo(dtype).min


def reverse_extended_attention_mask(extended_attention_mask, dtype=torch.long):
    min_value = torch.finfo(extended_attention_mask.dtype).min
    regular_mask = (extended_attention_mask != min_value).to(dtype)

    # Remove the extra dimensions
    return regular_mask.squeeze(1).squeeze(1)


def compress_embeddings(embeddings, attention_mask):
    return [embeddings[idx][:attention_mask[idx].sum()].detach().cpu() for idx in range(embeddings.shape[0])]


def compress_embeddings_api(tpl):
    assert len(tpl) == 2
    return compress_embeddings(tpl[0], reverse_extended_attention_mask(tpl[1]))


def uncompress_embeddings(compressed_embeddings, *, max_len, device=torch.device("cpu")):
    embedding_dim = compressed_embeddings[0].shape[-1]
    reconstructed_embeddings = torch.zeros(len(compressed_embeddings), max_len, embedding_dim, dtype=torch.float32, device=device)
    reconstructed_attention_mask = torch.zeros(len(compressed_embeddings), max_len, dtype=torch.long, device=device)

    for i, comp_emb in enumerate(compressed_embeddings):
        seq_len = comp_emb.shape[0]
        reconstructed_embeddings[i, :seq_len] = comp_emb
        reconstructed_attention_mask[i, :seq_len] = 1

    # Convert reconstructed attention mask to extended format for the classifier
    extended_attention_mask = get_extended_attention_mask(reconstructed_attention_mask)
    return reconstructed_embeddings, extended_attention_mask


def compress_gradient(gradient, received_data):
    return [
        gradient[idx][:len(received_data["embeddings"][idx])] for idx in range(gradient.shape[0])
    ]

def uncompress_gradient_api(compressed_gradient, client_output, max_len):
    return uncompress_embeddings(compressed_gradient, max_len=max_len)[0]


class CustomEmbedding(nn.Module):
    def __init__(self, bert_embedding):
        super().__init__()
        self.bert_embedding = bert_embedding

    def forward(self, input_ids, attention_mask):
        embeddings = self.bert_embedding(input_ids=input_ids)
        extended_attention_mask = get_extended_attention_mask(attention_mask)
        return embeddings, extended_attention_mask


class CustomTransformerLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, embeddings, extended_attention_mask):
        return self.layer(embeddings, extended_attention_mask)[0], extended_attention_mask


class CustomPooler(nn.Module):
    def __init__(self, pooler):
        super().__init__()
        self.pooler = pooler

    def forward(self, embeddings, extended_attention_mask):
        _ = (extended_attention_mask, )
        return self.pooler(embeddings),


class CustomDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        return self.dropout(x), # return tuple instead of plain tensor


class SequentialBert(FlexibleSequential):
    def __init__(self, **kwargs):
        custom_config = BertConfig(**kwargs)
        custom_bert_model = BertForSequenceClassification(custom_config)
        super().__init__(
            CustomEmbedding(custom_bert_model.bert.embeddings),
            *[CustomTransformerLayer(l) for l in custom_bert_model.bert.encoder.layer],
            CustomPooler(custom_bert_model.bert.pooler),
            CustomDropout(custom_bert_model.dropout),
            custom_bert_model.classifier
        )


def _test():
    print("Running tests...")
    seed = 10
    num_labels = 22
    config_kwargs = {
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 512,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
        "num_labels": num_labels,
    }
    set_seed(seed)
    custom_config = BertConfig(**config_kwargs)
    custom_bert_model = BertForSequenceClassification(custom_config)
    print(sum(p.numel() for p in custom_bert_model.parameters()))
    set_seed(seed)
    seq_model = SequentialBert(**config_kwargs)

    def get_random_batch(bs, sl):
        input_ids = torch.randint(0, config_kwargs["vocab_size"], (bs, sl))

        # Create attention masks with random lengths of 1s followed by trailing 0s
        attention_mask = torch.zeros((bs, sl), dtype=torch.long)
        for i in range(bs):
            seq_len = torch.randint(1, sl + 1, (1,)).item()  # Random length between 1 and seq_length
            attention_mask[i, :seq_len] = 1
        return input_ids, attention_mask

    for _ in range(10):
        _, mask = get_random_batch(8, 128)
        assert (mask == reverse_extended_attention_mask(get_extended_attention_mask(mask))).all()
    print("Mask are ok")

    # Generate random batch of input_ids and attention_mask
    for idx in range(10):
        batch_size = 8
        seq_length = 128

        # Random input ids (integer values between 0 and vocab_size-1)

        # Forward pass through the models to compare outputs
        with torch.no_grad():
            # Original model
            input_ids, attention_mask = get_random_batch(batch_size, seq_length)

            set_seed(idx)
            outputs1 = custom_bert_model(input_ids=input_ids, attention_mask=attention_mask)

            # Sequential model
            set_seed(idx)
            outputs2 = seq_model(input_ids, attention_mask)
            assert outputs2.shape == (batch_size, num_labels)

            # Compare outputs
            assert outputs1.logits.allclose(outputs2)

    # print("Testing completed, alles gut!")

    # # test with model splitting
    split_point = 5
    encoder_model = extract_submodel(seq_model, 0, split_point)
    clf_head_model = extract_submodel(seq_model, split_point, -1)

    for idx in range(10):
        batch_size = 8
        seq_length = 128

        # Forward pass through the models to compare outputs
        with torch.no_grad():
            # Original model
            input_ids, attention_mask = get_random_batch(batch_size, seq_length)

            set_seed(idx)
            outputs1 = custom_bert_model(input_ids=input_ids, attention_mask=attention_mask)

            # Sequential model
            set_seed(idx)
            outputs2 = clf_head_model(*encoder_model(input_ids, attention_mask))

            # Compare outputs
            assert outputs1.logits.allclose(outputs2)

    # lastly, try with embedding compression
    for idx in range(10):
        batch_size = 8
        seq_length = 128


        # Forward pass through the models to compare outputs
        with torch.no_grad():
            # Original model
            input_ids, attention_mask = get_random_batch(batch_size, seq_length)

            set_seed(idx)
            outputs1 = custom_bert_model(input_ids=input_ids, attention_mask=attention_mask)

            # Sequential model
            set_seed(idx)
            out = encoder_model(input_ids, attention_mask)

            compressed_embeddings = compress_embeddings_api(out)

            reconstructed_embeddings, extended_attention_mask = \
                uncompress_embeddings(compressed_embeddings, max_len=seq_length)
            outputs2 = clf_head_model(reconstructed_embeddings, extended_attention_mask)

            # Compare outputs
            assert outputs1.logits.allclose(outputs2)

if __name__ == "__main__":
    _test()
    print("Tests passed, alles gut!")
