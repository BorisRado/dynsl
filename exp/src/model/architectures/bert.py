import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from slbd.torch.sequential_model import FlexibleSequential

from exp.src.common.stochasticity import set_seed


def get_extended_attention_mask(attention_mask, dtype=torch.float32):
    extended_attention_mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - extended_attention_mask) * torch.finfo(dtype).min


def compress_embeddings(embeddings, attention_mask):
    return [embeddings[idx][:attention_mask[idx].sum()] for idx in range(embeddings.shape[0])]


def uncompress_embeddings(compressed_embeddings):
    reconstructed_embeddings = torch.zeros_like(embeddings)
    reconstructed_attention_mask = torch.zeros_like(attention_mask)

    for i, comp_emb in enumerate(compressed_embeddings):
        seq_len = comp_emb.size(0)
        reconstructed_embeddings[i, :seq_len] = comp_emb
        reconstructed_attention_mask[i, :seq_len] = 1

    # Convert reconstructed attention mask to extended format for the classifier
    extended_attention_mask = get_extended_attention_mask(reconstructed_attention_mask)
    return reconstructed_embeddings, extended_attention_mask

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
        return self.pooler(embeddings)


class SequentialBert(FlexibleSequential):
    def __init__(self, **kwargs):
        custom_config = BertConfig(**kwargs)
        custom_bert_model = BertForSequenceClassification(custom_config)
        super().__init__(
            CustomEmbedding(custom_bert_model.bert.embeddings),
            *[CustomTransformerLayer(l) for l in custom_bert_model.bert.encoder.layer],
            CustomPooler(custom_bert_model.bert.pooler),
            custom_bert_model.dropout,
            custom_bert_model.classifier
        )


if __name__ == "__main__":
    seed = 10
    config_kwargs = {
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 512,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
        "num_labels": 10,
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

            # Compare outputs
            assert outputs1.logits.allclose(outputs2)

    # print("Testing completed, alles gut!")

    # # test with model splitting
    split_point = 5
    encoder_model = FlexibleSequential(*seq_model.layers[:split_point])
    clf_head_model = FlexibleSequential(*seq_model.layers[split_point:])
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
            embeddings, _ = encoder_model(input_ids, attention_mask)

            compressed_embeddings = compress_embeddings(embeddings, attention_mask)

            reconstructed_embeddings, extended_attention_mask = \
                uncompress_embeddings(compressed_embeddings)
            outputs2 = clf_head_model(reconstructed_embeddings, extended_attention_mask)

            # Compare outputs
            assert outputs1.logits.allclose(outputs2)
