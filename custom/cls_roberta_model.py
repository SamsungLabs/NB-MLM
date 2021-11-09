import torch
from fairseq.checkpoint_utils import prune_state_dict
from overrides import overrides

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.roberta import (
    RobertaModel,
    RobertaEncoder,
    RobertaClassificationHead
)


@register_model("cls_roberta")
class CLSRobertaModel(RobertaModel):
    def __init__(self, args, encoder: RobertaEncoder):
        super().__init__(args=args, encoder=encoder)

    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser)
        parser.add_argument("--tie-cls", type=bool, default=False,
                            help="whatever tie input embeddings and CLS head weights")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        cls_base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = CLSRobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def load_state_dict(self, state_dict, strict=False, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, args)
        return super().load_state_dict(new_state_dict, False)

    @overrides
    def forward(self, src_tokens: torch.Tensor,
                masked_tokens: torch.Tensor = None,
                return_all_hiddens: bool = False,
                **kwargs):
        x, extra = self.decoder(src_tokens, masked_tokens, return_all_hiddens, **kwargs)
        return x, extra


class CLSRobertaEncoder(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args=args, dictionary=dictionary)
        self.clf_head = RobertaClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=len(self.dictionary),
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout
        )
        if args.tie_cls:
            self.clf_head.out_proj.weight = self.sentence_encoder.embed_tokens.weight

    @overrides
    def forward(self, src_tokens: torch.Tensor,
                masked_tokens: torch.Tensor = None,
                return_all_hiddens: bool = False,
                **unused):
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        x = self.output_layer(x, masked_tokens=masked_tokens)

        cls_states = extra["cls_states"]
        cls_bow_logits = self.clf_head(cls_states.unsqueeze(1))
        extra["cls_bow_logits"] = cls_bow_logits

        return x, extra

    def extract_features(self, src_tokens: torch.Tensor,
                         return_all_hiddens: bool = False,
                         **unused):
        inner_states, cls_states = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1]
        return features, {"cls_states": cls_states, "inner_states": inner_states if return_all_hiddens else None}

    @overrides
    def output_layer(self, features: torch.Tensor,
                     masked_tokens: torch.Tensor = None,
                     **unused):
        return self.lm_head(features, masked_tokens)


@register_model_architecture("cls_roberta", "cls_roberta")
def cls_base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.tie_cls = getattr(args, "tie_cls", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)


@register_model_architecture("cls_roberta", "cls_roberta_base")
def cls_roberta_base_architecture(args):
    cls_base_architecture(args)


@register_model_architecture("cls_roberta", "cls_roberta_large")
def cls_roberta_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    cls_base_architecture(args)
