local base_config = import "base-mlm.jsonnet";

{
    base_config: base_config {
        criterion: "cls_bow_mlm",
        arch: "cls_roberta_base",
    },

    cls_metrics_interval: 250,
    grad_norm_interval: 250,
}
