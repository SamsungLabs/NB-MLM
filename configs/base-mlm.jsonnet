local output_dir = std.extVar("OUTPUT_DIR");
local user_dir = std.extVar("USER_DIR");
local fairseq_dir = std.extVar("FAIRSEQ_DIR");

{
    restore_file: std.native("to_path")(fairseq_dir, "roberta.base/model.pt"),
    tensorboard_logdir: std.native("to_path")(output_dir, "log"),
    save_dir: std.native("to_path")(output_dir, "ckpt"),
    user_dir: user_dir,

    task: "masked_lm",
    criterion: "masked_lm",
    arch: "roberta_base",

    sample_break_mode: "complete_doc",
    tokens_per_sample: 512,

    lr_scheduler: "polynomial_decay",
    optimizer: "adam",
    adam_betas: "0.9,0.98",
    adam_eps: 1e-6,

    clip_norm: 0.0,
    dropout: 0.1,
    attention_dropout: 0.1,
    weight_decay: 0.01,

    log_format: "simple",
    log_interval: 1,
    fp16: true,
    ddp_backend: "no_c10d",
    skip_invalid_size_inputs_valid_test: true,
}
