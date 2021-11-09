local output_dir = std.extVar("OUTPUT_DIR");
local user_dir = std.extVar("USER_DIR");
local fairseq_dir = std.extVar("FAIRSEQ_DIR");
local roberta_dir = std.extVar("ROBERTA_DIR");

{
    base_config: {
        restore_file: std.extVar("RESTORE_FILE"),
        tensorboard_logdir: std.native("to_path")(output_dir, "log"),
        save_dir: std.native("to_path")(output_dir, "ckpt"),
        user_dir: user_dir,

        task: "sentence_prediction",
        criterion: "sentence_prediction",
        arch: "roberta_base",

        max_positions: 512,
        max_tokens: 4400,

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
        fp16_init_scale: 4,
        threshold_loss_scale: 1,
        fp16_scale_window: 128,
        best_checkpoint_metric: "accuracy",
        maximize_best_checkpoint_metric: true,
        required_batch_size_multiple: 1,
        init_token: 0,
        separator_token: 2,

        reset_optimizer: true,
        reset_dataloader: true,
        reset_meters: true,

        skip_invalid_size_inputs_valid_test: true,
        
        find_unused_parameters: true,
        no_epoch_checkpoints: true,
    }
}
