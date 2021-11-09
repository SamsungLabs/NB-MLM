local base_config = import "base-mlm.jsonnet";

local output_dir = std.extVar("OUTPUT_DIR");
local experiment_dir = std.extVar("EXPERIMENT_DIR");
local path_to_scores = std.extVar("PATH_TO_SCORES");
local scores_dir = std.native("to_path")(experiment_dir, "SCORES");

{
    base_config: base_config {
        task: "temp_masked_lm",
    },

    path_to_scores: std.native("to_path")(scores_dir, path_to_scores),
    log_dir_mask: std.native("to_path")(output_dir, "logmask"),
    log_mask: false
}
