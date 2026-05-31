import os
from dataclasses import dataclass


SCRIPT_DIR="/home/yeh/workspace/spec/SpecForge/qwen3_8b_dflash"
ROOT_DIR="/home/yeh/workspace/spec/SpecForge"
DATASET_DIR="/home/yeh/workspace/spec/process_datasets"

os.environ["SPECFORGE_DATA_NUM_PROC"]="32"
os.environ["TORCHINDUCTOR_CACHE_DIR"]=f"{ROOT_DIR}/cache/compiled_kernels"

@dataclass
class DFlashTrainConfig:
    """model"""
    target_model_path: str = f"/home/yeh/workspace/spec/models/Qwen/Qwen3-8B"
    target_model_backend: str = "sglang" # ["sglang", "hf"]
    draft_config_path: str = f"{ROOT_DIR}/configs/qwen3-8b-dflash.json"
    block_size: int = 16

    """dataset"""
    train_data_path: str = f"{DATASET_DIR}/sharegpt/sharegpt_train.jsonl"
    chat_template: str = "qwen"
    dataloade_num_workers: int = 8
    build_dataset_num_proc: int = int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8))


    """training"""
    num_epochs: int = 6
    batch_size: int = 2
    learning_rate: float = 6e-4
    warmup_ratio: float = 0.04
    max_grad_norm: float = 1.0
    max_length: int = 3072
    accumulation_steps: int = 1
    seed: int = 42
    # resume # store_true

    """output"""
    output_dir: str = f"{ROOT_DIR}/outputs/qwen3-8b-perfectblend"
    cache_dir: str = f"./cache"
    log_interval: int = 50
    eval_interval: int = 1000
    save_interval: int = 1000



    attention_backend: str = "flex_attention" # ["eager", "sdpa", "flex_attention"]

    """optimization"""
    tp_size: int = 8
    
    """tracker"""
    # TrackerArgs

    """distributed"""
    dist_timeout: int = 30

    """sglang backend"""
    # SGLangBackendArgs



    num_anchors: int = 512
    loss_decay_gamma: float = 7.0


    report_to: str = "wandb"
    wandb_project: str = "spec"
    wandb_name: str = "qwen3-8b-dflash-sharegpt"


