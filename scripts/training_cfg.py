from dataclasses import dataclass
import sys
from pathlib import Path
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingConfig:
    log_dir: str
    output_dir: str
    output_data_dir: str
    data_dir: str
    ckpt_name: str
    data_test_path: str
    rank: int
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    seed: int = 42
    pretrained_model_name_or_path: str = 'runwayml/stable-diffusion-v1-5'
    enable_xformers_memory_efficient_attention: bool = True
    config: str = "VQ4_mir.yaml"
    data_root_path: str = ""
    checkpoints_total_limit: int = None
    validation_epochs: int = 1

    data_json_file: str = ""
    # AdamW
    learning_rate: float = 1e-4
    learning_rate_lora: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    resolution: int = 512
    n_epochs: int = 100
    checkpointing_steps: int = 1000
    train_batch_size: int = 4
    dataloader_num_workers: int = 1

    lr_scheduler_name: str = 'constant'
    class_labels_conditioning: str = 'timesteps'

    resume_from_checkpoint: bool = False
    noise_offset: float = 0.1
    max_grad_norm: float = 1.0
    lr: float = 8e-5
    secret_len: int = 48
    lambda_1: float = 40
    lambda_2: float = 1
    lambda_3: float = 20
    lambda_4: float = 1
    max_image_weight_ratio: float = 2.0


def load_training_config(config_path: str) -> TrainingConfig:
    data_dict = OmegaConf.load(config_path)
    return TrainingConfig(**data_dict)
