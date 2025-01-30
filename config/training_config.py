from dataclasses import dataclass

@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0 