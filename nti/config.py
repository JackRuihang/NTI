from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    codebook_dim: int = 256
    sequence_length: int = 256
    num_classes: int = 1000
    hidden_size: int = 768
    depth: int = 20
    num_heads: int = 16
    mlp_size: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    class_dropout: float = 0.1
    qk_norm: bool = True
    layer_norm_eps: float = 1.0e-6

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size < self.codebook_dim:
            raise ValueError("ICB requires hidden_size >= codebook_dim")
        if self.sequence_length <= 0 or self.vocab_size <= 1:
            raise ValueError("sequence_length and vocab_size must be positive")
        if not 0.0 <= self.class_dropout < 1.0:
            raise ValueError("class_dropout must be in [0, 1)")


@dataclass
class ICBConfig:
    enabled: bool = False
    transport_mode: str = "learned_qr"
    input_scale: float = 1.0
    input_residual_ratio: float = 0.1
    geometry_temperature: float = 16.0
    residual_confidence_cap: float = 0.01
    qr_epsilon: float = 1.0e-6

    def validate(self, vocab_size: int) -> None:
        if self.enabled:
            raise ValueError("Codebook bridges are not part of NTI")
        if self.transport_mode not in {"learned_qr", "fixed_stiefel"}:
            raise ValueError("transport_mode must be learned_qr or fixed_stiefel")
        if self.input_scale <= 0 or self.geometry_temperature <= 0:
            raise ValueError("ICB scales and temperatures must be positive")
        if self.input_residual_ratio < 0:
            raise ValueError("input_residual_ratio must be non-negative")
        if not 1.0 / vocab_size < self.residual_confidence_cap < 1.0:
            raise ValueError("residual_confidence_cap must be in (1 / vocab_size, 1)")


@dataclass
class TokenInterfaceConfig:
    mode: str = "free"
    read_weight: float = 0.0
    write_weight: float = 0.0

    def validate(self, *, icb_enabled: bool, training: "TrainingConfig") -> None:
        if self.mode != "free" or icb_enabled:
            raise ValueError("NTI requires the standard free token interface")
        if self.read_weight != 0 or self.write_weight != 0:
            raise ValueError("NTI does not use auxiliary token-interface losses")


@dataclass
class TrainingConfig:
    arm: str = "nti_soft"
    target_mode: str = "tv_history_conditioned"
    history_source: str = "online_two_pass"
    replay_bank_root: str = ""
    replay_start_step: int = 0
    replay_refresh_steps: int = 10_000
    replay_end_step: int = 0
    replay_top_k: int = 32
    replay_target_encoding: str = "sparse_topk"
    replay_stream_mode: str = "mixed"
    alpha: float = 0.5
    codebook_kernel_temperature: float = 0.07
    history_temperature: float = 1.0
    history_top_k: int = 0
    history_top_p: float = 1.0
    replacement_max_ratio: float = 0.5
    replacement_ratio_sampling: str = "fixed"
    replacement_start_fraction: float = 0.0
    replacement_warmup_fraction: float = 0.3
    replacement_decay_start_fraction: float = 0.5
    replacement_end_fraction: float = 1.0
    replacement_schedule: str = "linear_hold"
    branch_active_only_when_replacement_positive: bool = False
    auxiliary_loss_gate: str = "constant"
    clean_loss_weight: float = 1.0
    branch_loss_weight: float = 1.0
    branch_image_fraction: float = 1.0
    branch_loss_scope: str = "active_prefix"
    epochs: int = 400
    max_steps: int = 250_000
    global_batch_size: int = 2048
    per_device_batch_size: int = 32
    learning_rate: float = 4.0e-4
    end_learning_rate: float = 1.0e-5
    warmup_steps: int = 62_500
    beta1: float = 0.9
    beta2: float = 0.96
    weight_decay: float = 0.03
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    seed: int = 42
    num_workers: int = 8
    log_every: int = 50
    save_every: int = 50_000
    checkpoint_milestone_every: int = 0
    checkpoint_keep_last: int = 0

    def validate(self) -> None:
        if self.history_source not in {"online_two_pass", "lagged_replay"}:
            raise ValueError("history_source must be online_two_pass or lagged_replay")
        if self.target_mode != "tv_history_conditioned":
            raise ValueError("NTI requires tv_history_conditioned targets")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.codebook_kernel_temperature <= 0 or self.history_temperature <= 0:
            raise ValueError("target/history temperatures must be positive")
        if self.history_top_k < 0 or not 0.0 < self.history_top_p <= 1.0:
            raise ValueError("history_top_k/top_p are invalid")
        if not 0.0 <= self.replacement_max_ratio <= 1.0:
            raise ValueError("replacement_max_ratio must be in [0, 1]")
        if self.replacement_ratio_sampling not in {
            "fixed",
            "per_image_uniform",
        }:
            raise ValueError(
                "replacement_ratio_sampling must be fixed or per_image_uniform"
            )
        if not 0.0 <= self.replacement_start_fraction <= 1.0:
            raise ValueError("replacement_start_fraction must be in [0, 1]")
        if not 0.0 < self.replacement_warmup_fraction <= 1.0:
            raise ValueError("replacement_warmup_fraction must be in (0, 1]")
        if not 0.0 <= self.replacement_decay_start_fraction <= 1.0:
            raise ValueError("replacement_decay_start_fraction must be in [0, 1]")
        if not 0.0 < self.replacement_end_fraction <= 1.0:
            raise ValueError("replacement_end_fraction must be in (0, 1]")
        if self.replacement_schedule not in {
            "constant",
            "linear_hold",
            "ramp_cosine_zero",
            "bell_cosine",
        }:
            raise ValueError(
                "replacement_schedule must be constant, linear_hold, ramp_cosine_zero, "
                "or bell_cosine"
            )
        if (
            self.replacement_schedule == "ramp_cosine_zero"
            and self.replacement_warmup_fraction >= 1.0
        ):
            raise ValueError(
                "ramp_cosine_zero requires replacement_warmup_fraction < 1"
            )
        if self.replacement_schedule == "bell_cosine" and not (
            0.0
            <= self.replacement_start_fraction
            < self.replacement_warmup_fraction
            <= self.replacement_decay_start_fraction
            < self.replacement_end_fraction
            <= 1.0
        ):
            raise ValueError("bell_cosine requires start < warmup <= decay_start < end")
        if not isinstance(self.branch_active_only_when_replacement_positive, bool):
            raise TypeError("branch_active_only_when_replacement_positive must be bool")
        if self.auxiliary_loss_gate not in {
            "constant",
            "normalized_replacement",
        }:
            raise ValueError(
                "auxiliary_loss_gate must be constant or normalized_replacement"
            )
        if self.auxiliary_loss_gate == "normalized_replacement":
            if self.replacement_max_ratio <= 0.0:
                raise ValueError(
                    "normalized_replacement auxiliary loss gate requires a "
                    "positive replacement_max_ratio"
                )
            if not self.branch_active_only_when_replacement_positive:
                raise ValueError(
                    "normalized_replacement auxiliary loss gate requires "
                    "branch_active_only_when_replacement_positive=true"
                )
        if self.clean_loss_weight < 0 or self.branch_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.clean_loss_weight == 0 and self.branch_loss_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        if (
            not isfinite(self.branch_image_fraction)
            or not 0.0 <= self.branch_image_fraction <= 1.0
        ):
            raise ValueError("branch_image_fraction must be finite and in [0, 1]")
        if self.clean_loss_weight == 0 and self.branch_image_fraction == 0:
            raise ValueError(
                "branch_image_fraction=0 requires a positive clean_loss_weight"
            )
        if self.branch_loss_scope not in {"all", "active_prefix"}:
            raise ValueError("branch_loss_scope must be all or active_prefix")
        if (
            self.target_mode == "tv_history_conditioned"
            and self.branch_loss_scope != "active_prefix"
        ):
            raise ValueError(
                "history-gated targets require branch_loss_scope=active_prefix"
            )
        if self.epochs <= 0 or self.max_steps <= 0:
            raise ValueError("epochs and max_steps must be positive")
        if self.global_batch_size <= 0 or self.per_device_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps]")
        if self.learning_rate <= 0 or self.end_learning_rate < 0:
            raise ValueError("learning rates are invalid")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.log_every <= 0 or self.save_every <= 0:
            raise ValueError("log_every and save_every must be positive")
        if self.checkpoint_milestone_every < 0 or self.checkpoint_keep_last < 0:
            raise ValueError("checkpoint retention values must be non-negative")
        if (
            self.checkpoint_milestone_every > 0
            and self.checkpoint_milestone_every % self.save_every != 0
        ):
            raise ValueError(
                "checkpoint_milestone_every must be a multiple of save_every"
            )
        if self.replay_target_encoding not in {
            "sparse_topk",
            "categorical_mc",
            "conditional_non_gt_mc",
        }:
            raise ValueError(
                "replay_target_encoding must be sparse_topk, categorical_mc, "
                "or conditional_non_gt_mc"
            )
        if self.replay_stream_mode not in {"mixed", "all_records"}:
            raise ValueError("replay_stream_mode must be mixed or all_records")
        if self.history_source == "online_two_pass":
            if self.replay_bank_root:
                raise ValueError(
                    "replay_bank_root requires history_source=lagged_replay"
                )
            return
        if self.target_mode != "tv_history_conditioned":
            raise ValueError(
                "lagged_replay requires a supported history-dependent target"
            )
        if self.replay_stream_mode == "mixed":
            if self.clean_loss_weight <= 0.0 or self.branch_loss_weight <= 0.0:
                raise ValueError(
                    "mixed lagged_replay requires positive clean and branch weights"
                )
        else:
            if self.clean_loss_weight != 0.0 or self.branch_loss_weight != 1.0:
                raise ValueError(
                    "all-record replay requires clean_loss_weight=0 and "
                    "branch_loss_weight=1"
                )
            if self.branch_image_fraction != 1.0:
                raise ValueError("all-record replay requires branch_image_fraction=1")
            if self.replay_target_encoding != "conditional_non_gt_mc":
                raise ValueError(
                    "all-record replay requires conditional_non_gt_mc encoding"
                )
        if not self.replay_bank_root:
            raise ValueError("lagged_replay requires replay_bank_root")
        if self.replay_start_step < 0:
            raise ValueError("lagged_replay replay_start_step must be non-negative")
        if self.replay_refresh_steps <= 0:
            raise ValueError("replay_refresh_steps must be positive")
        if not self.replay_start_step < self.replay_end_step <= self.max_steps:
            raise ValueError("replay steps must satisfy 0 <= start < end <= max_steps")
        if self.replay_start_step % self.replay_refresh_steps:
            raise ValueError("replay_start_step must align to replay_refresh_steps")
        if self.replay_end_step % self.replay_refresh_steps:
            raise ValueError("replay_end_step must align to replay_refresh_steps")
        if self.replay_top_k <= 0:
            raise ValueError("replay_top_k must be positive")
        if self.replay_target_encoding == "sparse_topk" and self.replay_top_k >= 1024:
            raise ValueError("sparse replay_top_k must be in [1, 1023]")
        scheduled_end = round(self.replacement_end_fraction * self.max_steps)
        if scheduled_end != self.replay_end_step:
            raise ValueError(
                "replay_end_step must equal the replacement schedule end step"
            )
        if not self.branch_active_only_when_replacement_positive:
            raise ValueError(
                "lagged_replay requires branch_active_only_when_replacement_positive"
            )
        expected_gate = (
            "normalized_replacement"
            if self.replay_stream_mode == "mixed"
            else "constant"
        )
        if self.auxiliary_loss_gate != expected_gate:
            raise ValueError(
                f"{self.replay_stream_mode} lagged_replay requires "
                f"auxiliary_loss_gate={expected_gate}"
            )


@dataclass
class MonitoringConfig:
    """Training observability that does not change optimization semantics."""

    preview_every: int = 0
    preview_seed: int = 42
    preview_class_ids: tuple[int, ...] = (
        1,
        7,
        282,
        604,
        724,
        179,
        751,
        404,
        850,
        954,
    )
    preview_rows: int = 2
    preview_temperature: float = 1.0
    preview_top_k: int = 0
    preview_top_p: float = 1.0
    preview_guidance_scale: float = 22.0
    preview_guidance_scale_pow: float = 2.75
    preview_decode_batch_size: int = 1
    grad_norm_every: int = 0

    def __post_init__(self) -> None:
        # YAML represents sequences as lists; keep the in-memory schema stable.
        if isinstance(self.preview_class_ids, list):
            self.preview_class_ids = tuple(self.preview_class_ids)

    def validate(self, num_classes: int) -> None:
        if self.preview_every < 0 or self.grad_norm_every < 0:
            raise ValueError("monitoring intervals must be non-negative")
        if self.preview_every == 0:
            return
        if self.preview_seed < 0:
            raise ValueError("preview_seed must be non-negative")
        if not self.preview_class_ids:
            raise ValueError("preview_class_ids must be nonempty")
        if any(
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < num_classes
            for class_id in self.preview_class_ids
        ):
            raise ValueError(
                f"preview_class_ids must contain integer IDs in [0, {num_classes})"
            )
        if len(set(self.preview_class_ids)) != len(self.preview_class_ids):
            raise ValueError("preview_class_ids must not contain duplicates")
        if self.preview_rows <= 0:
            raise ValueError("preview_rows must be positive")
        if not isfinite(self.preview_temperature) or self.preview_temperature <= 0:
            raise ValueError("preview_temperature must be finite and positive")
        if self.preview_top_k < 0:
            raise ValueError("preview_top_k must be non-negative")
        if not isfinite(self.preview_top_p) or not 0 < self.preview_top_p <= 1:
            raise ValueError("preview_top_p must be finite and in (0, 1]")
        if not isfinite(self.preview_guidance_scale) or self.preview_guidance_scale < 0:
            raise ValueError("preview_guidance_scale must be finite and non-negative")
        if (
            not isfinite(self.preview_guidance_scale_pow)
            or self.preview_guidance_scale_pow <= 0
        ):
            raise ValueError("preview_guidance_scale_pow must be finite and positive")
        if self.preview_decode_batch_size <= 0:
            raise ValueError("preview_decode_batch_size must be positive")


@dataclass
class DataConfig:
    train_manifest: str = ""
    validation_manifest: str = ""
    tokenizer_checkpoint: str = "assets/tokenizer.bin"
    rear_source_root: str = "external/reAR"
    image_root: str = "data/imagenet"


@dataclass
class RunConfig:
    name: str = "nti_soft_rears"
    output_dir: str = "outputs"
    resume: str = ""
    compile_model: bool = False
    deterministic: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    icb: ICBConfig = field(default_factory=ICBConfig)
    token_interface: TokenInterfaceConfig = field(default_factory=TokenInterfaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_ConfigT = TypeVar("_ConfigT")


def _construct_dataclass(cls: Type[_ConfigT], values: Mapping[str, Any]) -> _ConfigT:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(values))


def experiment_config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    allowed = {
        "model",
        "icb",
        "token_interface",
        "training",
        "monitoring",
        "data",
        "run",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown ExperimentConfig sections: {sorted(unknown)}")
    icb = _construct_dataclass(ICBConfig, values.get("icb", {}))
    training = _construct_dataclass(TrainingConfig, values.get("training", {}))
    if "token_interface" in values:
        token_interface = _construct_dataclass(
            TokenInterfaceConfig, values["token_interface"]
        )
    else:
        token_interface = TokenInterfaceConfig(mode="icb" if icb.enabled else "free")
    config = ExperimentConfig(
        model=_construct_dataclass(ModelConfig, values.get("model", {})),
        icb=icb,
        token_interface=token_interface,
        training=training,
        monitoring=_construct_dataclass(MonitoringConfig, values.get("monitoring", {})),
        data=_construct_dataclass(DataConfig, values.get("data", {})),
        run=_construct_dataclass(RunConfig, values.get("run", {})),
    )
    config.model.validate()
    config.icb.validate(config.model.vocab_size)
    config.training.validate()
    config.monitoring.validate(config.model.num_classes)
    config.token_interface.validate(
        icb_enabled=config.icb.enabled, training=config.training
    )
    if config.training.history_source == "lagged_replay" and (
        config.icb.enabled or config.token_interface.mode != "free"
    ):
        raise ValueError("lagged_replay requires the free token interface")
    if (
        config.training.history_source == "lagged_replay"
        and config.training.replay_target_encoding == "sparse_topk"
        and config.training.replay_top_k >= config.model.vocab_size
    ):
        raise ValueError("replay_top_k must be smaller than model.vocab_size")
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load NTI configs") from error
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, Mapping):
        raise ValueError("the config root must be a mapping")
    return experiment_config_from_dict(values)


def rear_s_config(*, icb: bool = False) -> ExperimentConfig:
    config = ExperimentConfig()
    config.icb.enabled = icb
    config.token_interface.mode = "icb" if icb else "free"
    config.model.validate()
    config.icb.validate(config.model.vocab_size)
    config.training.validate()
    config.monitoring.validate(config.model.num_classes)
    config.token_interface.validate(
        icb_enabled=config.icb.enabled, training=config.training
    )
    return config


def tiny_config(*, icb: bool = False) -> ExperimentConfig:
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=32,
            codebook_dim=8,
            sequence_length=12,
            num_classes=5,
            hidden_size=32,
            depth=2,
            num_heads=4,
            mlp_size=64,
            dropout=0.0,
            attention_dropout=0.0,
            class_dropout=0.0,
        ),
        icb=ICBConfig(
            enabled=icb,
            residual_confidence_cap=0.2,
        ),
        token_interface=TokenInterfaceConfig(mode="icb" if icb else "free"),
        training=TrainingConfig(global_batch_size=8, per_device_batch_size=2),
    )
    config.model.validate()
    config.icb.validate(config.model.vocab_size)
    config.training.validate()
    config.monitoring.validate(config.model.num_classes)
    config.token_interface.validate(
        icb_enabled=config.icb.enabled, training=config.training
    )
    return config
