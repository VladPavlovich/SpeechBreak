import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    p_train: int = 1.0
    p_val: int = 0.90
    p_test: int = 1.0
    filter_train: bool = True
    filter_val: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench", "advbench", "tdc2023", "harmbench", "malicious_instruct")
    max_new_tokens: int = 500 #nonthinking
    max_new_tokens_think: int = 1500 #thinking
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "llamaguard3")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching")
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048
    save_dir: str = "runs"
    # SpeechBreak: what the model sees. "text" | "audio" | "audio+text".
    # train_modality (default: modality) is used for direction extraction and
    # selection (train/val); modality for test splits and eval datasets, so
    # e.g. train_modality="text", modality="audio" tests cross-modal transfer.
    modality: str = "text"
    train_modality: str = None
    tts: str = "mms"
    # Extra eval sets that are used when modality != "text" (ship real audio).
    audio_evaluation_datasets: Tuple[str] = ("voicebench_advbench",)

    def __post_init__(self):
        if self.train_modality is None:
            self.train_modality = self.modality
        if {self.modality, self.train_modality} == {"audio", "audio+text"}:
            raise ValueError("audio and audio+text cannot be mixed in one run "
                             "(whether the transcript is shown is a model-level setting)")
        if self.modality != "text":
            self.evaluation_datasets = tuple(self.evaluation_datasets) + tuple(self.audio_evaluation_datasets)

    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{self.save_dir}", self.model_alias)
