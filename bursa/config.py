from dataclasses import dataclass


@dataclass
class Config:
    auto_post_enabled: bool = True
    model_auto_post_enabled: bool = True
