from pipeline.model_utils.model_base import ModelBase

AUDIO_TEXT_MODE = {"audio": "audio_only", "audio+text": "append"}


def is_audio_model(model_path: str) -> bool:
    p = model_path.lower()
    return 'qwen2-audio' in p or 'qwen2_audio' in p


def construct_model_base(model_path: str, modality: str = "text") -> ModelBase:
    if modality != "text" and not is_audio_model(model_path):
        raise ValueError(f"modality={modality!r} needs an audio model, got {model_path}")

    model_path_lower = model_path.lower()
    # Audio models first: their names also contain text-model family names.
    if is_audio_model(model_path):
        from pipeline.model_utils.qwen2_audio_model import Qwen2AudioModel
        return Qwen2AudioModel(model_path, audio_text_mode=AUDIO_TEXT_MODE.get(modality, "audio_only"))
    if 'qwen' in model_path_lower:
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama' in model_path_lower or "nemo" in model_path_lower:
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path_lower:
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path_lower:
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    elif 'yi' in model_path_lower:
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    elif "mistral" in model_path_lower:
        from pipeline.model_utils.mistral_model import MistralModel
        return MistralModel(model_path)
    elif "glm" in model_path_lower:
        from pipeline.model_utils.glm_model import GLM4Model
        return GLM4Model(model_path)
    elif "phi" in model_path_lower:
        from pipeline.model_utils.phi_model import PhiModel
        return PhiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
