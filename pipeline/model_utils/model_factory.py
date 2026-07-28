from pipeline.model_utils.model_base import ModelBase


def construct_model_base(model_path: str) -> ModelBase:
    model_path_lower = model_path.lower()

    # --- Speech models (checked first to avoid false matches) ---
    if 'qwen2.5-omni' in model_path_lower or 'qwen_omni' in model_path_lower:
        from pipeline.model_utils.qwen_omni_model import QwenOmniModel
        return QwenOmniModel(model_path)

    if 'csm' in model_path_lower or 'sesame' in model_path_lower:
        from pipeline.model_utils.csm_model import CSMModel
        return CSMModel(model_path)

    if 'moshi' in model_path_lower or 'kyutai' in model_path_lower:
        from pipeline.model_utils.moshi_model import MoshiModel
        return MoshiModel(model_path)

    # --- Text models (original routing) ---
    if 'qwen' in model_path_lower:
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)

    if 'llama' in model_path_lower or 'nemo' in model_path_lower:
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)

    if 'gemma' in model_path_lower:
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)

    if 'yi' in model_path_lower:
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)

    if 'mistral' in model_path_lower:
        from pipeline.model_utils.mistral_model import MistralModel
        return MistralModel(model_path)

    if 'glm' in model_path_lower:
        from pipeline.model_utils.glm_model import GLM4Model
        return GLM4Model(model_path)

    if 'phi' in model_path_lower:
        from pipeline.model_utils.phi_model import PhiModel
        return PhiModel(model_path)

    raise ValueError(f"Unknown model family: {model_path}")