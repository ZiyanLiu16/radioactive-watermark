import torch
from typing import Tuple, Any, Optional
from importlib.util import find_spec
from transformers import AutoTokenizer, AutoModelForCausalLM
print("transformers version:", transformers.__version__)


def _mlx_available() -> bool:
    """Return True if MLX and mlx-lm are importable, False otherwise."""
    return (find_spec("mlx") is not None) and (find_spec("mlx_lm") is not None)


def load_model_and_tokenizer(
    model_name: str,
    backend: str = "auto",
    device_map: Optional[str] = "auto",
    torch_dtype: Optional[str] = "auto",
    trust_remote_code: Optional[bool] = None,
    use_auth_token: Optional[bool] = None,
) -> Tuple[Any, Any, str]:
    """Load a causal language model and tokenizer with a simple, pluggable backend.

    This utility centralizes model loading across the codebase so all callers use the
    exact same configuration and behavior. It supports both the standard PyTorch
    Transformers backend and Apple's MLX backend on Apple Silicon.

    Args:
        model_name: Hugging Face model ID or local path (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        backend: Which backend to use. One of:
            - "auto" (default): use MLX if available, otherwise PyTorch
            - "pytorch": force PyTorch/Transformers
            - "mlx": force MLX (requires `mlx` and `mlx-lm`)
        device_map: Forwarded to `AutoModelForCausalLM.from_pretrained` when using PyTorch.
            Common values: "auto", "cpu", or a device map dict. Use None to omit.
        torch_dtype: Forwarded to `AutoModelForCausalLM.from_pretrained` when using PyTorch.
            Common values: "auto", torch.float16, torch.bfloat16. Use None to omit.
        trust_remote_code: Forwarded to `from_pretrained` (PyTorch only). Set to True if a
            model requires custom code from its repository.
        use_auth_token: Forwarded to both tokenizer and model `from_pretrained` (PyTorch only).
            Set to True or an HF token string if accessing gated/private models.

    Returns:
        A 3-tuple of:
            - model: The loaded causal LM (PyTorch nn.Module or MLX model)
            - tokenizer: The corresponding tokenizer
            - backend_used: A string indicating the backend actually used ("pytorch" or "mlx")

    Behavior:
        - If `backend` is "auto", this function detects MLX availability and prefers MLX on
          Apple Silicon; otherwise it falls back to PyTorch.
        - For PyTorch, the tokenizer's `pad_token` is set to `eos_token` if not set.
        - All provided kwargs are forwarded only to the relevant backend.
    """
    if backend == "auto":
        backend = "mlx" if _mlx_available() else "pytorch"

    if backend == "mlx":
        if not _mlx_available():
            raise ImportError("MLX backend requested but not available. Install mlx and mlx-lm.")
        from mlx_lm import load as mlx_load  # lazy import
        model, tokenizer = mlx_load(model_name)
        return model, tokenizer, backend

    # Default to PyTorch
    tokenizer_kwargs = {}
    if use_auth_token is not None:
        tokenizer_kwargs["use_auth_token"] = use_auth_token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    model_kwargs = {}
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if trust_remote_code is not None:
        model_kwargs["trust_remote_code"] = trust_remote_code
    if use_auth_token is not None:
        model_kwargs["use_auth_token"] = use_auth_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, "pytorch"


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    backend: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response from a single-turn user prompt."""
    if backend == "mlx":
        from mlx_lm import generate as mlx_generate  # lazy import
        text = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )
        return text.strip()

    # PyTorch path: use chat template like Llama-3
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response.strip() 