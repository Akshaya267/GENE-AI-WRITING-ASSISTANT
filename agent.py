"""
agent.py
--------
GENE — Generative ENgine Agent

AI Agent Controller for the GENE AI Writing Assistant.

Architecture:
    User
      ↓
    Streamlit UI
      ↓
    GeneAgent
      ↓
    Hugging Face Transformers
      ↓
    Qwen2.5-Instruct
      ↓
    Generated Answer

Design goals:
    • Fast model reuse
    • Better instruction following
    • Proper Qwen chat-template usage
    • Reduced unnecessary generation
    • Cleaner output
    • Stable CPU/GPU inference
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline,
)


# ============================================================================
# TASK INSTRUCTIONS
# ============================================================================

TASK_INSTRUCTIONS = {
    "Story": (
        "Write an engaging and coherent short story based only on the "
        "user's request. Use natural language and appropriate structure. "
        "Do not add unrelated information. Return only the finished story.\n\n"
        "USER REQUEST:\n{prompt}"
    ),

    "Email": (
        "Write a professional email based exactly on the user's request.\n"
        "Requirements:\n"
        "- Include a clear Subject line.\n"
        "- Use an appropriate greeting.\n"
        "- Keep the body concise and natural.\n"
        "- Include only information provided or clearly requested by the user.\n"
        "- Do not invent names, dates, places, events, or facts.\n"
        "- Use placeholders only when necessary.\n"
        "- End with a professional sign-off.\n"
        "Return only the completed email.\n\n"
        "USER REQUEST:\n{prompt}"
    ),

    "Summary": (
        "Summarize the user's provided text accurately and concisely.\n"
        "Keep the important facts, ideas, conclusions, and relationships.\n"
        "Do not introduce information that is not present in the source.\n"
        "Return only the summary.\n\n"
        "TEXT TO SUMMARIZE:\n{prompt}"
    ),

    "Explanation": (
        "Explain the user's question or topic clearly and accurately.\n"
        "Start with the direct answer.\n"
        "Then explain the key points in simple language.\n"
        "Use a short example when useful.\n"
        "Avoid unnecessary jargon and repetition.\n"
        "Do not invent facts.\n"
        "Return only the explanation.\n\n"
        "USER QUESTION:\n{prompt}"
    ),

    "Creative Text": (
        "Create original text that directly satisfies the user's request.\n"
        "Follow the requested style, format, tone, and purpose.\n"
        "Do not add unrelated information.\n"
        "Return only the requested text.\n\n"
        "USER PROMPT:\n{prompt}"
    ),
}


# ============================================================================
# DEFAULT MODEL
# ============================================================================

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

@dataclass
class GenerationParams:
    """
    Generation settings used by GENE.

    Defaults are optimized for reasonably fast and focused responses.
    """

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_return_sequences: int = 1
    repetition_penalty: float = 1.05
    do_sample: bool = True


# ============================================================================
# AGENT RESPONSE
# ============================================================================

@dataclass
class AgentResponse:
    success: bool
    output: str = ""
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    task: str = ""
    instruction: str = ""


# ============================================================================
# VALIDATION ERROR
# ============================================================================

class ValidationError(Exception):
    """Raised when user input is invalid."""


# ============================================================================
# GENE AGENT
# ============================================================================

class GeneAgent:
    """
    Main GENE AI controller.

    The expensive tokenizer/model loading happens only once per
    GeneAgent instance.

    app.py caches the GeneAgent using Streamlit's cache_resource,
    so the same model can be reused between requests.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.model_name = model_name

        self._pipe: Optional[Pipeline] = None
        self._tokenizer = None
        self._model = None

        # ---------------------------------------------------------------
        # Device selection
        # ---------------------------------------------------------------

        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_id = 0
        else:
            self.device = "cpu"
            self.device_id = -1

    # ========================================================================
    # MODEL LOADING
    # ========================================================================

    def load(self) -> None:
        """
        Load tokenizer and model once.

        Subsequent requests reuse the already loaded model.
        """

        if self._pipe is not None:
            return

        try:

            # -----------------------------------------------------------
            # TOKENIZER
            # -----------------------------------------------------------

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            if self._tokenizer.pad_token_id is None:

                self._tokenizer.pad_token = (
                    self._tokenizer.eos_token
                )

            # -----------------------------------------------------------
            # MODEL
            # -----------------------------------------------------------

            if self.device == "cuda":

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            else:

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

                self._model.eval()

            # -----------------------------------------------------------
            # PIPELINE
            # -----------------------------------------------------------

            self._pipe = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.device_id,
            )

        except Exception as exc:

            self._pipe = None
            self._tokenizer = None
            self._model = None

            raise RuntimeError(
                f"Unable to load model '{self.model_name}': {exc}"
            ) from exc

    # ========================================================================
    # STATUS
    # ========================================================================

    @property
    def is_loaded(self) -> bool:
        """Return True when the model pipeline is ready."""

        return self._pipe is not None

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def validate_input(
        prompt: str,
        task: str,
    ) -> None:

        if prompt is None:
            raise ValidationError(
                "Prompt cannot be empty."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValidationError(
                "Prompt cannot be empty."
            )

        if len(prompt) < 3:
            raise ValidationError(
                "Prompt is too short. Please provide more detail."
            )

        if len(prompt) > 4000:
            raise ValidationError(
                "Prompt is too long. Maximum 4000 characters."
            )

        if task not in TASK_INSTRUCTIONS:
            raise ValidationError(
                f"Unknown task type: {task}"
            )

    # ========================================================================
    # INSTRUCTION CREATION
    # ========================================================================

    @staticmethod
    def build_instruction(
        prompt: str,
        task: str,
    ) -> str:

        if task not in TASK_INSTRUCTIONS:

            raise ValidationError(
                f"Unknown task type: {task}"
            )

        return TASK_INSTRUCTIONS[task].format(
            prompt=prompt.strip()
        )

    # ========================================================================
    # CHAT MESSAGES
    # ========================================================================

    @staticmethod
    def _build_messages(
        instruction: str,
    ) -> list[dict[str, str]]:

        return [
            {
                "role": "system",
                "content": (
                    "You are GENE, a precise and helpful AI writing "
                    "assistant. Follow the user's request exactly. "
                    "Answer directly. Do not invent facts. "
                    "Do not repeat the instructions."
                ),
            },
            {
                "role": "user",
                "content": instruction,
            },
        ]

    # ========================================================================
    # OUTPUT CLEANING
    # ========================================================================

    @staticmethod
    def _clean_output(
        text: str,
    ) -> str:

        if not text:
            return ""

        text = str(text).strip()

        # ---------------------------------------------------------------
        # Remove common assistant prefixes.
        # ---------------------------------------------------------------

        prefixes = (
            "assistant:",
            "Assistant:",
            "ASSISTANT:",
            "GENE:",
            "Gene:",
        )

        for prefix in prefixes:

            if text.startswith(prefix):

                text = text[
                    len(prefix):
                ].strip()

                break

        # ---------------------------------------------------------------
        # Remove Qwen/template artifacts.
        # ---------------------------------------------------------------

        markers = (
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|end|>",
        )

        for marker in markers:

            text = text.replace(
                marker,
                "",
            )

        return text.strip()

    # ========================================================================
    # GENERATION
    # ========================================================================

    def generate(
        self,
        prompt: str,
        task: str,
        params: GenerationParams,
    ) -> AgentResponse:

        # ----------------------------------------------------------------
        # 1. VALIDATION
        # ----------------------------------------------------------------

        try:

            self.validate_input(
                prompt,
                task,
            )

        except ValidationError as exc:

            return AgentResponse(
                success=False,
                error=str(exc),
                task=task,
            )

        # ----------------------------------------------------------------
        # 2. BUILD INSTRUCTION
        # ----------------------------------------------------------------

        instruction = self.build_instruction(
            prompt,
            task,
        )

        # ----------------------------------------------------------------
        # 3. LOAD MODEL
        # ----------------------------------------------------------------

        try:

            self.load()

        except Exception as exc:

            return AgentResponse(
                success=False,
                error=str(exc),
                task=task,
                instruction=instruction,
            )

        if (
            self._pipe is None
            or self._tokenizer is None
            or self._model is None
        ):

            return AgentResponse(
                success=False,
                error="GENE model is not available.",
                task=task,
                instruction=instruction,
            )

        # ----------------------------------------------------------------
        # 4. SAFE GENERATION PARAMETERS
        # ----------------------------------------------------------------

        max_new_tokens = max(
            16,
            min(
                int(params.max_new_tokens),
                512,
            ),
        )

        temperature = max(
            0.1,
            min(
                float(params.temperature),
                1.2,
            ),
        )

        top_p = max(
            0.1,
            min(
                float(params.top_p),
                1.0,
            ),
        )

        top_k = max(
            0,
            min(
                int(params.top_k),
                100,
            ),
        )

        repetition_penalty = max(
            1.0,
            min(
                float(params.repetition_penalty),
                1.5,
            ),
        )

        num_return_sequences = max(
            1,
            min(
                int(params.num_return_sequences),
                3,
            ),
        )

        do_sample = bool(
            params.do_sample
        )

        # ----------------------------------------------------------------
        # 5. BUILD QWEN CHAT INPUT
        # ----------------------------------------------------------------

        messages = self._build_messages(
            instruction
        )

        try:

            model_input = (
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        except Exception as exc:

            return AgentResponse(
                success=False,
                error=(
                    "Unable to build Qwen chat prompt: "
                    f"{exc}"
                ),
                task=task,
                instruction=instruction,
            )

        # ----------------------------------------------------------------
        # 6. GENERATION
        # ----------------------------------------------------------------

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "num_return_sequences": num_return_sequences,
            "return_full_text": False,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if do_sample:

            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )

        start = time.perf_counter()

        try:

            # ------------------------------------------------------------
            # inference_mode reduces unnecessary autograd overhead.
            # ------------------------------------------------------------

            with torch.inference_mode():

                outputs = self._pipe(
                    model_input,
                    **generation_kwargs,
                )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start
            )

            return AgentResponse(
                success=False,
                error=f"Generation failed: {exc}",
                elapsed_seconds=elapsed,
                task=task,
                instruction=instruction,
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------------------
        # 7. EXTRACT
        # ----------------------------------------------------------------

        text = self._extract_output(
            outputs
        )

        # ----------------------------------------------------------------
        # 8. CLEAN
        # ----------------------------------------------------------------

        text = self._clean_output(
            text
        )

        if not text:

            return AgentResponse(
                success=False,
                error=(
                    "The model returned an empty response. "
                    "Please try again."
                ),
                elapsed_seconds=elapsed,
                task=task,
                instruction=instruction,
            )

        # ----------------------------------------------------------------
        # 9. RETURN
        # ----------------------------------------------------------------

        return AgentResponse(
            success=True,
            output=text,
            elapsed_seconds=elapsed,
            task=task,
            instruction=instruction,
        )

    # ========================================================================
    # OUTPUT EXTRACTION
    # ========================================================================

    @staticmethod
    def _extract_output(
        outputs,
    ) -> str:

        if not outputs:
            return ""

        first = outputs[0]

        if not isinstance(first, dict):
            return str(first).strip()

        generated = first.get(
            "generated_text",
            "",
        )

        # ---------------------------------------------------------------
        # Chat output
        # ---------------------------------------------------------------

        if isinstance(
            generated,
            list,
        ):

            for message in reversed(
                generated
            ):

                if not isinstance(
                    message,
                    dict,
                ):
                    continue

                if (
                    message.get("role")
                    == "assistant"
                ):

                    content = message.get(
                        "content",
                        "",
                    )

                    if content:
                        return str(
                            content
                        ).strip()

            last = generated[-1]

            if isinstance(
                last,
                dict,
            ):

                return str(
                    last.get(
                        "content",
                        "",
                    )
                ).strip()

            return str(last).strip()

        # ---------------------------------------------------------------
        # Normal generated string
        # ---------------------------------------------------------------

        return str(
            generated
        ).strip()