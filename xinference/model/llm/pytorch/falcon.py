# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

import torch

from xinference.constants import XINFERENCE_CACHE_DIR
from xinference.types import ChatCompletionMessage

from .core import PytorchChatModel, PytorchModelConfig

if TYPE_CHECKING:
    from ... import ModelSpec


class FalconPytorch(PytorchChatModel):
    _system_prompt = ""
    _sep = "\n"
    _user_name = "USER"
    _assistant_name = "ASSISTANT"

    def __init__(
        self,
        model_uid: str,
        model_spec: "ModelSpec",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_spec,
            model_path,
            system_prompt=self._system_prompt,
            sep=self._sep,
            user_name=self._user_name,
            assistant_name=self._assistant_name,
            stop="\nUser",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
            pytorch_model_config=pytorch_model_config,
        )
        self._use_fast_tokenizer = False

    def _load_model(self, kwargs: dict):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.generation.utils import GenerationConfig
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=True,
            revision=kwargs["revision"],
            cache_dir=XINFERENCE_CACHE_DIR,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            cache_dir=XINFERENCE_CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs,
        )
        model.generation_config = GenerationConfig.from_pretrained(self._model_path)
        return model, tokenizer

    def _to_prompt(
        self,
        prompt: str,
        system_prompt: str,
        chat_history: List[ChatCompletionMessage],
    ):
        ret = self._system_prompt
        for i, message in enumerate(chat_history):
            role, content = message["role"], message["content"]
            if content:
                ret += role + ": " + content.replace("\r\n", "\n").replace("\n\n", "\n")
                ret += "\n\n"
            else:
                ret += role + ":"
        return ret
