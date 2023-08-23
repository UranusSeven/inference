.. _models_builtin_starchat_beta:

=============
Starchat-beta
=============

- **Model Name:** starchat-beta
- **Languages:** en
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** HuggingFaceH4/starchat-beta

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name starchat-beta --size-in-billions 16 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec 2 (ggmlv3, 16 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 16
- **Quantizations:** q4_0, q4_1, q5_0, q5_1, q8_0
- **Model ID:** TheBloke/starchat-beta-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name starchat-beta --size-in-billions 16 --model-format ggmlv3 --quantization ${quantization}
