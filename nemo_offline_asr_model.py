# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import contextlib
import json
import os
import re

import numpy as np
import pytorch_lightning as pl
import riva.client.proto.riva_asr_pb2 as riva_asr_pb2
import torch
import triton_python_backend_utils as pb_utils
from nemo.collections.asr.models import ASRModel
from nemo.utils import logging, model_utils

logging.setLevel(logging.CRITICAL)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def get_nemo_parameters(self, config):
        parameters = config["parameters"]
        # engine_dir = parameters["engine_dir"]["string_value"]
        engine_dir = os.path.dirname(__file__)
        nemo_config_path = os.path.join(engine_dir, "nemo_config.json")
        with open(nemo_config_path) as io:
            nemo_params = json.load(io)
        return nemo_params

    def check_bf16_support(self, device):
        if torch.cuda.get_device_capability(device) >= (8, 0):
            return True
        return False

    def strip_special_tokens(self, text: str):
        """
        assuming all special tokens are of format <token>
        Note that if any label/pred is of format <token>, it will be stripped
        """
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        text = re.sub(r'<[^>]+>', '', text)
        # strip spaces at the beginning and end;
        # this is training data artifact, will be fixed in future (@kpuvvada)
        return text.strip()

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = json.loads(args["model_config"])
        self.nemo_params = self.get_nemo_parameters(self.model_config)

        self.default_language = self.model_config['parameters']["language_code"]['string_value']
        self.lang_codes = list(
            self.model_config['parameters']["language_code"]['string_value'].replace(" ", "").split(",")
        )
        if len(self.lang_codes) > 1:
            if len(self.lang_codes[0]) >= 2:
                self.default_language = self.lang_codes[0]

        model_path = self.nemo_params["nemo_model_path"]
        self.decoding_type = self.nemo_params["nemo_decoder_type"].lower()
        use_fp32 = False
        self.dtype = "float16"
        if use_fp32:
            self.dtype = "float32"
        bf16_support = True
        if torch.cuda.is_available():
            map_location = torch.device("cuda:0")
            accelerator = "gpu"
            device = [0]
            bf16_support = self.check_bf16_support(map_location)
        else:
            map_location = torch.device("cpu")
            device = 1
            accelerator = "cpu"
        if bf16_support and self.dtype == "float16":
            self.dtype = "bfloat16"
        self.dtype = getattr(torch, self.dtype)
        model_cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)
        self.asr_model = imported_class.restore_from(restore_path=model_path, map_location=map_location,)
        trainer = pl.Trainer(devices=device, accelerator=accelerator)
        self.asr_model.set_trainer(trainer)
        self.asr_model.freeze()
        self.asr_model.eval()
        self.asr_model.to(self.dtype)

        if self.decoding_type == "canary":
            if self.default_language not in self.lang_codes and '-' in self.default_language:
                self.default_language = self.default_language.split('-')[0]
            if self.default_language not in self.lang_codes:
                self.default_language = self.lang_codes[0]

        if torch.cuda.is_available() and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            self.autocast = torch.cuda.amp.autocast
        else:

            @contextlib.contextmanager
            def autocast(self, dtype=None, enabled=True):
                yield

    def execute(self, requests):
        with self.autocast(dtype=self.dtype, enabled=True):
            with torch.no_grad():
                responses = []
                all_transcripts = []
                batch_transcript = []
                for i, request in enumerate(requests):
                    processed_audio = pb_utils.get_input_tensor_by_name(request, "audio_signal").as_numpy()
                    processed_length = pb_utils.get_input_tensor_by_name(request, "length").as_numpy()

                    processed_audio = torch.from_numpy(processed_audio).to(self.asr_model.device).to(self.dtype)
                    processed_length = torch.from_numpy(processed_length).to(self.asr_model.device).to(self.dtype)
                    if self.decoding_type == "ctc":
                        encoded, encoded_len, _ = self.asr_model.forward(
                            processed_signal=processed_audio, processed_signal_length=processed_length,
                        )
                        (batch_transcript, _,) = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                            encoded, encoded_len, return_hypotheses=False
                        )
                    elif self.decoding_type == "rnnt":
                        encoded, encoded_len = self.asr_model.forward(
                            processed_signal=processed_audio, processed_signal_length=processed_length,
                        )
                        (batch_transcript, _,) = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                            encoded, encoded_len, return_hypotheses=False
                        )
                    elif self.decoding_type == "canary":
                        batched_serialized_config = np.array(
                            pb_utils.get_input_tensor_by_name(request, "config").as_numpy()
                        )

                        decoder_input_ids = self.get_batched_prompts(
                            processed_audio.shape[0], batched_serialized_config
                        )
                        print(f"{decoder_input_ids=}")
                        (log_probs, encoded_len, enc_states, enc_mask,) = self.asr_model.forward(
                            processed_signal=processed_audio, processed_signal_length=processed_length,
                        )

                        decode_result = self.asr_model.decoding.decode_predictions_tensor(
                            encoder_hidden_states=enc_states,
                            encoder_input_mask=enc_mask,
                            decoder_input_ids=decoder_input_ids,
                            return_hypotheses=False,
                        )

                        # Old NeMo: returns (list[str], None) -> use index 0
                        # New NeMo: returns list[Hypothesis] directly -> extract .text
                        if isinstance(decode_result, tuple):
                            batch_transcript = decode_result[0]
                        else:
                            batch_transcript = decode_result

                        batch_transcript = [
                            (h.text if hasattr(h, 'text') else h)
                            for h in batch_transcript
                        ]
                        batch_transcript = [
                            self.strip_special_tokens(t) for t in batch_transcript
                        ]

                    all_transcripts.append(batch_transcript)

        for batch_transcripts in all_transcripts:
            batch_response = []
            for transcript in batch_transcripts:
                response = riva_asr_pb2.RecognizeResponse()
                result = riva_asr_pb2.SpeechRecognitionResult()
                alternative = riva_asr_pb2.SpeechRecognitionAlternative()
                alternative.transcript = transcript
                result.alternatives.append(alternative)
                response.results.append(result)
                batch_response.append([response.SerializeToString()])
            out = pb_utils.Tensor("logprobs", np.array(batch_response, dtype=np.bytes_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out])
            responses.append(inference_response)
        return responses

    def get_prompt_ids_from_cfg(self, cfg):
        # Support for language codes without country codes
        if cfg['source_language'] not in self.lang_codes and '-' in cfg['source_language']:
            cfg['source_language'] = cfg['source_language'].split('-')[0]

        if cfg['source_language'] not in self.lang_codes:
            logging.warning(f"Invalid language {cfg['source_language']=} specified")
            cfg["source_language"] = self.default_language

        # For transcribe, target == source. For translate, validate target.
        if cfg['task'] == 'translate':
            if cfg.get('target_language') is None:
                cfg['target_language'] = cfg['source_language']
            if cfg['target_language'] not in self.lang_codes and '-' in cfg['target_language']:
                cfg['target_language'] = cfg['target_language'].split('-')[0]
            if cfg['target_language'] not in self.lang_codes:
                logging.error(
                    f"Invalid target language {cfg['target_language']=}, defaulting to {self.default_language}"
                )
                cfg['target_language'] = self.default_language
        else:
            # transcribe: target == source
            cfg['target_language'] = cfg['source_language']

        pnc_token = "<|pnc|>" if cfg.get('pnc', True) else "<|nopnc|>"

        # Indic-Canary (Canary 2 style) prompt format — no <|transcribe|>/<|translate|> token
        prompt = (
            "<|startofcontext|>"
            "<|startoftranscript|>"
            "<|emo:undefined|>"
            f"<|{cfg['source_language']}|>"
            f"<|{cfg['target_language']}|>"
            f"{pnc_token}"
            "<|noitn|>"
            "<|noromanized|>"
            "<|notimestamp|>"
            "<|nodiarize|>"
        )

        return self.asr_model.tokenizer._tokenize_special_prompt(prompt)
        
    def get_batched_prompts(self, batch_size, batched_config):
        batched_prompt_ids = []
        for i in range(batch_size):
            config_len = int.from_bytes(bytes(batched_config[i][:4]), 'little')
            serialized_config = bytes(batched_config[i][4 : 4 + config_len])
            req_obj = riva_asr_pb2.StreamingRecognizeRequest()
            req_obj.ParseFromString(serialized_config)
            req_cfg = {d.name: v for d, v in req_obj.streaming_config.config.ListFields()}
            # TODO Set pnc to True to fix high WER issue https://jirasw.nvidia.com/browse/RIVA-4709
            # req_cfg['pnc'] = req_cfg.get('enable_automatic_punctuation', False)
            req_cfg['pnc'] = True
            req_cfg.update(req_obj.streaming_config.config.custom_configuration)

            if 'source_language' not in req_cfg:
                src_lang_code = req_obj.streaming_config.config.language_code

                if src_lang_code == "" or src_lang_code is None:
                    logging.warning(
                        f"Invalid language {src_lang_code} specified, defaulting to {self.default_language}"
                    )
                    req_cfg["source_language"] = self.default_language
                else:
                    req_cfg["source_language"] = src_lang_code

            if "task" not in req_cfg:
                req_cfg["task"] = "transcribe"

            if req_cfg["task"] not in ["translate", "transcribe"]:
                logging.error(f"Invalid task {req_cfg['task']}  specified")
                req_cfg["task"] = "transcribe"

            if req_cfg["task"] == "transcribe":
                req_cfg["target_language"] = req_cfg["source_language"]

            if "target_language" not in req_cfg:
                req_cfg["target_language"] = self.default_language

            prompt_id = self.get_prompt_ids_from_cfg(req_cfg)
            prompt_id = torch.tensor(prompt_id, dtype=torch.int32)
            batched_prompt_ids.append(prompt_id)

        return torch.nn.utils.rnn.pad_sequence(
            batched_prompt_ids, batch_first=True, padding_value=self.asr_model.tokenizer.pad_id
        ).to(device='cuda:0')

    def finalize(self):
        del self.asr_model
