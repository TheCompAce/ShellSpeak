from enum import Enum
import json
import sqlite3
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

transformers.logging.set_verbosity_error()

from modules.responseCache import ResponseCache

executor = ThreadPoolExecutor()



class ModelTypes(Enum):
    OpenAI = "OpenAI"
    OpenAI4 = "OpenAI4"
    Mistral = "Mistral"
    StableBeluga7B = "StableBeluga7B"
    Zephyr7bAlpha = "Zephyr7bAlpha"
    Zephyr7bBeta = "Zephyr7bBeta"
    Falcon7BInst = "Falcon7BInst"

class LLM:
    def __init__(self, model_type, use_cache=False, cache_file=None):
        self.ClearModel(model_type)
        self.use_cache = use_cache
        if use_cache:
            self.cache = ResponseCache(cache_file)

    def ClearModel(self, model_type):
        self.model = ModelTypes(model_type)
        self.modelObj = None
        self.tokenizerObj = None
        self.pipeObj = None

    def SetupModel(self):
        if self.model == ModelTypes.Mistral:
            return self._setup_mistral()
        elif self.model == ModelTypes.StableBeluga7B:
            return self._setup_beluga_7b()
        elif self.model == ModelTypes.Zephyr7bAlpha:
            return self._setup_zephyr_7b()
        elif self.model == ModelTypes.Zephyr7bBeta:
            return self._setup_zephyr_7bB()

    async def async_ask(llm, system_prompt, user_prompt, model_type=None):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, llm.ask, system_prompt, user_prompt, model_type)
        return response

    def ask(self, system_prompt, user_prompt, model_type=None):
        if self.use_cache:
            cached_response = self.cache.get(system_prompt, user_prompt)
            if cached_response:
                return cached_response
        response = self._ask(system_prompt, user_prompt, model_type)
        if self.use_cache:
            self.cache.set(system_prompt, user_prompt, response)
        return response

    def _ask(self, system_prompt, user_prompt, model_type = None):
        
        if model_type is None:
            model_type = self.model
        elif model_type is not self.model:
                self.ClearModel(model_type)
        if model_type == ModelTypes.OpenAI:
            return self._ask_openai(system_prompt, user_prompt)
        elif model_type == ModelTypes.OpenAI4:
            return self._ask_openai(system_prompt, user_prompt, model="gpt-4-1106-preview", max_tokens=8190)
        elif model_type == ModelTypes.Mistral:
            return self._ask_mistral(system_prompt, user_prompt)
        elif model_type == ModelTypes.StableBeluga7B:
            return self._ask_stable_beluga_7b(system_prompt, user_prompt)
        elif model_type == ModelTypes.Zephyr7bAlpha:
            return self._ask_zephyr_7b(system_prompt, user_prompt)
        elif model_type == ModelTypes.Zephyr7bBeta:
            return self._ask_zephyr_7bB(system_prompt, user_prompt)
        elif model_type == ModelTypes.Falcon7BInst:
            return self._ask_falcon_7b_instruct(system_prompt, user_prompt)

    def _ask_openai(self, system_prompt, user_prompt, model = "gpt-3.5-turbo-1106", max_tokens=4096):
        # Placeholder for actual OpenAI API request
        # Uncomment and complete the following code in your local environment
        api_key = os.environ.get("OPENAI_API_KEY", "your-default-openai-api-key-here")
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data ={
            "model" : model,
            "messages" :  [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        tries = 2
        response = None
        is_error = False
        while tries > 0:
            try:
                response = requests.post(api_url, headers=headers, json=data, timeout=(2, 60))
                tries = 0
            except requests.Timeout:
                tries -= 1
                if tries == 0:
                    is_error = True
                    response = "Timeout"
            except requests.exceptions.RequestException as e:
                is_error = True
                response = e.response
                tries -= 1

        if response:
            if not is_error:
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]
                else:
                    print(f"response = {response.__dict__}")
                    return f"Error (_ask_openai): {response.status_code} - {response.json()}"
            else:
                print(f"response = {response}")
                return f"Error (_ask_openai): {response}"
        else:
            print(f"response = {response.__dict__}")
            return f"Error (_ask_openai): No Reponse."

    def _ask_mistral(self, system_prompt, user_prompt):
        if self.tokenizerObj is None or self.modelObj is None:
            self._setup_mistral()
        prompt = f"<s>[INST] {system_prompt} {user_prompt} [/INST]"
        inputs = self.tokenizerObj(prompt, return_tensors="pt")
        outputs = self.modelObj.generate(**inputs, max_new_tokens=4096)
        decoded = self.tokenizerObj.decode(outputs[0], skip_special_tokens=True)
        return decoded
    
    def _setup_mistral(self):
        if self.modelObj is None or self.tokenizerObj is None:
            self.modelObj = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
            self.tokenizerObj = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    def _setup_beluga_7b(self):
        if self.modelObj is None or self.tokenizerObj is None:
            self.modelObj = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga-7B", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            self.tokenizerObj = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-7B", use_fast=False)


    def _ask_stable_beluga_7b(self, system_prompt, user_prompt):
        if self.tokenizerObj is None or self.modelObj is None:
            self._setup_beluga_7b()
        prompt = f"### System: {system_prompt}\\n\\n### User: {user_prompt}\\n\\n### Assistant:\\n"
        inputs = self.tokenizerObj(prompt, return_tensors="pt").to("cuda")
        output = self.modelObj.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=4096)
        return self.tokenizerObj.decode(output[0], skip_special_tokens=True)

    def _ask_zephyr_7b(self, system_prompt, user_prompt):
        if self.pipeObj is None:
            self._setup_zephyr_7b()
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.pipeObj.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeObj(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

    def _setup_zephyr_7b(self):
        if self.pipeObj is None:
            self.pipeObj= pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

    def _ask_zephyr_7bB(self, system_prompt, user_prompt):
        if self.pipeObj is None:
            self._setup_zephyr_7bB()
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.pipeObj.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeObj(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

    def _setup_zephyr_7bB(self):
        if self.pipeObj is None:
            self.pipeObj= pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

    def _setup_falcon_7b_instruct(self):
        if self.modelObj is None or self.tokenizerObj is None:
            self.modelObj = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct").to("cuda")
            self.tokenizerObj = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")



    def _ask_falcon_7b_instruct(self, system_prompt, user_prompt):
        if self.tokenizerObj is None or self.modelObj is None:
            self._setup_falcon_7b_instruct()
        device = 0  # This assumes that you have at least one GPU and it's device 0
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.modelObj,
            tokenizer=self.tokenizerObj,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=device,  # Specify the device here
        )
        sequences = pipeline(
            f"{system_prompt}\n{user_prompt}",
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizerObj.eos_token_id,
        )
        return sequences[0]['generated_text']



    def __repr__(self):
        return f"LLMBase(model={self.model})"
