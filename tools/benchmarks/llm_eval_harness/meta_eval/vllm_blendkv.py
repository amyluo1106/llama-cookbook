# lm_eval --model vllm_cacheblend   --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=8192,add_bos_token=True,seed=42,enforce_eager=True --tasks meta_mmlu_pro_instruct --batch_size auto --output_path eval_results --include_path /workplace/amyluo/llama-cookbook/tools/benchmarks/llm_eval_harness/meta_eval/work_dir --seed 42  --log_samples
# lm_eval --model vllm_cacheblend   --model_args pretrained=/home/amyluo/Llama,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=8192,add_bos_token=True,seed=42,enforce_eager=True --tasks meta_mmlu_pro_instruct --batch_size auto --output_path eval_results --include_path /workplace/amyluo/llama-cookbook/tools/benchmarks/llm_eval_harness/meta_eval/work_dir --seed 42  --log_samples
# lm_eval --model vllm_cacheblend --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1,max_model_len=8192,add_bos_token=True,seed=42,enable_prefix_caching=False --tasks meta_mmlu_pro_instruct --batch_size 2 --output_path eval_results --include_path ./work_dir --seed 42  --log_samples
import os
import copy
from typing import Dict, List, Literal, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import Collator
from lm_eval.utils import eval_logger

@register_model("vllm_cacheblend")
class VLLMCacheBlend(VLLM):
    def __init__(
        self,
        pretrained: str,
        blend_separator: str = " # # ",
        randomize_shots: bool = True,
        chunk_size: int = 256,
        recomp_ratio: str = "0.15",
        use_layerwise: bool = True,
        use_local_cpu: bool = True,
        max_local_cpu_size: int = 150,
        *args,
        **kwargs,
    ):
        # Set environment variables for cache blending before initializing the model
        os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
        os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_separator
        os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
        os.environ["LMCACHE_USE_LAYERWISE"] = "True"
        os.environ["LMCACHE_BLEND_RECOMPUTE_RATIO"] = recomp_ratio
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(max_local_cpu_size)
        
        # Store these for our own methods
        self.blend_separator = blend_separator
        self.randomize_shots = randomize_shots
        
        from vllm.config import KVTransferConfig
        
        # Create the lmcache connector
        try:
            lmcache_connector = "LMCacheConnectorV1"
            
            # Create KV transfer config
            ktc = KVTransferConfig(
                kv_connector=lmcache_connector,
                kv_role="kv_both",
            )
            
            kwargs['kv_transfer_config'] = ktc
            kwargs['enable_prefix_caching'] = False
            
            eval_logger.info("Set up KVTransferConfig with LmcacheKVConnector")
        except Exception as e:
            eval_logger.error(f"Failed to set up LmcacheKVConnector: {e}")
            eval_logger.warning("Continuing without KV transfer config")
        
        # Initialize the parent class
        super().__init__(pretrained=pretrained, *args, **kwargs)
        
        # Log the configuration
        eval_logger.info(f"Initialized VLLMCacheBlend with separator: {blend_separator}")
        eval_logger.info(f"Cache blend settings: chunk_size={chunk_size}, recomp_ratio={recomp_ratio}")
        eval_logger.info(f"Layerwise: {use_layerwise}, Local CPU: {use_local_cpu} (max size: {max_local_cpu_size})")
    
    def _insert_blend_separators_for_mmlu(self, prompt_text):
        """
        Insert blend separators for MMLU-style prompts, separating:
        1. The instruction/system prompt
        2. Each few-shot example
        3. The test question
        """
        # Debug print to see what prompts look like
        eval_logger.debug(f"Processing MMLU prompt of length: {len(prompt_text)}")
        
        # First check if this is a few-shot prompt or not
        has_examples = "<|start_header_id|>user<|end_header_id|>" in prompt_text
        
        # If it's not a few-shot prompt, just return the regular prompt
        if not has_examples:
            return prompt_text
        
        # Split on user markers to get individual examples
        examples = prompt_text.split("<|start_header_id|>user<|end_header_id|>")
        
        # First element is usually empty or contains initial instructions
        examples = ["<|start_header_id|>user<|end_header_id|>" + ex for ex in examples if ex.strip()]
        
        # Take the last example as the test question
        final_test_question = examples.pop() if examples else None
        
        # Randomize the order of few-shot examples if requested
        if self.randomize_shots and len(examples) > 1:
            import random
            random.shuffle(examples)
        
        # Start with the initial instruction prompt
        initial_prompt = "Choose the best multiple choice answer.\n"
        final_prompt = initial_prompt
        
        # Add each example with a blend separator
        for example in examples:
            final_prompt = final_prompt + self.blend_separator + example
        
        # Add the test question with a blend separator
        if final_test_question:
            final_prompt = final_prompt + self.blend_separator + final_test_question
        
        return final_prompt
    
    def _insert_blend_separators(self, prompt_text):
        """
        Generic blend separator insertion strategy.
        This can be expanded for different prompt types.
        """
        # Check if this looks like an MMLU-style prompt
        if "<|start_header_id|>user<|end_header_id|>" in prompt_text:
            return self._insert_blend_separators_for_mmlu(prompt_text)
        
        # For other prompt types, we can use paragraph breaks as a heuristic
        parts = prompt_text.split('\n\n')  # Split on paragraph breaks
        
        if len(parts) <= 1:  # No clear separation points
            return prompt_text
        
        # Join with blend separators
        return self.blend_separator.join(parts)
    
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Override generate_until to insert blend separators in the prompts.
        """
        res = []

        # batch tokenize contexts with blend separators
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        
        # Insert blend separators into the context strings
        context_with_separators = [self._insert_blend_separators(ctx) for ctx in context]
        
        # Debug log to see the changes
        for i in range(min(2, len(context))):
            eval_logger.debug(f"Original prompt: {context[i][:100]}...")
            eval_logger.debug(f"With separators: {context_with_separators[i][:100]}...")
            eval_logger.debug(f"Separator count: {context_with_separators[i].count(self.blend_separator)}")
        
        # Use the parent class's tokenization method on our modified contexts
        context_encoding = super().tok_encode(context_with_separators, add_special_tokens=self.add_bos_token)
        
        # Create the batched requests
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            return -len(_requests[0][1]), _requests[0][0]

        # Group requests by their generation_kwargs
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm),
            desc="Running generate_until requests with cache blending",
        )
        
        # For each different set of kwargs, we execute all requests, by batch
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # We assume all gen kwargs in the batch are the same
            gen_kwargs = all_gen_kwargs[0]
            # Unpack our keyword arguments
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # Edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # Add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # Set max length for inputs, minus room for generation
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # Perform batched generation with our token IDs that include blend separators
            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            # Cache generations
            for output, context in zip(cont, context):
                generated_text = output.outputs[0].text
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # Reorder results back to original form
        return re_ords.get_original(res)
