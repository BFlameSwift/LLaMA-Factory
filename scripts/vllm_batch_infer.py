# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from typing import Optional

import gc

import fire
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from transformers import Seq2SeqTrainingArguments

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    n: int = 1,
    is_sampled: bool = False,
    batch_size: int = 65536,
    use_streaming: bool = False,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    check_version("vllm>=0.4.3,<=0.7.3")
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")
    adapter_name_or_path = None
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    if use_streaming:
        data_args.streaming = True
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    inputs, prompts, labels = [], [], []
    images = []

    
    
    if not is_sampled:
        top_k = generating_args.top_k
        top_p = generating_args.top_p
        temperature = generating_args.temperature
    print("*" * 70)
    print("is_sampled:", is_sampled)
    print("top_k:", top_k)
    print("*" * 70)
    
    sampling_params = SamplingParams(
        n=n,
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        # temperature=generating_args.temperature,
        # top_p=generating_args.top_p or 1.0,  # top_p must > 0
        # top_k=generating_args.top_k or -1,  # top_k must > 0
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
        
    llm_engine = LLM(**engine_args)

    # 分批读取数据并进行推理
    # def batch_iterator(data_list, batch_size):
    #     """简单的分批切片迭代器。"""
    #     for i in range(0, len(data_list), batch_size):
    #         yield data_list[i : i + batch_size]
    def batch_iterator(dataset_iter, batch_size):
        if use_streaming:
            # streaming 模式下，dataset_iter 是个可迭代对象
            batch = []
            for item in dataset_iter:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            # 非 streaming, dataset_iter 是个 Dataset 对象，可以获取长度
            length = len(dataset_iter)
            for start_idx in range(0, length, batch_size):
                end_idx = min(start_idx + batch_size, length)
                # 这会返回一个 dataset 对象，只含下标 [start_idx, end_idx) 的那些行
                sub_dataset = dataset_iter.select(range(start_idx, end_idx))
                yield sub_dataset
    # def batch_iterator(ds, batch_size):
    #     length = len(ds)
    #     for start_idx in range(0, length, batch_size):
    #         end_idx = min(start_idx + batch_size, length)
    #         # 这会返回一个 dataset 对象，只含下标 [start_idx, end_idx) 的那些行
    #         sub_dataset = ds.select(range(start_idx, end_idx))
    #         yield sub_dataset
    print(f"Batch size: {batch_size}")
    print(type(dataset_module["train_dataset"]))
    
    total_samples = len(dataset_module["train_dataset"])
    print(f"Total dataset size: {total_samples}")
    processed_samples = 0

    # 打开文件写入（在循环外打开，避免反复打开）
    with open(save_name, "w", encoding="utf-8") as f:
        # 遍历所有 batch
        batch_idx = 0
        print("time to start:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for batch_data in batch_iterator(dataset_module["train_dataset"], batch_size):
            batch_inputs = []
            batch_prompts = []
            batch_labels = []
            batch_images = []
            llm_engine = LLM(**engine_args)
            print("start batch", batch_idx)
            print("data to process:", total_samples - processed_samples)
            start_time = time.time()
            # 准备好输入给 vLLM 的格式
            print(type(batch_data))
            # print(batch_data.keys())
            # breakpoint()
            # print(type(batch_data[0]))
            print("*" * 70)
            for sample in batch_data:
                if sample["images"]:
                    # 处理多模态图像数据
                    multi_modal_data = {
                        "image": template_obj.mm_plugin._regularize_images(
                            sample["images"],
                            image_max_pixels=image_max_pixels,
                            image_min_pixels=image_min_pixels
                        )
                    }
                    batch_images.append(sample["images"])
                else:
                    multi_modal_data = None
                    batch_images.append([])

                # input_ids
                input_ids = sample["input_ids"]
                # prompt
                prompt_text = tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)

                # label
                filtered_label_ids = list(filter(lambda x: x != IGNORE_INDEX, sample["labels"]))
                label_text = tokenizer.decode(filtered_label_ids, skip_special_tokens=skip_special_tokens)

                # 整理成 vLLM 推理需要的形式
                batch_inputs.append({"prompt_token_ids": input_ids, "multi_modal_data": multi_modal_data})
                batch_prompts.append(prompt_text)
                batch_labels.append(label_text)
            end_time = time.time()
            print("*" * 70)
            print(f"Data loading time: {end_time - start_time:.2f} seconds")
            print(f"Total dataset size: {len(batch_inputs)}")
            print("per sample time:", (end_time - start_time) / len(batch_inputs))
            print(f"Batch size: {len(batch_inputs)}")
            print("*" * 70)

            before_inference_time = time.time()
            # 调用 vLLM 进行推理
            batch_results = llm_engine.generate(
                batch_inputs,
                sampling_params,
                lora_request=lora_request
            )
            end_inference_time = time.time()
            print(f"Batch inference time: {end_inference_time - before_inference_time:.2f} seconds")
            print(f"Total dataset size: {len(batch_results)}")
            print("per sample time:", (end_inference_time - before_inference_time) / len(batch_results))
            print(f"Batch size: {len(batch_results)}")

            # 将结果写入文件
            for text, result, label, image in zip(batch_prompts, batch_results, batch_labels, batch_images):
                # 该输入可能有多个生成 (n>1)，取所有结果
                predictions = [output.text for output in result.outputs]
                f.write(json.dumps({
                    "prompt": text,
                    "preds": predictions,
                    "label": label,
                    "images": image
                }, ensure_ascii=False) + "\n")

            processed_samples += len(batch_data)
            print(f"Processed {processed_samples}/{total_samples} samples...")
            print("time to finish:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + (total_samples - processed_samples) * (end_inference_time - before_inference_time) / len(batch_results))))
            
            del batch_data, batch_inputs, batch_prompts, batch_labels, batch_results, batch_images
            gc.collect()

    print("*" * 70)
    print(f"{processed_samples} generated results have been saved to {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)
