from collections import defaultdict
import contextlib
import os


# reduce memory: 在导入 torch 前开启可扩展 CUDA 段，降低 allocator 碎片造成的显存峰值。
def _ensure_cuda_alloc_conf(option):
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    key = option.split(":", 1)[0]
    entries = [entry.strip() for entry in current.split(",") if entry.strip()]
    entries = [entry for entry in entries if entry.split(":", 1)[0] != key]
    entries.append(option)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(entries)


_ensure_cuda_alloc_conf("expandable_segments:True")

import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.fsdp_utils import register_optimizer_offload_hooks

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block

SD3_TRANSFORMER_LAYER_CLASSES = (JointTransformerBlock,)
SD3_TEXT_ENCODER_LAYER_CLASSES = (CLIPEncoderLayer, T5Block)


# reduce memory: FSDP2 只在多进程训练时启用，单卡保持原始路径避免额外开销。
def should_use_fsdp2(config, accelerator):
    if accelerator.num_processes <= 1:
        return False
    if hasattr(config, "fsdp2"):
        return bool(config.fsdp2)
    return True


# reduce memory: 采样轨迹可常驻 CPU，训练当前 batch 前再搬回 GPU。
def maybe_tensor_to_cpu(value, enabled):
    if enabled and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


# reduce memory: CPU 上的 replay buffer 在训练前按需搬到当前 GPU。
def tensor_tree_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: tensor_tree_to_device(item, device) for key, item in value.items()}
    return value


# reduce memory: CPU replay buffer 使用 CPU mask/index，避免为了筛选整批搬回 GPU。
def index_tensor(value, index):
    if isinstance(value, torch.Tensor) and value.device.type == "cpu" and isinstance(index, torch.Tensor):
        return value[index.cpu()]
    return value[index]


# reduce memory: 采样缓存搬到 CPU 后主动释放 PyTorch CUDA cache。
def empty_cuda_cache_if_needed(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize_dtensor_state_dict(state_dict):
    materialized = {}
    for key, value in state_dict.items():
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        materialized[key] = value
    return materialized


# reduce memory: 按 transformer/text encoder block 做 FSDP2 切分，并可叠加 activation checkpointing。
def apply_fsdp2_sharding(
    module,
    layer_classes,
    mesh,
    mp_policy,
    offload_policy,
    activation_checkpointing=False,
):
    modules_to_shard = [submodule for submodule in module.modules() if isinstance(submodule, layer_classes)]

    if activation_checkpointing and modules_to_shard:
        # reduce memory: 对大 block 启用重计算，用额外计算换取更低激活显存。
        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda submodule: isinstance(submodule, layer_classes),
        )

    for submodule in modules_to_shard:
        fully_shard(
            submodule,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )

    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )
    return module


def prepare_sd3_with_fsdp2(pipeline, config, accelerator, inference_dtype):
    mesh = init_device_mesh(accelerator.device.type, (accelerator.num_processes,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=inference_dtype,
        reduce_dtype=inference_dtype,
        output_dtype=None,
    )
    offload_policy = (
        # reduce memory: FSDP2 CPU offload 把空闲参数分片放到 CPU，降低 GPU 常驻参数显存。
        CPUOffloadPolicy(pin_memory=True)
        if bool(getattr(config, "fsdp2_cpu_offload", False))
        else OffloadPolicy()
    )

    if bool(getattr(config, "fsdp2_shard_text_encoders", True)):
        # reduce memory: 文本编码器同样按层切分，避免 T5/CLIP 在每张卡完整常驻。
        pipeline.text_encoder = apply_fsdp2_sharding(
            pipeline.text_encoder,
            SD3_TEXT_ENCODER_LAYER_CLASSES,
            mesh,
            mp_policy,
            offload_policy,
            activation_checkpointing=False,
        )
        pipeline.text_encoder_2 = apply_fsdp2_sharding(
            pipeline.text_encoder_2,
            SD3_TEXT_ENCODER_LAYER_CLASSES,
            mesh,
            mp_policy,
            offload_policy,
            activation_checkpointing=False,
        )
        pipeline.text_encoder_3 = apply_fsdp2_sharding(
            pipeline.text_encoder_3,
            SD3_TEXT_ENCODER_LAYER_CLASSES,
            mesh,
            mp_policy,
            offload_policy,
            activation_checkpointing=False,
        )
    else:
        pipeline.text_encoder.to(accelerator.device)
        pipeline.text_encoder_2.to(accelerator.device)
        pipeline.text_encoder_3.to(accelerator.device)

    # reduce memory: SD3 transformer 主体按 JointTransformerBlock 切分，是主要显存节省来源。
    pipeline.transformer = apply_fsdp2_sharding(
        pipeline.transformer,
        SD3_TRANSFORMER_LAYER_CLASSES,
        mesh,
        mp_policy,
        offload_policy,
        activation_checkpointing=bool(
            getattr(
                config,
                "fsdp2_activation_checkpointing",
                getattr(config, "activation_checkpointing", True),
            )
        ),
    )

    return pipeline.transformer


def disable_adapter(transformer):
    module = transformer.module if hasattr(transformer, "module") else transformer
    if hasattr(module, "disable_adapter"):
        return module.disable_adapter()
    return contextlib.nullcontext()


@contextlib.contextmanager
def fsdp2_accumulate(accelerator, transformer, use_fsdp2):
    with accelerator.accumulate(transformer):
        if use_fsdp2:
            # reduce memory: accumulation 内关闭不必要的梯度同步，减少 FSDP2 反传通信/缓冲峰值。
            transformer.set_requires_gradient_sync(accelerator.sync_gradients)
            transformer.set_is_last_backward(accelerator.sync_gradients)
        try:
            yield
        finally:
            if use_fsdp2:
                transformer.set_requires_gradient_sync(True)
                transformer.set_is_last_backward(True)


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    # reduce memory: CPU offload 下梯度可能在 CPU，规约标量临时放到 CUDA 以兼容 NCCL。
    reduce_device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.is_available() and dist.is_initialized() and torch.cuda.is_available()
        else device
    )
    if norm_type == float("inf"):
        local_norm = torch.stack(
            [
                (p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad)
                .detach()
                .abs()
                .max()
                .to(device)
                for p in parameters
            ]
        ).max()
        local_norm = local_norm.to(reduce_device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_norm, op=dist.ReduceOp.MAX)
        total_norm = local_norm
    else:
        norm_type = float(norm_type)
        local_norm = torch.zeros((), device=device, dtype=torch.float32)
        for p in parameters:
            grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            local_norm += grad.detach().float().norm(norm_type).pow(norm_type)
        local_norm = local_norm.to(reduce_device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_norm, op=dist.ReduceOp.SUM)
        total_norm = local_norm.pow(1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        clip_coef = clip_coef.item()
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm


def has_fsdp2_modules(model):
    return any(isinstance(module, FSDPModule) for module in model.modules())

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(
    transformer,
    pipeline,
    sample,
    j,
    embeds,
    pooled_embeds,
    config,
    negative_embeds=None,
    negative_pooled_embeds=None,
):
    if config.train.cfg:
        if bool(getattr(config.train, "cfg_sequential", False)):
            if negative_embeds is None or negative_pooled_embeds is None:
                raise ValueError("Sequential CFG requires negative prompt embeddings.")
            # reduce memory: 训练 CFG 拆成 uncond/text 两次前向，避免 hidden_states 临时拼成双 batch。
            noise_pred_uncond = transformer(
                hidden_states=sample["latents"][:, j],
                timestep=sample["timesteps"][:, j],
                encoder_hidden_states=negative_embeds,
                pooled_projections=negative_pooled_embeds,
                return_dict=False,
            )[0]
            noise_pred_text = transformer(
                hidden_states=sample["latents"][:, j],
                timestep=sample["timesteps"][:, j],
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        else:
            noise_pred = transformer(
                hidden_states=torch.cat([sample["latents"][:, j]] * 2),
                timestep=torch.cat([sample["timesteps"][:, j]] * 2),
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                    # reduce memory: eval 采样同样复用 sequential CFG，避免测试阶段临时双倍 batch。
                    sequential_guidance=bool(
                        getattr(config.sample, "cfg_sequential", getattr(config.train, "cfg_sequential", False))
                    ),
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)

    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    unwrapped_transformer = unwrap_model(transformer, accelerator)
    if has_fsdp2_modules(unwrapped_transformer):
        if config.use_lora:
            state_dict = get_peft_model_state_dict(unwrapped_transformer)
            state_dict = materialize_dtensor_state_dict(state_dict)
        else:
            # reduce memory: 保存 FSDP2 全量权重时 offload 到 CPU，避免 rank0 GPU 聚合爆显存。
            state_dict = get_model_state_dict(
                unwrapped_transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        if accelerator.is_main_process:
            unwrapped_transformer.save_pretrained(save_root_lora, state_dict=state_dict)
        del state_dict
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    elif accelerator.is_main_process:
        unwrapped_transformer.save_pretrained(save_root_lora)

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config
    # reduce memory: 3090 不支持 bf16 时自动退到 fp16，保持半精度省显存路径可用。
    bf16_fallback_to_fp16 = (
        config.mixed_precision == "bf16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    )
    if bf16_fallback_to_fp16:
        config.mixed_precision = "fp16"

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    use_fsdp2 = should_use_fsdp2(config, accelerator)
    if bf16_fallback_to_fp16:
        accelerator.print("bf16 is not supported on this GPU; falling back to fp16.")
    if accelerator.num_processes <= 1:
        accelerator.print("FSDP2 requires --num_processes > 1; using the original single-process path.")
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(dtype=inference_dtype)
    pipeline.text_encoder_2.to(dtype=inference_dtype)
    pipeline.text_encoder_3.to(dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    if use_fsdp2:
        # reduce memory: 多卡训练时在模型进入 optimizer 前完成 FSDP2 切分，降低每卡参数常驻显存。
        transformer = prepare_sd3_with_fsdp2(pipeline, config, accelerator, inference_dtype)
        accelerator.print("Using PyTorch FSDP2 for SD3 transformer and text encoders.")
    else:
        pipeline.transformer.to(accelerator.device)
        transformer = pipeline.transformer

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = None
    if config.train.ema:
        # This ema setting affects the previous 20 x 8 = 160 steps on average.
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    if bool(getattr(config, "fsdp_optimizer_offload", False)):
        # reduce memory: optimizer state 在 step 后卸载到 CPU，降低 Adam 状态常驻显存。
        register_optimizer_offload_hooks(optimizer)

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=0,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    if accelerator.state.deepspeed_plugin is not None:
            # 显式告诉 DeepSpeed 每张 GPU 的微批次大小是多少
            accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size

    

    if use_fsdp2:
        # reduce memory: FSDP2 模型已手动 fully_shard，这里只 prepare optimizer/dataloader，避免二次包裹。
        optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            optimizer,
            train_dataloader,
            test_dataloader,
        )
    else:
        (
            transformer,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
            optimizer,
            train_dataloader,
            test_dataloader,
        ) = accelerator.prepare(
            transformer,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
            optimizer,
            train_dataloader,
            test_dataloader,
        )

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=128,
        device=accelerator.device,
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    # reduce memory: 控制采样得到的 prompt embedding、latent、log_prob 是否放到 CPU replay buffer。
    sample_cpu_offload = bool(getattr(config, "sample_cpu_offload", False))

    while epoch < config.num_epochs:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if config.eval_freq > 0 and epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if config.save_freq > 0 and epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        # reduce memory: 采样 CFG 拆成两次前向，避免每个 denoise step 临时双倍 batch。
                        sequential_guidance=bool(
                            getattr(config.sample, "cfg_sequential", getattr(config.train, "cfg_sequential", False))
                        ),
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)
            # reduce memory: reward 计算只需要图像内容，先把图像转 CPU，释放采样阶段 GPU 占用。
            images_for_reward = images.detach().cpu() if sample_cpu_offload else images

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images_for_reward, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    # reduce memory: replay buffer 中的大张量放 CPU，避免采样后整条轨迹常驻 GPU。
                    "prompt_ids": maybe_tensor_to_cpu(prompt_ids, sample_cpu_offload),
                    "prompt_embeds": maybe_tensor_to_cpu(prompt_embeds, sample_cpu_offload),
                    "pooled_prompt_embeds": maybe_tensor_to_cpu(pooled_prompt_embeds, sample_cpu_offload),
                    "timesteps": maybe_tensor_to_cpu(timesteps, sample_cpu_offload),
                    "latents": maybe_tensor_to_cpu(
                        latents[:, :-1], sample_cpu_offload
                    ),  # each entry is the latent before timestep t
                    "next_latents": maybe_tensor_to_cpu(
                        latents[:, 1:], sample_cpu_offload
                    ),  # each entry is the latent after timestep t
                    "log_probs": maybe_tensor_to_cpu(log_probs, sample_cpu_offload),
                    "rewards": rewards,
                }
            )
            if sample_cpu_offload:
                # reduce memory: 采样结果已转 CPU 后，立即删除 GPU 临时变量并清 cache。
                images = images_for_reward
                del latents, log_probs, timesteps, prompt_embeds, pooled_prompt_embeds, prompt_ids
                empty_cuda_cache_if_needed(True)

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            # reduce memory: 只在跨卡 gather prompt id 时临时搬回 GPU，随后立即释放。
            prompt_ids_for_gather = samples["prompt_ids"].to(accelerator.device, non_blocking=True)
            prompt_ids = accelerator.gather(prompt_ids_for_gather).cpu().numpy()
            del prompt_ids_for_gather
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        local_advantages = advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
        # reduce memory: advantages 跟随 replay buffer 放 CPU，训练当前 batch 时再搬回 GPU。
        samples["advantages"] = local_advantages.cpu() if sample_cpu_offload else local_advantages.to(accelerator.device)
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        # reduce memory: CPU replay buffer 在 CPU 上完成 mask 筛选，避免整批迁回 GPU。
        samples = {k: index_tensor(v, mask) for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=samples["timesteps"].device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if sample_cpu_offload:
                    # reduce memory: 只把当前训练 micro-batch 搬到 GPU，不让全 epoch 样本常驻显存。
                    sample = tensor_tree_to_device(sample, accelerator.device)
                if config.train.cfg:
                    if bool(getattr(config.train, "cfg_sequential", False)):
                        # reduce memory: sequential CFG 下不拼接 negative/positive embedding，避免双 batch 激活。
                        embeds = sample["prompt_embeds"]
                        pooled_embeds = sample["pooled_prompt_embeds"]
                        negative_embeds = train_neg_prompt_embeds[:len(sample["prompt_embeds"])]
                        negative_pooled_embeds = train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])]
                    else:
                        # concat negative prompts to sample prompts to avoid two forward passes
                        embeds = torch.cat(
                            [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                        )
                        pooled_embeds = torch.cat(
                            [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                        )
                        negative_embeds = None
                        negative_pooled_embeds = None
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]
                    negative_embeds = None
                    negative_pooled_embeds = None

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with fsdp2_accumulate(accelerator, transformer, use_fsdp2):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                transformer,
                                pipeline,
                                sample,
                                j,
                                embeds,
                                pooled_embeds,
                                config,
                                negative_embeds,
                                negative_pooled_embeds,
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with disable_adapter(transformer):
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                            transformer,
                                            pipeline,
                                            sample,
                                            j,
                                            embeds,
                                            pooled_embeds,
                                            config,
                                            negative_embeds,
                                            negative_pooled_embeds,
                                        )

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            if use_fsdp2:
                                accelerator.unscale_gradients(optimizer)
                                fsdp2_clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                            else:
                                accelerator.clip_grad_norm_(
                                    transformer.parameters(), config.train.max_grad_norm
                                )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)
