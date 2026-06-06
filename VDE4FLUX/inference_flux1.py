import argparse
import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__} does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__} does not support custom sigma schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def predict_alpha_beta(anchor_timesteps, anchor_alpha, anchor_beta, current_t):
    if not anchor_timesteps:
        raise ValueError("VDE needs at least one anchor step before prediction.")

    if len(anchor_timesteps) < 2:
        return anchor_alpha[0], anchor_beta[0]

    t1, t2 = anchor_timesteps[-2], anchor_timesteps[-1]
    alpha1, alpha2 = anchor_alpha[-2], anchor_alpha[-1]
    beta1, beta2 = anchor_beta[-2], anchor_beta[-1]

    dt = t2 - t1
    if bool((dt == 0).item() if torch.is_tensor(dt) else dt == 0):
        return alpha2, beta2

    weight = (current_t - t1) / dt
    if torch.is_tensor(weight):
        weight = weight.to(dtype=alpha1.dtype, device=alpha1.device)
    else:
        weight = torch.tensor(weight, dtype=alpha1.dtype, device=alpha1.device)

    return torch.lerp(alpha1, alpha2, weight), torch.lerp(beta1, beta2, weight)


def decompose_velocity(latents, noise_pred):
    lat_norm = torch.norm(latents)
    lat_unit = latents / (lat_norm + 1e-6)

    vel_tangential_norm = torch.sum(noise_pred * lat_unit)
    vel_tangential = vel_tangential_norm * lat_unit
    vel_normal = noise_pred - vel_tangential
    vel_normal_norm = torch.norm(vel_normal)
    vel_normal_unit = vel_normal / (vel_normal_norm + 1e-6)

    alpha = vel_tangential_norm / (lat_norm + 1e-6)
    beta = vel_normal_norm / (lat_norm + 1e-6)
    return alpha, beta, vel_normal_unit


@torch.no_grad()
def vde_flux_call(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    stable_step: int = 13,
    interval: int = 3,
    save_latents_dir: Optional[str] = None,
):
    if stable_step < 1:
        raise ValueError("`stable_step` must be >= 1.")
    if interval < 1:
        raise ValueError("`interval` must be >= 1.")
    if stable_step >= num_inference_steps:
        raise ValueError("`stable_step` must be smaller than `num_inference_steps`.")

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None

    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    anchor_timesteps = []
    anchor_alpha = []
    anchor_beta = []
    vel_normal_unit = None
    real_infer_steps = 0


    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            should_run_transformer = (i <= stable_step) or (i == len(timesteps) - 1) or ((i - stable_step) % interval == 0)

            if should_run_transformer:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                real_infer_steps += 1

                if i >= stable_step - 1:
                    alpha, beta, vel_normal_unit = decompose_velocity(latents, noise_pred)
                    anchor_timesteps.append(t)
                    anchor_alpha.append(alpha)
                    anchor_beta.append(beta)
            else:
                alpha_pred, beta_pred = predict_alpha_beta(anchor_timesteps, anchor_alpha, anchor_beta, t)
                lat_norm = torch.norm(latents)
                
                noise_pred = alpha_pred * latents + beta_pred * lat_norm * vel_normal_unit


            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    self._vde_real_infer_steps = real_infer_steps

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)
    return FluxPipelineOutput(images=image)


def dtype_from_name(name):
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in choices:
        raise ValueError(f"Unsupported dtype: {name}")
    return choices[key]


def get_args():
    parser = argparse.ArgumentParser(description="FLUX.1 inference with VDE acceleration.")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", type=str, default="A black colored car with a large VDE logo printed on the side.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default=os.environ.get("FLUX_DEVICE_MAP", "balanced"))
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--vde", action="store_true", help="Enable VDE training-free acceleration.")
    parser.add_argument("--stable-step", type=int, default=10)
    parser.add_argument("--interval", type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    dtype = dtype_from_name(args.dtype)

    if args.vde:
        FluxPipeline.__call__ = vde_flux_call
        tag = f"vde_s{args.stable_step}i{args.interval}"
    else:
        tag = "original"

    load_kwargs = {"torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    pipe = FluxPipeline.from_pretrained(args.model, **load_kwargs)
    if not args.device_map:
        pipe = pipe.to(args.device)
    generator = torch.Generator("cpu").manual_seed(args.seed)

    call_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    if args.vde:
        call_kwargs.update(
            {
                "stable_step": args.stable_step,
                "interval": args.interval,
            }
        )
    start = time.perf_counter()
    result = pipe(args.prompt, **call_kwargs)
    elapsed = time.perf_counter() - start

    output = args.output or f"output_flux1_{tag}_seed{args.seed}.png"
    result.images[0].save(output)

    print(f"[INFO] method: {tag}")
    if args.vde:
        print(f"[INFO] transformer calls: {getattr(pipe, '_vde_real_infer_steps', 'unknown')}/{args.num_inference_steps}")
    print(f"[INFO] saved: {output}")
    print(f"[INFO] elapsed time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
