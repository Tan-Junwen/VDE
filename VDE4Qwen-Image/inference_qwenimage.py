import argparse
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.utils import is_torch_xla_available, logging

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
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

    alpha_pred = torch.lerp(alpha1, alpha2, weight)
    beta_pred = torch.lerp(beta1, beta2, weight)

    return alpha_pred, beta_pred


def vde_qwenimage_call(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    stable_step: int = 49,
    interval: int = 1,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    if true_cfg_scale > 1 and not has_neg_prompt:
        logger.warning(
            f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
        )
    elif true_cfg_scale <= 1 and has_neg_prompt:
        logger.warning("negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1")

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    num_channels_latents = self.transformer.config.in_channels // 4
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled.")
        guidance = None
    else:
        guidance = None

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    anchor_timesteps = []
    anchor_alpha_cond = []
    anchor_beta_cond = []
    anchor_normal_cond = None

    anchor_alpha_uncond = []
    anchor_beta_uncond = []
    anchor_normal_uncond = None

    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            should_run_transformer = (i <= stable_step) or (i == len(timesteps) - 1) or ((i - stable_step) % interval == 0)

            if should_run_transformer:
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                neg_noise_pred = None
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                if i >= stable_step - 1:
                    lat_norm = torch.norm(latents)
                    lat_unit = latents / (lat_norm + 1e-6)

                    anchor_timesteps.append(t)

                    vt_norm_cond = torch.sum(noise_pred * lat_unit)
                    vt_cond = vt_norm_cond * lat_unit
                    vn_cond = noise_pred - vt_cond
                    vn_norm_cond = torch.norm(vn_cond)
                    anchor_normal_cond = vn_cond / (vn_norm_cond + 1e-6)

                    anchor_alpha_cond.append(vt_norm_cond / (lat_norm + 1e-6))
                    anchor_beta_cond.append(vn_norm_cond / (lat_norm + 1e-6))

                    if do_true_cfg:
                        vt_norm_uncond = torch.sum(neg_noise_pred * lat_unit)
                        vt_uncond = vt_norm_uncond * lat_unit
                        vn_uncond = neg_noise_pred - vt_uncond
                        vn_norm_uncond = torch.norm(vn_uncond)
                        anchor_normal_uncond = vn_uncond / (vn_norm_uncond + 1e-6)

                        anchor_alpha_uncond.append(vt_norm_uncond / (lat_norm + 1e-6))
                        anchor_beta_uncond.append(vn_norm_uncond / (lat_norm + 1e-6))
            else:
                alpha_pred_cond, beta_pred_cond = predict_alpha_beta(anchor_timesteps, anchor_alpha_cond, anchor_beta_cond, t)
                lat_norm = torch.norm(latents)
                noise_pred = alpha_pred_cond * latents + beta_pred_cond * lat_norm * anchor_normal_cond

                neg_noise_pred = None
                if do_true_cfg:
                    alpha_pred_uncond, beta_pred_uncond = predict_alpha_beta(
                        anchor_timesteps, anchor_alpha_uncond, anchor_beta_uncond, t
                    )
                    neg_noise_pred = alpha_pred_uncond * latents + beta_pred_uncond * lat_norm * anchor_normal_uncond

            if do_true_cfg:
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return QwenImagePipelineOutput(images=image)


def dtype_from_name(name):
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def get_args():
    parser = argparse.ArgumentParser(description="Qwen-Image inference with optional VDE acceleration.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--prompt", type=str, default="A bright sunny mountain lake, blue sky, colorful flowers.")
    parser.add_argument("--negative-prompt", type=str, default="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device-map", type=str, default="balanced")
    parser.add_argument("--vde", action="store_true", help="Enable VDE acceleration.")
    parser.add_argument("--stable-step", type=int, default=49)
    parser.add_argument("--interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()

    if args.vde:
        QwenImagePipeline.__call__ = vde_qwenimage_call

    load_kwargs = {"torch_dtype": dtype_from_name(args.dtype)}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    pipe = QwenImagePipeline.from_pretrained(args.model, **load_kwargs)
    if not args.device_map:
        pipe = pipe.to(args.device)

    generator_device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    call_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "generator": generator,
    }
    if args.vde:
        call_kwargs.update({"stable_step": args.stable_step, "interval": args.interval})

    tag = f"vde_s{args.stable_step}i{args.interval}" if args.vde else "original"
    start = time.perf_counter()
    image = pipe(**call_kwargs).images[0]
    elapsed = time.perf_counter() - start

    output = args.output or f"output_qwenimage_{tag}_seed{args.seed}.png"
    image.save(output)
    print(f"[INFO] method: {tag}")
    print(f"[INFO] saved: {output}")
    print(f"[INFO] elapsed time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
