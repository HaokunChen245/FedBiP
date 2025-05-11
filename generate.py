import logging
import os
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPProcessor, CLIPModel

from model import model_types
from config import parse_args
from utils_model import save_model, load_model

from PIL import Image
import clip

args = parse_args()    

def unfreeze_layers_unet(unet, condition):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet

def cvtImg(img):
    img = img.permute([0, 2, 3, 1])
    img = img - img.min()
    img = (img / img.max())
    return img.numpy().astype(np.float32)

def show_examples(x):
    plt.figure(figsize=(10, 10))
    imgs = cvtImg(x)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_examples(x):
    plt.figure(figsize=(10, 5),dpi=200)
    imgs = cvtImg(x)
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)

def show_image(image):
    return Image.fromarray(image)

def prompt_with_template(profession, template):
    profession = profession.lower()
    custom_prompt = template.replace("{{placeholder}}", profession)
    return custom_prompt

def get_prompt_embeddings(prompt_domain, prompt_class, labels, tokenizer, text_encoder, padding_type="do_not_pad"):
    prompt_init = []
    for cid in labels:
        if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
            c = args.categories[cid].lower().replace("_", " ")           
            padding = True
            max_length=tokenizer.model_max_length
            if args.dataset=='dermamnist':
                prompt_init.append(f'A dermatoscopic image of a {c}, a type of pigmented skin lesions')
            elif args.dataset=='bloodmnist':
                prompt_init.append(f'A microscopic image of a {c}, a type of blood cell')
            else:
                prompt_init.append(f'A centered satellite photo of a {c.lower().replace("_", " ")}')
        else:
            prompt_init.append(f'a X style of a X')            
            padding=True
            max_length=None

    inputs = tokenizer(prompt_init, 
        # max_length=tokenizer.model_max_length, 
        padding=padding,
        max_length=max_length, 
        truncation=True,
        return_tensors="pt"
    )
    input_ids = torch.LongTensor(inputs.input_ids)
    text_f = text_encoder(input_ids.to('cuda'))[0]
    if args.dataset in ['bloodmnist', 'dermamnist', 'ucm']:
        st_idx_map = {
            'bloodmnist': 7,
            'dermamnist': 8,
            'ucm': 7
        }
        start_idx = st_idx_map[args.dataset]
        for idx, cid in enumerate(labels):
            num_prompt_class = len(prompt_class[cid])
            text_f[idx][start_idx:start_idx+num_prompt_class] = prompt_class[cid]
            
        start_idx_domain = 2
        num_prompt_domain_map = {
            'dermamnist': 4,
            'ucm': 3,
        }
        num_prompt_domain = num_prompt_domain_map[args.dataset]
        text_f[:, start_idx_domain:start_idx_domain+num_prompt_domain] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)
        
    else:
        num_prompt_domain = 1
        text_f[:, 2:2+num_prompt_domain] = prompt_domain.unsqueeze(0).repeat(labels.shape[0], 1, 1)
        num_prompt_class = 1
        text_f[:, -1-num_prompt_class:-1] = prompt_class[labels]

    return text_f

def main():
    args = parse_args()    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'test_config.yaml'), 'w'))

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    if args.scheduler == 'ddim':
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False,
            num_train_timesteps=1000,
            steps_offset=1,
        )
    elif args.scheduler == 'pndm':
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
    elif args.scheduler == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
        )
    else:
        raise NotImplementedError(args.scheduler)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    num_concepts=7

    device=torch.device('cuda')

    model=StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    model=model.to(device)
    if args.fp16:
        print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()

    dataloader = None
    # following https://arxiv.org/pdf/2306.16064
    categories = args.categories

    def generate_data_per_domain_prompt(model, categories, unet, device, args, idx=None):
        domain = args.domain
        domains = args.domains
        sample_per_class_per_domain = 160
        
        did = domains.index(domain)

        from collections import defaultdict
        # load latents
        latents = defaultdict(list)
        if idx is not None:
            latents_root = f'/root/InterpretDiffusion/exps_{args.dataset}/fedlip_prompt_d_{domain}_multiclient'
            latents_mean = torch.load(f"{latents_root}/mean_{idx}.pt")
            latents_std = torch.load(f"{latents_root}/std_{idx}.pt")
        else:
            latents_root = f'/root/InterpretDiffusion/exps_{args.dataset}/fedlip_prompt_d_{domain}_multiclient_interpolated_test'
            latents_mean = torch.load(f"{latents_root}/mean.pt")
            latents_std = torch.load(f"{latents_root}/std.pt")
        concept = None

        # load soft prompts
        latents_root = f'/root/InterpretDiffusion/exps_{args.dataset}/fedlip_prompt_d_{domain}'
        if idx is not None:
            prompt_class = torch.load(f"{latents_root}/prompt_class_{idx}.pth")
        else:
            prompt_class = torch.load(f"{latents_root}/prompt_class.pth")
        if 'avg_cprompt' in args.test_type:
            prompt_class = [[] for _ in range(len(prompt_class))]
            for d in domains:
                t = torch.load(f'/root/InterpretDiffusion/exps_{args.dataset}/fedlip_prompt_d_{d}/prompt_class.pth')
                for i in range(len(t)):
                    prompt_class[i] += [t[i]]
            for i in range(len(prompt_class)):
                prompt_class[i] = torch.stack(prompt_class[i], dim=0).mean(dim=0)
            prompt_class = torch.cat(prompt_class, dim=0)

        if idx is not None:
            prompt_domain = torch.load(f"{latents_root}/prompt_domain_{idx}.pth")
        else:
            prompt_domain = torch.load(f'{latents_root}/prompt_domain.pth')
        if "wnoise" in args.test_type and prompt_domain is not None:
            # add random noise to spec_concept     
            intensity = float(args.test_type.split("wnoise_")[1][:3])
            prompt_domain = prompt_domain + torch.randn_like(prompt_domain) * intensity

        for cid, c in enumerate(categories):
            save_image_dir=os.path.join(args.output_dir, c, f"{domain}_{args.test_type}")
            if idx is not None:
                save_image_dir += f"_{idx}"
            os.makedirs(save_image_dir, exist_ok=True)   
            
            for i in range(sample_per_class_per_domain):
                if not (i>=args.start_idx and i<=args.end_idx): continue
                # if os.path.exists(f"{save_image_dir}/{i}.jpg"): continue
                seed = did * 1000000 + cid * 1000 + i
                labels = torch.tensor([cid] * 1).to(device)
                prompt_embeds = get_prompt_embeddings(
                    prompt_domain, prompt_class, labels, 
                    tokenizer, text_encoder, 
                    padding_type="max_length")

                if True:
                    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        print(alpha)
                        mean, std = latents_mean[c][i % len(latents_mean[c])], latents_std[c][i % len(latents_mean[c])]
                        mean = mean * alpha + latents_mean[c][(i+5) % len(latents_mean[c])] * (1-alpha)
                        std = std * alpha * alpha + latents_std[c][(i+5) % len(latents_mean[c])] * (1-alpha) * (1-alpha)
                        sample = torch.randn(
                            mean.shape,
                            device=device,
                            dtype=mean.dtype,
                        )
                        latent = mean + std * sample

                        image = predict_cond(
                            model=model, 
                            prompt=None, prompt_embeds=prompt_embeds,
                            seed=seed, condition=concept,
                            img_size=args.resolution, num_inference_steps=args.num_inference_steps,
                            negative_prompt=args.negative_prompt, latent=latent,
                        )
                        image.save(f"{save_image_dir}/{i}_{alpha}.jpg")

                sample = torch.randn(
                    mean.shape,
                    device=device,
                    dtype=mean.dtype,
                )
                latent = mean + std * sample

                image = predict_cond(
                    model=model, 
                    prompt=None, prompt_embeds=prompt_embeds,
                    seed=seed, condition=concept,
                    img_size=args.resolution, num_inference_steps=args.num_inference_steps,
                    negative_prompt=args.negative_prompt, latent=latent,
                )
                image.save(f"{save_image_dir}/{i}.jpg")
    
    if 'multiclient' in args.test_type:
        for idx in range(5):
            generate_data_per_domain_prompt(model=model, categories=categories, unet=unet, device=device, args=args, idx=idx)
    else:
        generate_data_per_domain_prompt(model=model, categories=categories, unet=unet, device=device, args=args)

def predict_cond(model, 
                prompt, 
                seed, 
                condition, 
                img_size,
                num_inference_steps=50,
                interpolator=None, 
                negative_prompt=None,
                latent=None,
                prompt_embeds=None,
                ):
    
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    output = model(prompt=prompt, prompt_embeds=prompt_embeds,
                height=img_size, width=img_size, 
                num_inference_steps=num_inference_steps, 
                generator=generator, 
                controlnet_cond=condition,
                controlnet_interpolator=interpolator,
                negative_prompt=negative_prompt,
                latents=latent.unsqueeze(0) if latent is not None else None,
                )
    image = output[0][0]
    return image

if __name__ == "__main__":
    main()