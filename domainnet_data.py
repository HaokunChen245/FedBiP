import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from collections import defaultdict

class DomainNetDataset(Dataset):
    def __init__(self, args, domain, split='train', num_shot=-1, 
        root_dir="data/domainnet", transform=None, tokenizer=None,
        client_id=None):
        self.root_dir = root_dir
        self.train_type = args.train_type
        self.client_id = client_id
        if transform is None:
            if 'train' in split:  
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
        else:
            self.transform = transform
        self.domain = domain
        self.tokenizer = tokenizer

        self.domains = args.domains + ["syn"]
        self.categories = args.categories
        self.image_paths = self._get_image_paths()

        self.split = split
        self.to_few_shot(num_shot)

    def to_few_shot(self, num_shot):
        if num_shot==0: return
        few_shot_dict = defaultdict(list)

        domain = self.domain
        split = self.split.split("_")[0]
        split_file = f'{domain}_{split}.txt'
        if not 'syn' in self.split:
            with open(f"{self.root_dir}/{split_file}", "r") as f:
                for l in f.readlines():
                    for category in self.categories:
                        if category in l and (len(few_shot_dict[f"{category}_{domain}"])<num_shot or num_shot==-1 or self.client_id is not None):
                            img_path, _ = l.strip().split(" ")
                            img_path = f"{self.root_dir}/{img_path}"
                            few_shot_dict[f"{category}_{domain}"].append([img_path, domain, category])   
            if self.client_id is not None:
                for category in self.categories:
                    data_per_c_per_d = min(len(few_shot_dict[f"{category}_{domain}"])//5, 16)
                    few_shot_dict[f"{category}_{domain}"] = few_shot_dict[f"{category}_{domain}"][data_per_c_per_d*self.client_id:data_per_c_per_d*(self.client_id+1)]
        else:
            file_suffix = self.split[6:].replace(f"_{num_shot}", "").replace(f"{num_shot}", "")
            print(file_suffix)
            domain_id = self.domains.index(self.domain)           
            for c in self.categories:
                p = f"data/datasets_domainnet/{c}/{self.domain}_{file_suffix}"
                if file_suffix=='syn_base':
                    p = f"data/datasets_domainnet/{c}/{self.domain}"     
                for img_id in os.listdir(p):
                    if img_id.endswith(".jpg") or img_id.endswith(".png"):                                                
                        img_path = f"{p}/{img_id}"
                        if len(few_shot_dict[f"{c}_syn"])<num_shot or num_shot==-1:
                            few_shot_dict[f"{c}_syn"].append([img_path, "syn", c])

        self.image_paths = []
        for k in few_shot_dict.keys():
            self.image_paths.extend(few_shot_dict[k])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, domain, category = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.train_type == 'general_only':
            prompt = [f"a {domain} style of a {category}"]
        elif 'fedlip' in self.train_type:
            prompt = ['an image']
        else:
            prompt = ["an image of a " + category]

        if self.tokenizer is None:
            prompt = None
        else:
            prompt = self.tokenizer(prompt)[0]

        if self.transform:
            image = self.transform(image)
        domain_id = self.domains.index(domain)
        class_id = self.categories.index(category)
        return image, prompt, domain_id, class_id, image_path

    def _get_image_paths(self):
        image_paths = []
        domain_dir = os.path.join(self.root_dir, self.domain)
        if os.path.isdir(domain_dir):
            for category in os.listdir(domain_dir):
                if not category in self.categories: continue
                category_dir = os.path.join(domain_dir, category)
                if os.path.isdir(category_dir):
                    for filename in os.listdir(category_dir):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            image_path = os.path.join(category_dir, filename)
                            image_paths.append([image_path, self.domain, category])
        return image_paths

if __name__ == "__main__":
    # Define the arguments
    class args:
        pass
    args.resolution = 512
    args.center_crop = True
    args.random_flip = True

    # Create an instance of the DomainNetDataset
    dataset = DomainNetDataset(args, domain='clipart', root_dir="data/domainnet", transform=None)

    # Print the length of the dataset
    print("Dataset length:", len(dataset))

    # Get the first item from the dataset
    first_item = dataset[0]

    print("First item shape:", first_item[0].shape, "Category:", first_item[1])
    
from torch.utils.data import ConcatDataset

def get_dataloader_domain(args,
        batch_size, transform, split,
        domain, tokenizer, collate_fn, num_shot=-1,
        num_workers=4, shuffle=True, client_id=None):
    dataset = DomainNetDataset(args, 
            domain=domain, 
            num_shot=num_shot, 
            split=split,
            root_dir="data/domainnet", 
            transform=transform,
            tokenizer=tokenizer, 
            client_id=client_id
        )
    categories = dataset.categories
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    print("Split:", split, "Domain:", domain, "Count:", len(dataset))

    return dataloader

def get_dataloader(args,
        batch_size, transform, split,
        tokenizer, collate_fn, num_shot=-1,
        num_workers=4, shuffle=True):

    datasets = []
    for d in args.domains:
        dataset = DomainNetDataset(args, 
            domain=d, 
            num_shot=num_shot, 
            split=split,
            root_dir="data/domainnet", 
            transform=transform,
            tokenizer=tokenizer, 
        )
        categories = dataset.categories
        datasets.append(dataset)
        print("Split:", split, "Domain:", d, ", Count:", len(dataset))

    all_data = ConcatDataset(datasets)        
    dataloader = torch.utils.data.DataLoader(
        all_data,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    return dataloader