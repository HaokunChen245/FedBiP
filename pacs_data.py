import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from collections import defaultdict

class PACSDataset(Dataset):
    def __init__(self, args, domain, split='train', num_shot=-1, root_dir="data/pacs/kfold", transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.train_type = args.train_type
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
        if not 'syn' in self.split:
            domain_dir = os.path.join(self.root_dir, self.domain)
            for category in self.categories:
                category_dir = os.path.join(domain_dir, category)
                for filename in os.listdir(category_dir):
                    img_path = os.path.join(category_dir, filename)
                    few_shot_dict[f"{category}_{domain}"].append([img_path, domain, category])                    

                if 'train' in self.split:
                    # take the first num_shot images
                    few_shot_dict[f"{category}_{domain}"] = few_shot_dict[f"{category}_{domain}"][:num_shot]
                elif self.split == 'test':   
                    # take the last num_shot images
                    few_shot_dict[f"{category}_{domain}"] = few_shot_dict[f"{category}_{domain}"][-num_shot:]
        else:
            file_suffix = self.split[6:].replace(f"_{num_shot}", "")
            domain_id = self.domains.index(self.domain)
            for c in self.categories:
                p = f"data/datasets_pacs/{c}/{self.domain}_{file_suffix}"
                if file_suffix=='syn_base':
                    p = f"data/datasets_pacs/{c}/{self.domain}"    
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
        prompt = [f"a {domain} style of a {category}"]

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
                category_dir = os.path.join(domain_dir, category)
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
    dataset = DomainNetDataset(args, domain='clipart', root_dir="data/pacs", transform=None)

    # Print the length of the dataset
    print("Dataset length:", len(dataset))

    # Get the first item from the dataset
    first_item = dataset[0]

    print("First item shape:", first_item[0].shape, "Category:", first_item[1])
    
from torch.utils.data import ConcatDataset

def get_dataloader_domain(args,
        batch_size, transform, split,
        domain, tokenizer, collate_fn, num_shot=-1,
        num_workers=4, shuffle=True):
    dataset = PACSDataset(args, 
            domain=domain, 
            num_shot=num_shot, 
            split=split,
            root_dir="data/pacs/kfold", 
            transform=transform,
            tokenizer=tokenizer, 
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
        dataset = PACSDataset(args, 
            domain=d, 
            num_shot=num_shot, 
            split=split,
            root_dir="data/pacs/kfold", 
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