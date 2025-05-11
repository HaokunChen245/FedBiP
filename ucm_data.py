import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from collections import defaultdict

class UCMerced_LandUseDataset(Dataset):
    def __init__(self, args, domain, split='train', num_shot=-1, root_dir="data/UCMerced_LandUse/Images", transform=None, tokenizer=None):
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
        if 'niid' in self.split and not 'syn' in self.split:
            if '001' in self.split:
                alpha = 0.01
            else:
                alpha = 0.5
            X = torch.load(f"data/ucm_niid/X_{alpha}_{domain.split('_')[1]}.pt")
            y = torch.load(f"data/ucm_niid/y_{alpha}_{domain.split('_')[1]}.pt")
            ct = [0 for _ in range(len(self.categories))]
            for i in range(len(y)):
                ct[int(y[i])]+=1
            print(ct)
            self.image_paths = [(X[i], domain, self.categories[int(y[i])]) for i in range(len(X))]
        else:
            self.to_few_shot(num_shot)

    def to_few_shot(self, num_shot):
        if num_shot==0: return
        few_shot_dict = defaultdict(list)

        domain = self.domain
        split = self.split.split("_")[0]
        if not 'syn' in self.split:
            client_id = int(self.domain.split("_")[1])            
            for category in self.categories:
                category_fn = category.replace("_", "")
                category_dir = os.path.join(self.root_dir, category_fn)
                for filename in os.listdir(category_dir):
                    if not 'jpg' in filename: continue
                    img_path = os.path.join(category_dir, filename)
                    few_shot_dict[f"{category}_{domain}"].append([img_path, domain, category])                    

                if 'train' in self.split:
                    # take the first num_shot images
                    few_shot_dict[f"{category}_{domain}"] = few_shot_dict[f"{category}_{domain}"][num_shot*client_id:num_shot*(client_id+1)]
                elif self.split == 'test':   
                    # take the last num_shot images
                    few_shot_dict[f"{category}_{domain}"] = few_shot_dict[f"{category}_{domain}"][::-1][num_shot*client_id:num_shot*(client_id+1)]
        else:
            file_suffix = self.split[6:].replace(f"_{num_shot}", "")
            domain_id = self.domains.index(self.domain)
            for c in self.categories:
                if not os.path.exists(f"data/datasets_ucm/{c}/{self.domain}_{file_suffix}"):
                    continue
                for img_id in os.listdir(f"data/datasets_ucm/{c}/{self.domain}_{file_suffix}"):
                    if img_id.endswith(".jpg") or img_id.endswith(".png"):                                                
                        img_path = f"data/datasets_ucm/{c}/{self.domain}_{file_suffix}/{img_id}"
                        if len(few_shot_dict[f"{c}_syn"])<num_shot or num_shot==-1:
                            few_shot_dict[f"{c}_syn"].append([img_path, "syn", c])

        self.image_paths = []
        for k in few_shot_dict.keys():
            self.image_paths.extend(few_shot_dict[k])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, domain, category = self.image_paths[idx]
        if 'niid' in self.split  and not 'syn' in self.split:
            image = image_path
        else:
            image = Image.open(image_path).convert("RGB")
            with open("data/ucm_img_paths.txt", 'a') as f:
                f.write(f"{image_path}\n")

        prompt = [f"A centered satellite photo of {category}"]

        if self.tokenizer is None:
            prompt = None
        else:
            prompt = self.tokenizer(prompt)[0]

        if self.transform and isinstance(image, Image.Image):
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

def get_datasets(args, transform, tokenizer, num_shot=-1):
    datasets = []
    num_shot = args.num_shot * 5
    d = 'client_0'
    dataset = UCMerced_LandUseDataset(args, 
        domain=d, 
        num_shot=num_shot, 
        split='train',
        transform=transform,
        tokenizer=tokenizer, 
    )
    categories = dataset.categories
    datasets.append(dataset)
    print("Split: train, Domain: all", "Count:", len(dataset))

    return dataset

def get_dataloader_domain(args,
        batch_size, transform, split,
        domain, tokenizer, collate_fn, num_shot=-1,
        num_workers=4, shuffle=True):
    dataset = UCMerced_LandUseDataset(args, 
            domain=domain, 
            num_shot=num_shot, 
            split=split,
            root_dir="data/UCMerced_LandUse/Images", 
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
        dataset = UCMerced_LandUseDataset(args, 
            domain=d, 
            num_shot=num_shot, 
            split=split,
            root_dir="data/UCMerced_LandUse/Images", 
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