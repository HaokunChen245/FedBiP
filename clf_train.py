import torch
from config import parse_args

import torchvision.models as models
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda'
args = parse_args()

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

if args.dataset == 'domainnet':
    from domainnet_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'pacs':
    from pacs_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'officehome':
    from officehome_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'bloodmnist':
    from bloodmnist_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'dermamnist':
    from dermamnist_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'ucm':
    from ucm_data import get_dataloader, get_dataloader_domain

def tokenize_captions(examples, is_train=False):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
    input_ids = inputs.input_ids
    return input_ids
        
def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [example[1] for example in examples]
    padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    domain_ids = torch.tensor([example[2] for example in examples])
    class_ids = torch.tensor([example[3] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_tokens.input_ids,
        "attention_mask": padded_tokens.attention_mask,
        "domain_ids": domain_ids,
        "class_ids": class_ids,
    }

def train(seed, train_setting):
    model = models.resnet18(pretrained=args.pretrained)
    print(f"Training with seed {seed} and setting {train_setting}")
    setup_seed(seed)
    categories = args.categories

    num_shots = int(train_setting.split("_")[-1])
    if 'multiclient' in train_setting:
        sets = []
        for domain in args.domains:
            for i in range(args.client_num):
                train_setting_temp = train_setting.replace(f'_{args.client_num}_', f'_{i}_')
                trainloader = get_dataloader_domain(
                        args, args.train_batch_size, None,
                        train_setting_temp, domain, tokenize_captions,  
                        collate_fn, num_shot=num_shots,
                        client_id=i)   
                sets.append(trainloader.dataset)        
        sets = torch.utils.data.ConcatDataset(sets)
        print(len(sets))
        train_dataloader = torch.utils.data.DataLoader(sets, 
            batch_size=args.train_batch_size, 
            collate_fn=collate_fn, 
            shuffle=True)
    elif 'fgl' in train_setting:
        num_shots = -1 
        train_dataloader = get_dataloader_domain(
            args, args.train_batch_size, None,
            train_setting, 'fgl', tokenize_captions,  
            collate_fn, num_shot=args.num_shot)  
    elif 'fedd3' in train_setting:
        num_shots = -1  
        xxx = []
        yyy = []   
        if args.dataset in ['ucm', 'dermamnist']:
            p = f'/root/InterpretDiffusion/datasets_fedd3/{args.dataset}/kip.pt'
            params = torch.load(p)
            x = np.asarray(params['x'])
            y = np.asarray(params['y'])
            xxx.append(torch.tensor(x))
            yyy.append(torch.tensor(y))
        else:
            for d in args.domains:
                p = f'/root/InterpretDiffusion/datasets_fedd3/{args.dataset}/kip_{d}.pt'
                params = torch.load(p)
                x = np.asarray(params['x'])
                y = np.asarray(params['y'])
                xxx.append(torch.tensor(x))
                yyy.append(torch.tensor(y))

        x = torch.cat(xxx).permute(0, 3, 1, 2)
        y = torch.cat(yyy).argmax(dim=1).long()
        print(x.shape, y.shape)
        dataset = torch.utils.data.TensorDataset(x, y)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    else:
        train_dataloader = get_dataloader(
                args, args.train_batch_size, None,
                train_setting, tokenize_captions,  
               collate_fn, num_shot=num_shots)    

    if args.dataset=='pacs':
        num_shot_test = 32
    elif args.dataset=='ucm':
        num_shot_test = 8
    else:
        num_shot_test = -1
    test_dataloader = get_dataloader(
            args, args.train_batch_size, None,
            'test', tokenize_captions,  
            collate_fn, num_shot=num_shot_test)  
    
    num_epochs = 50
    num_classes = len(categories)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    
    num_steps_per_epoch = len(train_dataloader)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     num_steps_per_epoch * num_epochs,
    #     eta_min=0.001)

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            if 'fedd3' in train_setting:
                outputs = model(batch[0].to(device))
                labels = batch[1].to(device)
            else:
                outputs = model(batch['pixel_values'].to(device))
                labels = batch['class_ids'].to(device)
            # Compute loss and perform backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()

        if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                d_count = defaultdict(list)

                for batch in test_dataloader:
                    inputs = batch["pixel_values"].to(device)
                    outputs = model(inputs)
                    labels = batch['class_ids'].to(device)
                    preds = torch.argmax(outputs, dim=1)
                    # Compute loss and accuracy
                    total_correct += preds.eq(labels).sum().item()
                    total_samples += inputs.size(0)
                    for i, did in enumerate(batch['domain_ids']):
                        d_count[did.item()].append(labels[i].item() == preds[i].item())

            tot_acc = 0.
            for k in d_count.keys():
                acc = sum(d_count[k]) / len(d_count[k]) * 100 
                tot_acc += acc
                print(f"{args.domains[k]}: {round(acc, 3)}", end=", ")

            if args.dataset in ['ucm', 'dermamnist']:
                print(f"Epoch {epoch+1}: Accuracy = {round(total_correct/total_samples, 3)}")
            else:
                print(f"{train_setting}/Epoch {epoch+1}: Accuracy = {round(tot_acc/len(d_count.keys()), 3)}")

if __name__=="__main__":    
    for seed in [0, 1, 2]:
        if not isinstance(args.train_type, list):
            args.train_type = [args.train_type]
        for train_setting in args.train_type:
            print(train_setting)
            train(seed, train_setting)