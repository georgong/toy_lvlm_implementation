#%% import neceesary class and define model
import torch
import torch.nn as nn
from transformers import ViTModel
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch.nn.functional as F
from torchvision import transforms
class visual_encoder(nn.Module):
    def __init__(self,visual_encoder_name = "google/vit-base-patch16-224"):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(visual_encoder_name)
        self.projection = nn.Linear(768,512)
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self,x):
        output = self.encoder(x)
        return self.projection(output.last_hidden_state[:, 0, :])
    

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn

class text_encoder(nn.Module):
    def __init__(self, model_name="t5-small", projection_dim=512):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5ForConditionalGeneration.from_pretrained(model_name)
        self.projection = nn.Linear(512, projection_dim)

        # 冻结 encoder 参数（可选）
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, texts: list[str] | str):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.encoder.device)

        # 调用 encoder（跳过 decoder/labels）
        encoder_outputs = self.encoder.encoder(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        last_hidden = encoder_outputs.last_hidden_state  # [B, seq_len, 512]
        pooled = last_hidden.mean(dim=1)  # or use [:,0,:] if you prefer CLS-style
        return self.projection(pooled)    # [B, projection_dim]


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.samples = []
        self.transform = transform

        for entry in hf_dataset:
            image = entry["image"]
            for i in range(1,len(entry),1):  # caption_0 ~ caption_6
                caption = entry[f"caption_{i}"]
                self.samples.append((image, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, caption = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, caption

class Flickr8kValidationSet(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.samples = [
            (entry["image"], entry["caption_0"]) for entry in hf_dataset
        ]
        self.transform = transform
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, caption = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, caption

if __name__ == "__main__":
    from datasets import load_dataset
    from torchvision import transforms
    device=  torch.device("mps")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to [0,1] tensor, shape: [C, H, W]
    ])
    flickr8k = load_dataset("jxie/flickr8k", data_dir="data")
    text_encoder = text_encoder().to(device)
    visual_encoder = visual_encoder().to(device)
#%% test model with dummy output
    temperature = 0.02 
    num_epochs = 5
    from torch.utils.data import DataLoader
    train_data = flickr8k["train"]
    train_dataset = Flickr8kDataset(train_data, transform=image_transform)
    optimizer = torch.optim.AdamW(
    list(visual_encoder.parameters()) + list(text_encoder.parameters()),
    lr=1e-4,
    weight_decay=1e-5
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for _ in range(num_epochs):
        pbar = tqdm(train_loader)
        for batch in pbar:
            image = batch[0].to(device)
            text = batch[1]
            visual_embed = visual_encoder.forward(image)
            text_embed = text_encoder.forward(text)
            visual_embed.shape,text_embed.shape
            norm_visual_embed = F.normalize(visual_embed)
            norm_text_embed = F.normalize(text_embed)
            logits = norm_visual_embed @ norm_text_embed.T
            logits = logits / temperature
            batch_size = logits.size(0)
            labels = torch.arange(batch_size, device=logits.device)
            loss_i2t = F.cross_entropy(logits, labels)        # 图像作为 query，文本作为 target
            loss_t2i = F.cross_entropy(logits.T, labels)      # 文本作为 query，图像作为 target
            loss = (loss_i2t + loss_t2i) / 2
            loss.backward()
            pbar.set_description(f"loss:{loss.item()}")
            optimizer.step()
            optimizer.zero_grad()
    

# %%
    