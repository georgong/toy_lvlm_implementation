#%% import neceesary class and define model
import torch
import torch.nn as nn
from transformers import ViTModel
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
class VisualEncoder(nn.Module):
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


class TextEncoder(nn.Module):
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
            for i in range(1,len(entry)-1,1):  # caption_0 ~ caption_6
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
    
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.text_encoder = TextEncoder()
        #self.temperature_learner = nn.Parameter(
        #    torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        #    )

    def forward(self,images,texts):
        visual_embed = self.visual_encoder.forward(images)
        text_embed = self.text_encoder.forward(texts)
        norm_visual_embed = F.normalize(visual_embed, dim = -1)
        norm_text_embed = F.normalize(text_embed, dim = -1)
        logits = norm_visual_embed @ norm_text_embed.T
        return logits / 0.02

    @torch.no_grad()
    def generate_image_from_texts(self, texts, image):
        """
        Given multiple texts and one image, find which text best describes the image
        (i.e. text retrieval given an image).
        """
        image_embed = self.visual_encoder(image.unsqueeze(0))     # [1, D]
        text_embed = self.text_encoder(texts)                     # [N, D]

        image_embed = F.normalize(image_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        sim = image_embed @ text_embed.T                          # [1, N]
        sim_scores = sim.softmax(dim=-1)                          # [1, N]
        best_idx = sim_scores.argmax(dim=-1).item()
        return best_idx, sim_scores.squeeze()

    @torch.no_grad()
    def generate_image_from_images(self, text, images):
        """
        Given one text and multiple images, find which image best matches the text.
        (i.e. image retrieval given a query caption).
        """
        text_embed = self.text_encoder([text])                    # [1, D]
        image_embed = self.visual_encoder(images)                 # [N, D]

        text_embed = F.normalize(text_embed, dim=-1)
        image_embed = F.normalize(image_embed, dim=-1)

        sim = text_embed @ image_embed.T                          # [1, N]
        sim_scores = sim.softmax(dim=-1)                          # [1, N]
        best_idx = sim_scores.argmax(dim=-1).item()
        return best_idx, sim_scores.squeeze()

def evaluate_clip(model, dataloader, device="cuda"):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for images, texts in tqdm(dataloader):
            images = images.to(device)
            img_feat = F.normalize(model.visual_encoder(images), dim=-1)
            txt_feat = F.normalize(model.text_encoder(texts), dim=-1)

            all_image_embeds.append(img_feat)
            all_text_embeds.append(txt_feat)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [N, D]

    sim_matrix = all_image_embeds @ all_text_embeds.T      # [N, N]
    sim_matrix = sim_matrix / 0.02                         # Match your model's scaling

    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    acc_i2t = (sim_matrix.argmax(dim=1) == labels).float().mean().item()
    acc_t2i = (sim_matrix.argmax(dim=0) == labels).float().mean().item()
    avg_acc = (acc_i2t + acc_t2i) / 2

    print(f"[Eval] Image → Text Accuracy: {acc_i2t * 100:.2f}%")
    print(f"[Eval] Text → Image Accuracy: {acc_t2i * 100:.2f}%")
    print(f"[Eval] Average Accuracy: {avg_acc * 100:.2f}%")

    return {
        "i2t_acc": acc_i2t,
        "t2i_acc": acc_t2i,
        "avg_acc": avg_acc
    }


if __name__ == "__main__":
    from datasets import load_dataset
    from torchvision import transforms
    device=  torch.device("mps")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to [0,1] tensor, shape: [C, H, W]
    ])
    flickr8k = load_dataset("jxie/flickr8k", data_dir="data")
    clip_model = CLIP().to(device)
#%% test model with dummy output
    num_epochs = 5
    from torch.utils.data import DataLoader
    train_data = flickr8k["train"]
    train_dataset = Flickr8kDataset(train_data, transform=image_transform)
    valid_dataset = Flickr8kValidationSet(train_data, transform=image_transform)
    optimizer = torch.optim.AdamW(
    clip_model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    unfrozen_visual_layers = 0
    unfrozen_text_layers = 0
    def unfreeze_last_n_layers(model, n, encoder_type="vit"):
        if encoder_type == "vit":
            layers = model.encoder.encoder.layer  # ViT-like encoder
        elif encoder_type == "t5":
            layers = model.encoder.encoder.block  # T5 encoder
        else:
            raise ValueError("Unsupported encoder type")

        num_layers = len(layers)
        for i in range(num_layers - n, num_layers):
            for p in layers[i].parameters():
                p.requires_grad = True
    for epoch in range(num_epochs):
        print(evaluate_clip(model = clip_model, dataloader= valid_loader, device = "mps"))
        # 每 2 个 epoch 解冻 2 层（可自定义）
        if epoch % 2 == 0 and epoch > 0:
            unfrozen_visual_layers += 2
            unfrozen_text_layers += 2
            print("unfreeze")

            unfreeze_last_n_layers(clip_model.visual_encoder, unfrozen_visual_layers, encoder_type="vit")
            unfreeze_last_n_layers(clip_model.text_encoder, unfrozen_text_layers, encoder_type="t5")

            # ⚠️ 重建 optimizer（包含所有 requires_grad=True 的参数）
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, clip_model.parameters()),
                lr=1e-4,
                weight_decay=1e-5
            )

        pbar = tqdm(train_loader)
        for batch in pbar:
            image = batch[0].to(device)
            text = batch[1]
            logits = clip_model(image, text)

            batch_size = logits.size(0)
            labels = torch.arange(batch_size, device=logits.device)

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
        
    

# %%
    