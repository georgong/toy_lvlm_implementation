#%% import neceesary class and define model
import torch
import torch.nn as nn
from transformers import ViTModel
from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch.nn.functional as F
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
    

class text_encoder(nn.Module):
    def __init__(self,text_encoder_name = "t5-small"):
        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(text_encoder_name)
        self.tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
        self.projection = nn.Linear(512,512)
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self,x:tuple):
        source_text,target_text = x
        input_ids = self.tokenizer(source_text, return_tensors="pt").input_ids
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(target_text, return_tensors="pt").input_ids
        outputs = self.encoder(input_ids=input_ids, 
                     labels=labels,  
                     output_hidden_states=True,
                     return_dict=True)
        encoder_last_hidden = outputs.encoder_hidden_states[-1]  # [1, src_len, 512]
        pooled = encoder_last_hidden.mean(dim=1)
        return self.projection(pooled)


if __name__ == "__main__":
    text_encoder = text_encoder()
    visual_encoder = visual_encoder()
#%% test model with dummy output
    temperature = 0.02 
    visual_embed = visual_encoder.forward(torch.randn(1, 3, 224, 224))
    source_text = "summarize: this is a long paragraph of text"
    target_text = "short summary"
    text_embed = text_encoder.forward((source_text,target_text))
    visual_embed.shape,text_embed.shape
    print(visual_embed.shape,text_embed.shape)
    norm_visual_embed = F.normalize(visual_embed)
    norm_text_embed = F.normalize(text_embed)
    logits = norm_visual_embed @ norm_text_embed.T
    logits = logits / temperature
    

# %%
    