import torch.nn as nn



# 768 -> hidden per visual
# 512 -> hidden per text
class BackboneAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(hidden_dim, input_dim)

        # initialize down proj with Kaiming Normal, up proj with zeros
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.up_proj.weight)
        self.up_proj.bias.data.fill_(0.0) # not specified in paper, maybe remove

    def forward(self, x):
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x



class PreFusionAdapter(nn.Module):
    def __init__(self, image_input_dim, text_input_dim, shared_dim, n_head):
        super().__init__()
        self.ln_image = nn.LayerNorm(image_input_dim)
        self.ln_text = nn.LayerNorm(text_input_dim)
        self.W_v2s = nn.Linear(image_input_dim, shared_dim)
        self.W_t2s = nn.Linear(text_input_dim, shared_dim)
        self.CA_image = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=n_head)
        self.CA_text = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=n_head)
        self.W_s2v = nn.Linear(shared_dim, image_input_dim)
        self.W_s2t = nn.Linear(shared_dim, text_input_dim)

        # initialize projections with zeros
        nn.init.zeros_(self.W_v2s.weight)
        nn.init.zeros_(self.W_t2s.weight)
        nn.init.zeros_(self.W_s2v.weight)
        nn.init.zeros_(self.W_s2t.weight)
        self.W_v2s.bias.data.fill_(0.0) # not specified in paper, maybe remove
        self.W_t2s.bias.data.fill_(0.0) # not specified in paper, maybe remove
        self.W_s2v.bias.data.fill_(0.0) # not specified in paper, maybe remove
        self.W_s2t.bias.data.fill_(0.0) # not specified in paper, maybe remove

    def forward(self, image, text):
        image = self.ln_image(image)
        text = self.ln_text(text)
        image = self.W_v2s(image)
        text = self.W_t2s(text)
        # in CA_image the query is the image, the key and value are the text
        image, _ = self.CA_image(query=image, key=text, value=text, need_weights=False)
        text, _ = self.CA_text(query=text, key=image, value=image, need_weights=False)
        image = self.W_s2v(image)
        text = self.W_s2t(text)
        return image, text



class PostFusionAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # cross attention without the projections at the start
        # then MHSA and MLP for each modality

    def forward(self, x):
        return x