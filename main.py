import clip
import torch
from PIL import Image
from matplotlib import pyplot as plt

model, preprocess = clip.load("ViT-B/16")
model.init_adapters()

img = original = Image.open("./refcocog/images/COCO_train2014_000000002842.jpg")
# img.show()
img = preprocess(img).unsqueeze(0)

text = sentence = "tennis racket"
text = clip.tokenize(text)

out_image, out_text, patch_tokens, text_tokens = model.encode(img, text)

'''

out_image: normal 1x512 tensor, features of the image in the final shared space
out_text: normal 1x512 tensor, features of the text in the final shared space
patch_tokens: 1x197x512 tensor, contains CLS token + 196 patch tokens from the 14x14 grid
text_tokens: 1xLx512 tensor, L = len(sent) + 2, because it's SOT + words + EOT tokens

Every token has been passed through the last layer norm and projection, so they are in the shared space

The out_image is simply the EOT token from the text_tokens, 
while the out_image is the CLS token from the patch_tokens.

'''
# print(out_image.shape, out_text.shape)
# print(patch_tokens.shape, text_tokens.shape)


print(f'Similarity: {torch.cosine_similarity(out_image, out_text).item():.4f}')
print(f'Similarity: {torch.cosine_similarity(patch_tokens[0, 0].unsqueeze(0), text_tokens[0, -1].unsqueeze(0)).item():.4f}')


def build_probability_map(out_text, patch_tokens):
    patch_tokens = patch_tokens.squeeze(0)[1:]
    sim = []
    for token in patch_tokens:
        sim.append(torch.cosine_similarity(token, out_text).item())
    
    sim = torch.sigmoid(torch.tensor(sim))
    sim = torch.tensor(sim).reshape(14, 14)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sim)
    axs[0].set_title(f'Text: {sentence}')
    axs[1].imshow(img[0].permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
    

build_probability_map(out_text, patch_tokens)
