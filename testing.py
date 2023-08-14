import clip

model, preprocess = clip.load("ViT-B/16")

import torch
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open("./refcocog/images/COCO_train2014_000000002309.jpg")
img = preprocess(img).unsqueeze(0)

# plt.imshow(img[0].permute(1, 2, 0))
# plt.show()

# out_image = model.encode_image(img)


text = sentence = "clock"
text = clip.tokenize(text)

# out_text = model.encode_text(text)

# print(f'Similarity: {torch.cosine_similarity(out_image, out_text).item():.4f}')


out_image, out_text, patch_tokens = model.encode(img, text)

print(f'Similarity: {torch.cosine_similarity(out_image, out_text).item():.4f}')


def build_probability_map(out_text, patch_tokens):
    patch_tokens = patch_tokens.squeeze(0)[1:]
    sim = []
    for token in patch_tokens:
        sim.append(torch.cosine_similarity(token, out_text).item())
    
    sim = torch.tensor(sim).reshape(14, 14)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sim)
    axs[0].set_title(f'Text: {sentence}')
    axs[1].imshow(img[0].permute(1, 2, 0))
    plt.show()
    

build_probability_map(out_text, patch_tokens)
