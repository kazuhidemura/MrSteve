# MineCLIP
mkdir -p downloads/weights/mineclip
gdown https://drive.google.com/uc?id=1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW -O downloads/weights/mineclip/attn.pth

# VPT-Nav
mkdir -p downloads/weights/vpt
aria2c https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.model -d downloads/weights/vpt
gdown https://drive.google.com/uc?id=1JwsKvO-YpGEgmSHIGUmWYcZFOnlq4ffV -O downloads/weights/vpt/vpt_nav.weights

# Steve-1
mkdir -p downloads/weights/steve1
gdown https://drive.google.com/uc?id=1E3fd_-H1rRZqMkUKHfiMhx-ppLLehQPI -O downloads/weights/steve1/steve1.weights
gdown https://drive.google.com/uc?id=1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES -O downloads/weights/steve1/steve1_prior.pt
