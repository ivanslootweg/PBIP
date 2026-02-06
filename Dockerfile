# // ...existing code...
FROM dockerdex.umcn.nl:5005/diag/base-images:pathology-pt2.7.1

RUN pip3 install \
    git+https://github.com/oval-group/smooth-topk.git \
    # torch==2.1.2 \
    # torchvision==0.16.2 \
    torchaudio==2.1.2 \
    opencv-python==4.9.0.80 \
    # albumentations==1.4.2 \
    # scikit-image==0.22.0 \
    # Pillow==10.2.0 \
    # numpy==1.23.5 \
    # scipy==1.12.0 \
    # scikit-learn==1.4.1.post1 \
    # matplotlib==3.8.4 \
    tqdm==4.66.5 \
    omegaconf==2.3.0 \
    packaging>=23.0 \
    ttach==0.0.3 \
    MedCLIP==0.0.3 \
    # pandas==2.2.1 \
    openslide-python==1.2.0 \
    PyYAML==6.0.1 \
    requests==2.32.3 \
    timm==0.9.16 \
    transformers==4.24.0 \
    tokenizers==0.13.3 \
    huggingface-hub==0.26.1 \
    wandb==0.16.2 \
    # jupyter==1.0.0 \
    jupyterlab==4.0.11 \
    tensorboardX==2.6.2.2 \
    geopandas \
    # wholeslidedata \
    h5py \
    icecream \
    plotly \
    umap-learn