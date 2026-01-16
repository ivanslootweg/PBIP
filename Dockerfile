# h5py
# icecream

FROM dockerdex.umcn.nl:5005/diag/base-images:pathology-pt2.7.1
RUN pip3 install \
    git+https://github.com/oval-group/smooth-topk.git \
    tensorboardX \
    geopandas \
    timm \
    wholeslidedata \
    omegaconf \
    packaging \
    ttach \
    MedCLIP \
    openslide-python \
    PyYAML \
    h5py \
    icecream \
    tokenizers \
    transformers \
    wandb \