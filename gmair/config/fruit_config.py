import os

dataset = 'fruit2d'
dataset_dir = './data/fruit2d/'

log_dir = './logs/fruit2d'

# load pretrained model
pre_model_path = None
start_iters = 0

# training options
batch_size = 16
num_workers = 8
input_image_shape = [3, 128, 128]

# learning rates
lr = 1e-4
encoder_what_lr = 5e-5
encoder_cat_lr = 5e-5
decoder_lr = 5e-5

# architecture
decoder_type = 'cnn'
n_backbone_features = 64

# Object attribute dimensions
n_what = 256

# number of clusters
num_classes = 10

object_shape = [32, 32] # [height, width]
box_max_shape = [72, 72] # [height, width]

# priors
priors = {
    'offset_y':[0., 1.],
    'offset_x':[0., 1.],
    'scale_y':[0., 1.],
    'scale_x':[0., 1.],
    'depth':[0., 1.],
}
# Dyanmic prior used by the object classification latent variable
obj_cls_prior_k = 0.01
obj_cls_prior_bottom = 6e-6

box_mu_bn_gamma = 0.5

freezing_what_iters = 0
freezing_cat_iters = 0

# coefficients of loss functions
alpha = {
    'pres' : 1.0,
    'what' : 1.0,
    'offset_y' : 1.0,
    'offset_x' : 1.0,
    'scale_y' : 1.0,
    'scale_x' : 1.0,
    'depth' : 1.0,
    'cls' : 1.0,
    'recon' : 8.0,
    'virtual' : 0.0
}

virtual_loss_step = 15000

# virtual bound for testing
virtual_b = 0.2

# testing
test_batch_size = 16
test_model_path = "./logs/fruit2d/fruit2d_best.pkl"
