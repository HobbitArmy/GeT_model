# -*- coding: utf-8 -*-
"""
Created on 2022/7/6 23:22

Load an grape leaf image and test what the symptom is.

@author: LU
"""

from PIL import Image
import torch
from torchvision import transforms
from timm import create_model
from timm.models.registry import register_model
from timm.models.vision_transformer_hybrid import _create_vision_transformer_hybrid, _cfg
from timm.models.vision_transformer import checkpoint_filter_fn

# %% 1. Define model, load weights, load image2tensor
@register_model
def gr8t_g100_pd8(pretrained=False, patch_size=8, mydepth=8, embed_dim=168, img_size: int = 384,
                  backbone_name='ghostnet_100', backbone_out_indices=1, num_heads=3, **kwargs):
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 384 x 384.
    """
    backbone = create_model(backbone_name, pretrained=pretrained, features_only=True,
                                 out_indices=[backbone_out_indices])
    model_kwargs = dict(patch_size=patch_size, embed_dim=embed_dim, depth=mydepth,
                        img_size=img_size, num_heads=num_heads, **kwargs)
    cfg = _cfg(
        first_conv='patch_embed.backbone.conv', input_size=(3, img_size, img_size), crop_pct=1.0)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_ghost_p8_d_s', backbone=backbone, pretrained=False, default_cfg=cfg, **model_kwargs)
    return model

# %% 2. Load pretrained image
def load_my_pth_dict(model_ft, pth_state_dict, num_classes=11):
    pth_state_dict = checkpoint_filter_fn(pth_state_dict, model_ft)
    model_dict = model_ft.state_dict()
    pretrained_dict = {k: v for k, v in pth_state_dict.items() if k in model_dict}
    pretrained_dict['head.weight'] = pretrained_dict['head.weight'][:num_classes]
    pretrained_dict['head.bias'] = pretrained_dict['head.bias'][:num_classes]
    model_dict.update(pretrained_dict)
    model_ft.load_state_dict(pretrained_dict)
    return model_ft


if __name__ == '__main__':
    device = 'cpu'
    # 1. To set path nad class name
    model_path = './model_pth/imgN_gr8t_g100_p2_d3_oi2_288_4.pth'
    image_path = './hold_out_test/healthy.png'
    glpd_cls = ['black_root', 'blight', 'colom', 'defi', 'esca', 'healthy',
                'leaf_hopper', 'mildew', 'powd', 'spot', 'viral']
    # 2. Define the model
    model_ft = gr8t_g100_pd8(pretrained=False, patch_size=2, mydepth=3,
                             embed_dim=168, img_size=384, backbone_out_indices=2, num_classes=11)
    # 3. Load pre-trained model
    pth_state_dict = torch.load(model_path).module.state_dict()
    model = load_my_pth_dict(model_ft, pth_state_dict)

    # 4. Input Image
    image = Image.open(image_path)
    trans = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip()
    ])
    input_tensor = trans(image.convert('RGB')).unsqueeze(0)

    # 5. Run model and show prediction
    with torch.no_grad():
        model = model.to(device)
        out = model(input_tensor)
    print('Top1 prediction:', glpd_cls[out.argmax().item()])

