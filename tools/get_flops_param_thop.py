import argparse
from thop import profile
from thop import clever_format
from mmcv import Config
# from mmcv.cnn import get_model_complexity_info
# from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
from torchsummaryX import summary
from torchstat import stat
from mmseg.models import build_segmentor
from ptflops import get_model_complexity_info
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        'config',
        help='train config file path')
    # parser.add_argument(
    #     '--shape',
    #     type=int,
    #     nargs='+',
    #     default=[512, 512],
    #     help='input image size')
    args = parser.parse_args()
    return args




# def get_tr_flops(net, input_shape):
#     flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
#     _, H, W = input_shape
#     net = net.backbone
#     try:
#         stage1 = sra_flops(H // 4, W // 4,
#                            net.block1[0].attn.sr_ratio,
#                            net.block1[0].attn.dim,
#                            net.block1[0].attn.num_heads) * len(net.block1)
#         stage2 = sra_flops(H // 8, W // 8,
#                            net.block2[0].attn.sr_ratio,
#                            net.block2[0].attn.dim,
#                            net.block2[0].attn.num_heads) * len(net.block2)
#         stage3 = sra_flops(H // 16, W // 16,
#                            net.block3[0].attn.sr_ratio,
#                            net.block3[0].attn.dim,
#                            net.block3[0].attn.num_heads) * len(net.block3)
#         stage4 = sra_flops(H // 32, W // 32,
#                            net.block4[0].attn.sr_ratio,
#                            net.block4[0].attn.dim,
#                            net.block4[0].attn.num_heads) * len(net.block4)
#     except:
#         stage1 = sra_flops(H // 4, W // 4,
#                            net.block1[0].attn.squeeze_ratio,
#                            64,
#                            net.block1[0].attn.num_heads) * len(net.block1)
#         stage2 = sra_flops(H // 8, W // 8,
#                            net.block2[0].attn.squeeze_ratio,
#                            128,
#                            net.block2[0].attn.num_heads) * len(net.block2)
#         stage3 = sra_flops(H // 16, W // 16,
#                            net.block3[0].attn.squeeze_ratio,
#                            320,
#                            net.block3[0].attn.num_heads) * len(net.block3)
#         stage4 = sra_flops(H // 32, W // 32,
#                            net.block4[0].attn.squeeze_ratio,
#                            512,
#                            net.block4[0].attn.num_heads) * len(net.block4)

#     print(stage1 + stage2 + stage3 + stage4)
#     flops += stage1 + stage2 + stage3 + stage4
#     return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    # if len(args.shape) == 1:
        # input_shape = (1, 3, args.shape[0], args.shape[0])
    # elif len(args.shape) == 2:
        # input_shape = (3, ) + tuple(args.shape)
    # else:
        # raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    #cfg = Config.fromfile('~/anaconda3/access/segformer/SegFormer/local_configs/segformer_maxvit/B1/segformer.b1_maxvit_8winsize.512x512.ade.160k.py')
    #cfg = Config.fromfile('local_configs/segformer_tensorized_maxvit/B1/segformer_tensorized.b1_ttall_5cores_6rank.512x512.ade.160k.py')
    cfg.model.pretrained = None 
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    model(torch.randn(1,3,512,512).cuda())
    # print(model)
    #input = torch.randn(1, 3, 512, 512).cuda()
    #macs, params = profile(model, inputs=(input,))
    #macs, params = clever_format([macs, params], "%.3f")
    #print(macs)
    #summary(model, torch.zeros((1,)+input_shape).cuda())
    #stat(model, input_size=input_shape)
    #macs,params=get_model_complexity_info(model,input_shape,as_strings=True,print_per_layer_stat=True,verbose=True)
    #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # if hasattr(model, 'forward_dummy'):
    #     model.forward = model.forward_dummy
    # else:
    #     raise NotImplementedError(
    #         'FLOPs counter is currently not currently supported with {}'.
    #         format(model.__class__.__name__))

    # # from IPython import embed; embed()
    # if hasattr(model.backbone, 'block1'):
    #     print('#### get transformer flops ####')
    #     with torch.no_grad():
    #         flops, params = get_tr_flops(model, input_shape)
    # else:
    #     print('#### get CNN flops ####')
    #     flops, params = get_model_complexity_info(model, input_shape)
    a = 0
    # split_line = '=' * 30
    # print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
    #     split_line, input_shape, flops, params))
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')


if __name__ == '__main__':
    main()