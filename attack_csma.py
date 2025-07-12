import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import random

from image_cam import GradCAM
from torch.autograd import Variable
from image_cam_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, \
    find_squeezenet_layer
import pickle as pkl

import time
from timm.models import create_model
import numpy as np

import torchvision
from pytorch_msssim import SSIM

def mix_images_with_patches(img1, img2, mix_factor, patch_size=2):
    """
    混合两组图像，每个块随机从img1或img2中取。

    Args:
        img1 (torch.Tensor): 第一组图像，形状为 (N, C, H, W)。
        img2 (torch.Tensor): 第二组图像，形状为 (N, C, H, W)。
        patch_size (int): 每个块的大小。

    Returns:
        torch.Tensor: 混合后的图像，形状为 (N, C, H, W)。
    """
    device = img1.device
    N, C, H, W = img1.shape
    # patch_size = 32  # 每个块的大小设置为 16x16 2,4,8,14,16,28,32,56,112
    # 采样出 [1, 1, H/patch_size, W/patch_size] 大小的矩阵，这里是 (1, 1, 14, 14)
    bernoulli_samples = torch.bernoulli(torch.full((1, 1, H // patch_size, W // patch_size), mix_factor, device=device))

    # 将伯努利采样的结果扩展回原始块的形状 (14, 14) -> (14, 14, 16, 16)，每个采样决定一个 16x16 块的值
    expanded_mask = bernoulli_samples.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    # 将 expanded_mask 扩展为 (N, C, H, W)
    expanded_mask = expanded_mask.repeat(N, C, 1, 1)

    # 混合图像块
    mixed_img = expanded_mask * img1 + (1 - expanded_mask) * img2
    # torchvision.utils.save_image(expanded_mask, 'vis/patch-mask.png')
    # torchvision.utils.save_image(torch.full((1, 1, H, W), 0.7, device=device), 'vis/global-mask.png')
    # torchvision.utils.save_image(mixed_img[0], 'vis/patch-mix.png')
    # torchvision.utils.save_image(img1[0], 'vis/patch-img1.png')
    # torchvision.utils.save_image(img2[0], 'vis/patch-img2.png')
    # sys.exit()

    return mixed_img

def pixels_mix(images1, images2, mix_factor):

    # 确保两组图像的形状相同
    assert images1.shape == images2.shape, "两组图像的形状必须相同"
    device = images1.device
    # 获取图像的形状
    batch_size, channels, height, width = images1.shape

    mask = torch.bernoulli(torch.full((1, 1, height, width), mix_factor, device=device))
    # torchvision.utils.save_image(mask, 'vis/pixel-mask.png')
    # sys.exit()
    mask = mask.repeat(batch_size, channels, 1, 1)
    images_mixed = mask * images1 + (1 - mask) * images2

    # mask = torch.bernoulli(torch.full((batch_size, channels, height, width), mix_factor, device=device))
    # images_mixed = mask * images1 + (1 - mask) * images2

    # # 创建一个与图像相同大小的mask，随机生成0或1，0代表不替换，1代表替换
    # mask = torch.rand(batch_size, channels, height, width) < mix_factor
    # images_mixed = images1.clone()  # 先复制一份images1
    # images_mixed[mask] = images2[mask]  # 将mask中为True的像素替换掉

    return images_mixed



class Attack(object):
    """
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """

    def __init__(self, name, model=None):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        # mean and std values are used in pytorch pretrained models
        # they are also used in Kinetics-400.
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, *input):
        r"""
        It defines the computation performed at every call (attack forward).
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def _transform_perts(self, perts):
        dtype = perts.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype).cuda()
        std = torch.as_tensor(self.std, dtype=dtype).cuda()
        perts.div_(std[:, None, None])
        return perts

    def _transform_video(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype).cuda()
        std = torch.as_tensor(self.std, dtype=dtype).cuda()
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[:, None, None]).add_(mean[:, None, None])
        return video

    def _transform_video_ILAF(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype).cuda()
        std = torch.as_tensor(self.std, dtype=dtype).cuda()
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[None, :, None, None, None]).add_(mean[None, :, None, None, None])
        return video

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images


def get_vits():
    model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        in_chans=3,
        global_pool=None,
        scriptable=False)
    model.cuda()
    model.eval()
    return model


def get_model(model_name):
    '''
    ['alexnet', 'vgg', 'resnet', 'densenet', 'squeezenet']
    '''
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        # model.features[11/7/4/1]
    elif model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        # model.features[29/20/11/1]
    elif model_name == 'resnet':
        model = models.resnet101(pretrained=True)
        # model = models.resnet50(pretrained=True)
    elif model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        # model.features.denseblock1/2/3/4
        # model.features.transition1/2/3,norm5
    elif model_name == 'squeezenet':
        model = models.squeezenet1_1(pretrained=True)
        # model.features[12/9/6/3].expand3x3_activation
    model.cuda()
    model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         m.eval()
    return model


def get_models(model_name_lists):
    models = []
    for model_name in model_name_lists:
        model = get_model(model_name)
        models.append(model)
    return models


def get_GradCam(model_name_lists):
    gradcams = []
    for model_name in model_name_lists:
        model_dict = dict(type=model_name, arch=get_model(model_name), input_size=(224, 224))
        this_gradcam = GradCAM(model_dict, False)
        gradcams.append(this_gradcam)
    return gradcams

class I2V_MF(Attack):
    '''
    The proposed adaptive I2V with multiple models and layers.
    Parameters:
        model_name_lists: the surrogate image model names. For example, model_name_lists = ['resnet', 'vgg', 'squeezenet', 'alexnet']
        depths: the layers used in each model. For example,  depths = {'resnet':[2,3], 'vgg':[2,3], 'squeezenet':[2,3], 'alexnet':[2,3]}
        step_size: the learning rate.
    Return:
        image_inps: video adversarial example.
        used_time: the time during attacking.
        cost_saved: the cost values of all steps
    '''

    def __init__(self, model_name_lists, depths, step_size, mix_factor, mix_type, momentum=0, coef_CE=False, epsilon=16 / 255, steps=60):
        super(I2V_MF, self).__init__("I2V_MF")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.loss_info = {}
        self.depths = depths
        self.momentum = momentum
        self.coef_CE = coef_CE
        self.models = get_models(model_name_lists)
        self.model_names = model_name_lists
        self.mix_factor = mix_factor
        self.mix_type = mix_type

        self.coeffs = torch.ones(len(model_name_lists) * 2).cuda()
        # print ('using image models:', model_name_lists)


        for i in range(len(self.models)):
            self.models[i].train()
            for m in self.models[i].modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
            model_name = self.model_names[i]
            self._attention_hook(self.models[i], model_name)

    def _find_target_layer(self, model, model_name):
        used_depth = self.depths[model_name]
        if model_name == 'resnet':
            if isinstance(used_depth, list):
                return [getattr(model, 'layer{}'.format(this_depth))[-1] for this_depth in used_depth]
            else:
                return getattr(model, 'layer{}'.format(used_depth))[-1]
        elif model_name == 'alexnet':
            depth_to_layer = {1: 1, 2: 4, 3: 7, 4: 11}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'vgg':
            depth_to_layer = {1: 1, 2: 11, 3: 20, 4: 29}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'squeezenet':
            depth_to_layer = {1: 3, 2: 6, 3: 9, 4: 12}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]].expand3x3_activation

    def _attention_hook(self, model, model_name):
        self.gradients = dict()
        self.gradients['value'] = []
        self.activations = dict()
        self.activations['value'] = []

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] += [grad_output[0]]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] += [output]
            return None

        target_layer = self._find_target_layer(model, model_name)
        # print (target_layer)
        if isinstance(target_layer, list):
            for i in target_layer:
                i.register_forward_hook(forward_hook)
                i.register_backward_hook(backward_hook)
        else:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)

    def forward(self, videos, labels, video_names):
        batch_size = videos.shape[0]
        b, c, f, h, w = videos.shape
        videos = videos.cuda()
        labels = labels.cuda()
        self.weights = []
        image_inps = videos.permute([0, 2, 1, 3, 4])
        image_inps = image_inps.reshape(b * f, c, h, w)

        # define modifer that updated by optimizer.
        modif = torch.Tensor(b * f, c, h, w).fill_(0.01 / 255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.step_size)

        unnorm_videos = self._transform_video(image_inps.clone().detach(), mode='back')  # [0, 1]

        unnorm_videos = Variable(unnorm_videos, requires_grad=False)

        init_feature_maps = []
        for n in range(len(self.models)):
            this_feature_maps = []
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []
            _ = self.models[n](image_inps)
            for mm in range(len(self.activations['value'])):
                activations = self.activations['value'][mm]
                activations = Variable(activations, requires_grad=False)
                this_feature_maps.append(activations)
            init_feature_maps.append(this_feature_maps)

        begin = time.time()
        cost_saved = np.zeros(self.steps)
        # previous_cs_loss = torch.ones_like(self.coeffs)
        for i in range(self.steps):

            true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0,max=1)
            true_image = self._transform_video(true_image, mode='forward')  # norm

            losses = []
            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](true_image)
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b * f, -1)
                    init_dir = init_activations.view(b * f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir).mean()
                    # ssim_loss = SSIM(data_range=1, channel=activations.shape[1])
                    # this_loss = ssim_loss(activations, init_activations)
                    this_losses.append(this_loss)
                losses.append(torch.stack(this_losses))  # 2,32


            # 极端时序情况
            T = true_image.size(0)
            # 初始化新的顺序索引
            shuffled_ind = [i for i in range(T - 1)]  # 排除最后一帧
            random.shuffle(shuffled_ind)
            # 构造新的帧顺序：第0帧是最后一帧，其余为洗牌结果
            ind = [T - 1] + shuffled_ind

            # ind = [x for x in range(true_image.size(0))]
            # random.shuffle(ind)
            # self-mix
            selfmix_losses = []
            if self.mix_type == 'global':
                mix_imgs = self.mix_factor * true_image + (1 - self.mix_factor) * true_image[ind]
                # mix_imgs = 0.5 * true_image + 1.0 * true_image[ind] # not the linear combinations
            elif self.mix_type == 'patch':
                mix_imgs = mix_images_with_patches(true_image, true_image[ind], self.mix_factor)
            elif self.mix_type == 'pixel':
                mix_imgs = pixels_mix(true_image, true_image[ind], self.mix_factor)

            # mix_imgs = mix_images_with_patches(true_image, true_image[ind], self.mix_factor)
            # torchvision.utils.save_image(self._transform_video(mix_imgs[0].clone().detach(), mode='back'), 'vis/patch-mix.png')
            # mix_imgs = pixels_mix(true_image, true_image[ind], self.mix_factor)
            # torchvision.utils.save_image(self._transform_video(mix_imgs[0].clone().detach(), mode='back'), 'vis-pixel.png')
            # mix_imgs = self.mix_factor * true_image + (1-self.mix_factor) * true_image[ind]
            # torchvision.utils.save_image(self._transform_video(mix_imgs[0].clone().detach(), mode='back'), 'vis-global.png')
            # sys.exit()
            # mix_imgs = pixels_mix(true_image, true_image[ind], self.mix_factor)
            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](mix_imgs)
                # _ = self.models[n](self.mix_factor * true_image + (1-self.mix_factor) * true_image[ind])
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b * f, -1)
                    init_dir = init_activations.view(b * f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir).mean()
                    # ssim_loss = SSIM(data_range=1, channel=activations.shape[1])
                    # this_loss = ssim_loss(activations, init_activations)
                    this_losses.append(this_loss)
                selfmix_losses.append(torch.stack(this_losses))  # 2,32

            # 极端时序情况
            T = true_image.size(0)
            # 初始化新的顺序索引
            shuffled_ind = [i for i in range(T - 1)]  # 排除最后一帧
            random.shuffle(shuffled_ind)
            # 构造新的帧顺序：第0帧是最后一帧，其余为洗牌结果
            ind = [T - 1] + shuffled_ind

            # ind = [x for x in range(true_image.size(0))]
            # random.shuffle(ind)
            # cross-mix
            crossmix_losses = []
            if self.mix_type == 'global':
                mix_imgs = self.mix_factor * true_image + (1-self.mix_factor) * image_inps[ind]
                # mix_imgs = 0.5 * true_image + 1.0 * image_inps[ind]
            elif self.mix_type == 'patch':
                mix_imgs = mix_images_with_patches(true_image, image_inps[ind], self.mix_factor)
            elif self.mix_type == 'pixel':
                mix_imgs = pixels_mix(true_image, image_inps[ind], self.mix_factor)

            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](mix_imgs)
                # _ = self.models[n](self.mix_factor * true_image + (1-self.mix_factor) * image_inps[ind])
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b * f, -1)
                    init_dir = init_activations.view(b * f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir).mean()
                    # ssim_loss = SSIM(data_range=1, channel=activations.shape[1])
                    # this_loss = ssim_loss(activations, init_activations)
                    this_losses.append(this_loss)
                crossmix_losses.append(torch.stack(this_losses))  # 2,32

            adv_cost = torch.sum(torch.stack(losses))
            self_cost = torch.sum(torch.stack(selfmix_losses))
            cross_cost = torch.sum(torch.stack(crossmix_losses))
            cost = (adv_cost + self_cost + cross_cost) / 3.0
            # cost = (adv_cost + self_cost) / 2.0  # ablation
            # cost = adv_cost

            # print(cost)
            # print(cost.item(), adv_cost.item(), self_cost.item(), cross_cost.item())
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            cost_saved[i] = cost.detach().item()

            for ind, vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        used_time = time.time() - begin

        true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        image_inps = self._transform_video(true_image, mode='forward')
        image_inps = image_inps.reshape(b, f, c, h, w)
        image_inps = image_inps.permute([0, 2, 1, 3, 4])
        return image_inps



class AENS_I2V_MF(Attack):
    '''
    The proposed adaptive I2V with multiple models and layers.
    Parameters:
        model_name_lists: the surrogate image model names. For example, model_name_lists = ['resnet', 'vgg', 'squeezenet', 'alexnet']
        depths: the layers used in each model. For example,  depths = {'resnet':[2,3], 'vgg':[2,3], 'squeezenet':[2,3], 'alexnet':[2,3]}
        step_size: the learning rate.
    Return:
        image_inps: video adversarial example.
        used_time: the time during attacking.
        cost_saved: the cost values of all steps
    '''

    def __init__(self, model_name_lists, depths, step_size, mix_factor, mix_type, momentum=0, coef_CE=False, epsilon=16 / 255, steps=60):
        super(AENS_I2V_MF, self).__init__("AENS_I2V_MF")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.loss_info = {}
        self.depths = depths
        self.momentum = momentum
        self.coef_CE = coef_CE
        self.models = get_models(model_name_lists)
        self.model_names = model_name_lists
        self.mix_factor = mix_factor
        self.mix_type = mix_type

        self.coeffs = torch.ones(len(model_name_lists) * 2).cuda()
        # print ('using image models:', model_name_lists)

        for i in range(len(self.models)):
            self.models[i].train()
            for m in self.models[i].modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
            model_name = self.model_names[i]
            self._attention_hook(self.models[i], model_name)

    def _find_target_layer(self, model, model_name):
        used_depth = self.depths[model_name]
        if model_name == 'resnet':
            if isinstance(used_depth, list):
                return [getattr(model, 'layer{}'.format(this_depth))[-1] for this_depth in used_depth]
            else:
                return getattr(model, 'layer{}'.format(used_depth))[-1]
        elif model_name == 'alexnet':
            depth_to_layer = {1: 1, 2: 4, 3: 7, 4: 11}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'vgg':
            depth_to_layer = {1: 1, 2: 11, 3: 20, 4: 29}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'squeezenet':
            depth_to_layer = {1: 3, 2: 6, 3: 9, 4: 12}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]].expand3x3_activation

    def _attention_hook(self, model, model_name):
        self.gradients = dict()
        self.gradients['value'] = []
        self.activations = dict()
        self.activations['value'] = []

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] += [grad_output[0]]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] += [output]
            return None

        target_layer = self._find_target_layer(model, model_name)
        # print (target_layer)
        if isinstance(target_layer, list):
            for i in target_layer:
                i.register_forward_hook(forward_hook)
                i.register_backward_hook(backward_hook)
        else:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)

    def forward(self, videos, labels, video_names):
        batch_size = videos.shape[0]
        b, c, f, h, w = videos.shape
        videos = videos.cuda()
        labels = labels.cuda()
        self.weights = []
        image_inps = videos.permute([0, 2, 1, 3, 4])
        image_inps = image_inps.reshape(b * f, c, h, w)

        # define modifer that updated by optimizer.
        modif = torch.Tensor(b * f, c, h, w).fill_(0.01 / 255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.step_size)

        unnorm_videos = self._transform_video(image_inps.clone().detach(), mode='back')  # [0, 1]

        unnorm_videos = Variable(unnorm_videos, requires_grad=False)

        init_feature_maps = []
        for n in range(len(self.models)):
            this_feature_maps = []
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []
            _ = self.models[n](image_inps)
            for mm in range(len(self.activations['value'])):
                activations = self.activations['value'][mm]
                activations = Variable(activations, requires_grad=False)
                this_feature_maps.append(activations)
            init_feature_maps.append(this_feature_maps)

        begin = time.time()
        cost_saved = np.zeros(self.steps)
        previous_cs_loss = torch.ones_like(self.coeffs)
        for i in range(self.steps):
            # self.gradients = dict()
            # self.gradients['value'] = []
            # self.activations = dict()
            # self.activations['value'] = []

            # update coeff
            self.coeffs = torch.softmax(torch.softmax(previous_cs_loss, dim=0) + self.momentum * self.coeffs, dim=0)
            self.weights.append(self.coeffs.clone().cpu().numpy())
            # print (self.coeffs.clone().cpu().numpy())
            true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0,max=1)
            true_image = self._transform_video(true_image, mode='forward')  # norm

            # 极端时序情况
            T = true_image.size(0)
            # 初始化新的顺序索引
            shuffled_ind = [i for i in range(T - 1)]  # 排除最后一帧
            random.shuffle(shuffled_ind)
            # 构造新的帧顺序：第0帧是最后一帧，其余为洗牌结果
            ind = [T - 1] + shuffled_ind

            # ind = [x for x in range(true_image.size(0))]
            # random.shuffle(ind)
            # self-mix
            selfmix_losses = []
            if self.mix_type == 'global':
                mix_imgs = self.mix_factor * true_image + (1 - self.mix_factor) * true_image[ind]
                # mix_imgs = 0.5 * true_image + 1.0 * true_image[ind]
            elif self.mix_type == 'patch':
                mix_imgs = mix_images_with_patches(true_image, true_image[ind], self.mix_factor)
            elif self.mix_type == 'pixel':
                mix_imgs = pixels_mix(true_image, true_image[ind], self.mix_factor)

            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](mix_imgs)
                # _ = self.models[n](self.mix_factor * true_image + (1-self.mix_factor) * true_image[ind])
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b * f, -1)
                    init_dir = init_activations.view(b * f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir)
                    # ssim_loss = SSIM(data_range=1, channel=activations.shape[1])
                    # this_loss = ssim_loss(activations, init_activations)

                    this_losses.append(this_loss)
                selfmix_losses.append(torch.stack(this_losses))  # 2,32

            # 极端时序情况
            T = true_image.size(0)
            # 初始化新的顺序索引
            shuffled_ind = [i for i in range(T - 1)]  # 排除最后一帧
            random.shuffle(shuffled_ind)
            # 构造新的帧顺序：第0帧是最后一帧，其余为洗牌结果
            ind = [T - 1] + shuffled_ind

            # ind = [x for x in range(true_image.size(0))]
            # random.shuffle(ind)
            # cross-mix
            crossmix_losses = []
            if self.mix_type == 'global':
                mix_imgs = self.mix_factor * true_image + (1-self.mix_factor) * image_inps[ind]
                # mix_imgs = 0.5 * true_image + 1.0 * image_inps[ind]
            elif self.mix_type == 'patch':
                mix_imgs = mix_images_with_patches(true_image, image_inps[ind], self.mix_factor)
            elif self.mix_type == 'pixel':
                mix_imgs = pixels_mix(true_image, image_inps[ind], self.mix_factor)

            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](mix_imgs)
                # _ = self.models[n](self.mix_factor * true_image + (1-self.mix_factor) * image_inps[ind])
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b * f, -1)
                    init_dir = init_activations.view(b * f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir)
                    # ssim_loss = SSIM(data_range=1, channel=activations.shape[1])
                    # this_loss = ssim_loss(activations, init_activations)

                    this_losses.append(this_loss)
                crossmix_losses.append(torch.stack(this_losses))  # 2,32

            adv_losses = []
            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](true_image)
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b * f, -1)
                    init_dir = init_activations.view(b * f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir)

                    # ssim_loss = SSIM(data_range=1, channel=activations.shape[1])
                    # this_loss = ssim_loss(activations, init_activations)
                    this_losses.append(this_loss)

                adv_losses.append(torch.stack(this_losses))  # 2,32

            losses = []
            for n in range(len(adv_losses)):
                loss = (adv_losses[n] + selfmix_losses[n] + crossmix_losses[n]) / 3.0  # i2vmf-self-cross
                # loss = (adv_losses[n] + selfmix_losses[n]) / 2.0 # ablation
                # loss = adv_losses[n]  # i2vmf
                losses.append(loss)

            used_coeffs = torch.unsqueeze(self.coeffs, dim=1)  # (lens_model*2) * 1
            # print(used_coeffs.size())
            # print(torch.cat(losses, dim=0).size())
            each_features_loss = torch.sum(used_coeffs * torch.cat(losses, dim=0), dim=1)  # 4*32
            cost = torch.mean(each_features_loss)

            if self.coef_CE:
                previous_cs_loss = each_features_loss.clone().detach()
            else:
                updated_features_loss = torch.sum(torch.cat(losses, dim=0).clone().detach(), dim=1)
                previous_cs_loss = updated_features_loss.clone().detach()

            # update previous_cs_loss

            # print (previous_cs_loss.clone().cpu().numpy())
            # print(cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            cost_saved[i] = cost.detach().item()

            for ind, vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        used_time = time.time() - begin

        true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0,
                                 max=1)
        image_inps = self._transform_video(true_image, mode='forward')
        image_inps = image_inps.reshape(b, f, c, h, w)
        image_inps = image_inps.permute([0, 2, 1, 3, 4])
        return image_inps
