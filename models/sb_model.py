import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from collections import namedtuple
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch

class Vgg16Experimental(torch.nn.Module):
    """Everything exposed so you can play with different combinations for style and content representation"""
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.layer_names = ['relu1_1', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu4_1', 'relu4_3', 'relu5_1']
        self.content_feature_maps_indices = [0,1,2]
        self.style_feature_maps_indices = list(range(len(self.layer_names)))  # all layers used for style representation
        self.content_layer_names = [self.layer_names[x] for x in self.content_feature_maps_indices]
        self.style_layer_names = [self.layer_names[x] for x in self.style_feature_maps_indices]

        self.conv1_1 = vgg_pretrained_features[0]
        self.relu1_1 = vgg_pretrained_features[1]
        self.conv1_2 = vgg_pretrained_features[2]
        self.relu1_2 = vgg_pretrained_features[3]
        self.max_pooling1 = vgg_pretrained_features[4]
        self.conv2_1 = vgg_pretrained_features[5]
        self.relu2_1 = vgg_pretrained_features[6]
        self.conv2_2 = vgg_pretrained_features[7]
        self.relu2_2 = vgg_pretrained_features[8]
        self.max_pooling2 = vgg_pretrained_features[9]
        self.conv3_1 = vgg_pretrained_features[10]
        self.relu3_1 = vgg_pretrained_features[11]
        self.conv3_2 = vgg_pretrained_features[12]
        self.relu3_2 = vgg_pretrained_features[13]
        self.conv3_3 = vgg_pretrained_features[14]
        self.relu3_3 = vgg_pretrained_features[15]
        self.max_pooling3 = vgg_pretrained_features[16]
        self.conv4_1 = vgg_pretrained_features[17]
        self.relu4_1 = vgg_pretrained_features[18]
        self.conv4_2 = vgg_pretrained_features[19]
        self.relu4_2 = vgg_pretrained_features[20]
        self.conv4_3 = vgg_pretrained_features[21]
        self.relu4_3 = vgg_pretrained_features[22]
        self.max_pooling4 = vgg_pretrained_features[23]
        self.conv5_1 = vgg_pretrained_features[24]
        self.relu5_1 = vgg_pretrained_features[25]
        self.conv5_2 = vgg_pretrained_features[26]
        self.relu5_2 = vgg_pretrained_features[27]
        self.conv5_3 = vgg_pretrained_features[28]
        self.relu5_3 = vgg_pretrained_features[29]
        self.max_pooling5 = vgg_pretrained_features[30]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1_1(x)
        conv1_1 = x
        x = self.relu1_1(x)
        relu1_1 = x
        x = self.conv1_2(x)
        conv1_2 = x
        x = self.relu1_2(x)
        relu1_2 = x
        x = self.max_pooling1(x)
        x = self.conv2_1(x)
        conv2_1 = x
        x = self.relu2_1(x)
        relu2_1 = x
        x = self.conv2_2(x)
        conv2_2 = x
        x = self.relu2_2(x)
        relu2_2 = x
        x = self.max_pooling2(x)
        x = self.conv3_1(x)
        conv3_1 = x
        x = self.relu3_1(x)
        relu3_1 = x
        x = self.conv3_2(x)
        conv3_2 = x
        x = self.relu3_2(x)
        relu3_2 = x
        x = self.conv3_3(x)
        conv3_3 = x
        x = self.relu3_3(x)
        relu3_3 = x
        x = self.max_pooling3(x)
        x = self.conv4_1(x)
        conv4_1 = x
        x = self.relu4_1(x)
        relu4_1 = x
        x = self.conv4_2(x)
        conv4_2 = x
        x = self.relu4_2(x)
        relu4_2 = x
        x = self.conv4_3(x)
        conv4_3 = x
        x = self.relu4_3(x)
        relu4_3 = x
        x = self.max_pooling4(x)
        x = self.conv5_1(x)
        conv5_1 = x
        x = self.relu5_1(x)
        relu5_1 = x
        x = self.conv5_2(x)
        conv5_2 = x
        x = self.relu5_2(x)
        relu5_2 = x
        x = self.conv5_3(x)
        conv5_3 = x
        x = self.relu5_3(x)
        relu5_3 = x
        x = self.max_pooling5(x)
        # expose only the layers that you want to experiment with here
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_1, relu2_1, relu2_2, relu3_1, relu3_2, relu4_1, relu4_3, relu5_1)

        return out

def gram_matrix(x, normalize=True):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if normalize:
        gram /= ch * h * w
    return gram

class SCLossCriterion(nn.Module):
    def __init__(self, args, device):
        super(SCLossCriterion, self).__init__()

        self.device = device
        self._prepare_model(args)

        content_layer_names = ''
        for x in self.content_layer_names:
            content_layer_names = content_layer_names + x + ','
        style_layer_names = ''
        for x in self.style_layer_names:
            style_layer_names = style_layer_names + x + ','

    def _prepare_model(self, args):
        model = Vgg16Experimental(requires_grad=False)
        self.content_layer_names = model.content_layer_names
        self.style_layer_names = model.style_layer_names
        self.model = model.to(self.device).eval()

    def _content_loss(self, real_content, fake_content):
        real_content = real_content.detach()
        return nn.MSELoss(reduction='mean')(real_content, fake_content)/len(self.content_layer_names)

    def _style_loss(self, real_style, fake_style, weighted=True):
        real_style = real_style.detach()     # we dont need the gradient of the target
        size = real_style.size()

        if not weighted:
            weights = torch.ones(size=real_style.shape[0])
        else:
            # https://arxiv.org/pdf/2104.10064.pdf
            Nl = size[1] * size[2]  # C x C = C^2
            real_style_norm = torch.linalg.norm(real_style, dim=(1, 2))
            fake_style_norm = torch.linalg.norm(fake_style, dim=(1, 2))
            normalize_term = torch.square(real_style_norm) + torch.square(fake_style_norm)
            weights = Nl / normalize_term

        se = (real_style.view(size[0], -1) - fake_style.view(size[0], -1)) ** 2
        return (se.mean(dim=1) * weights).mean()

    def forward(self, content_img, style_img, fake_img):
        content_img_feature_maps = self.model(content_img)
        style_img_feature_maps = self.model(style_img)
        fake_img_feature_maps = self.model(fake_img)

        real_content_representation = [x for cnt, x in enumerate(content_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        real_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        fake_content_representation = [x for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        fake_style_representation = [gram_matrix(x) for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        # content loss
        content_loss = 0
        for i, layer in enumerate(self.content_layer_names):
            content_loss += self._content_loss(real_content_representation[i], fake_content_representation[i])

        # style loss
        style_loss = 0
        for i, layer in enumerate(self.style_layer_names):
            style_loss += self._style_loss(real_style_representation[i], fake_style_representation[i])

        return content_loss, style_loss



class SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for SB model
        """
        parser.add_argument('--mode', type=str, default="sb", choices='(FastCUT, fastcut, sb)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.mode.lower() == "sb":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE','SB','feature','content']
        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D','E']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            
    def data_dependent_initialize(self, data,data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data,data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()  
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        
        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()       
        
    def set_input(self, input,input2=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times])
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]
        
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):
                
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_1     = self.netG(Xt, time_idx, z)
                
                Xt2       = self.real_A if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_12    = self.netG(Xt2, time_idx, z)
                
                
                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1B = self.netG(XtB, time_idx, z)
            if self.opt.nce_idt:
                self.XtB = XtB.detach()
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()
                      
        
        z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
        z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        
        self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])
        
        self.fake = self.netG(self.realt,self.time_idx,z_in)
        self.fake_B2 =  self.netG(self.real_A_noisy2,self.time_idx,z_in2)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
            
        if self.opt.phase == 'test':
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1     = self.netG(Xt, time_idx, z)
                    
                    setattr(self, "fake_"+str(t+1), Xt_1)
                    
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs =  self.real_A.size(0)
        
        fake = self.fake_B.detach()
        std = torch.rand(size=[1]).item() * self.opt.std
        
        pred_fake = self.netD(fake,self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B,self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return self.loss_D
    def compute_E_loss(self):
        
        bs =  self.real_A.size(0)
        
        """Calculate GAN loss for the discriminator"""
        
        XtXt_1 = torch.cat([self.real_A_noisy,self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2,self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        
        return self.loss_E
    def compute_G_loss(self):
        bs =  self.real_A.size(0)
        tau = self.opt.tau
        
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std
        
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake,self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)
            
            bs = self.opt.batch_size

            ET_XY    = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY
            self.loss_SB += self.opt.tau*torch.mean((self.real_A_noisy-self.fake_B)**2)
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_content = self.content_loss()

        content_loss_value,style_loss_value = self.calculate_style_content_loss()
        self.loss_feature = content_loss_value
        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + self.opt.lambda_NCE*loss_NCE_both + 0.1*self.loss_feature + 0.1*self.loss_content
        # self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + self.opt.lambda_NCE * loss_NCE_both + 0.2*self.loss_feature


        return self.loss_G


    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        feat_k = self.netG(src, self.time_idx*0,z,self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
        
        
        
    def content_loss(self):

        L1_function = torch.nn.L1Loss()
        threshold_A = 170
        threshold_B = 170
        real_A_mean = torch.mean(self.real_A,dim=1,keepdim=True)
        fake_B_mean = torch.mean(self.fake_B,dim=1,keepdim=True)

        real_A_normal = (real_A_mean - (threshold_A / 127.5 - 1)) * 100
        fake_B_normal = (fake_B_mean - (threshold_B / 127.5 - 1)) * 100

        real_A_sigmoid = torch.sigmoid(real_A_normal)
        fake_B_sigmoid = 1 - torch.sigmoid(fake_B_normal)
        
        content_loss_A = L1_function( real_A_sigmoid , fake_B_sigmoid )

        content_loss_rate = 5*np.exp(-(self.opt.counter/(self.opt.data_size)))
        # content_loss_rate = 1
        content_loss = (content_loss_A)*content_loss_rate


        return content_loss

    def calculate_style_content_loss(self):
        # 实例化Vgg16模型和损失函数
        vgg_model = Vgg16Experimental(requires_grad=False)
        sc_loss_criterion = SCLossCriterion(args={}, device = self.device)

        # 计算损失
        content_loss, style_loss = sc_loss_criterion(self.real_A,self.real_B,self.fake_B)

        return content_loss, style_loss




