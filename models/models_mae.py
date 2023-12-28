# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torchaudio
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.pos_embed import get_2d_sincos_pos_embed

class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1, fmax_aug_range=1000, device='cuda'):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False).to(device),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]).to(device), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)


    def forward(self, x):
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        x = (x ** 2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax


        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec

    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length,
                                               self.hopsize
                                               )

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(128, 992), patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0, timem=0,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10, fmax_aug_range=2000,
                 mask_type='random', chunked_mask_mean_height=3, chunked_mask_mean_width=3,
                 specgram_type='mel', adaptive_hopsize=False, train_decoder_only=False, train_last_layer_only=False,
                 norm_file='mean_std.npy', device='cuda'):
        super().__init__()

        if mask_type != 'random':
            print('WARNING (MaskedAutoEncoderViT): mask_type is being set to "random"---other types are not yet supported.')
            mask_type = 'random'

        if specgram_type not in ['mel', 'stft']:
            specgram_type = 'mel'

        self.specgram_type = specgram_type
        self.adaptive_hopsize = adaptive_hopsize

        self.mask_type = mask_type
        self.chunked_mask_mean_height = chunked_mask_mean_height
        self.chunked_mask_mean_width = chunked_mask_mean_width

        assert not (train_decoder_only and train_last_layer_only), "At most one of {train_decoder_only, train_last_layer_only} can be set True."
        no_encoder_grad = train_decoder_only or train_last_layer_only

        # --------------------------------------------------------------------------
        # Mel Spectrogram
        self.mel = AugmentMelSTFT(
            n_mels=n_mels, sr=sr, win_length=win_length, hopsize=hopsize, n_fft=n_fft, freqm=freqm, timem=timem,
            htk=htk, fmin=fmin, fmax=fmax, norm=norm, fmin_aug_range=fmin_aug_range, fmax_aug_range=fmax_aug_range, device=device)
        mean_std_file = np.load(norm_file, allow_pickle=True).item()
        self.frame_mean = torch.Tensor(mean_std_file['frame_mean']).to(device)
        self.frame_std = torch.Tensor(mean_std_file['frame_std']).to(device)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        if no_encoder_grad:
            self.patch_embed.requires_grad_(False)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=(not no_encoder_grad))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        if no_encoder_grad:
            for block in self.blocks:
                block.requires_grad_(False)

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        if train_last_layer_only:
            self.decoder_embed.requires_grad_(False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        if train_last_layer_only:
            for block in self.decoder_blocks:
                block.requires_grad_(False)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.device = device

        self.initialize_weights()

    def stft_forward(self, x, x_lens=None, scale=1.5):
        hopsize = self.mel.hopsize

        Sx = -scale*torch.ones(x.shape[0], 1, self.patch_embed.img_size[0], self.patch_embed.img_size[1])

        # Each spectrogram may have a different hopsize, so we have to loop through
        # the samples in the batch
        for i in range(x.shape[0]):
            if self.adaptive_hopsize:
                # stft length is N // hopsize + 1
                N = x_lens[i] if x_lens is not None else x.shape[1]
                self.patch_embed.img_size[1]
                hopsize = (N + self.patch_embed.img_size[1] - 2) // (self.patch_embed.img_size[1] - 1)
            
            Sxi = 20*torch.log10(torch.abs(torch.stft(x[i, 0, :], 256, hop_length=hopsize, win_length=256,
                            return_complex=True, center=True, normalized=False, window=torch.hamming_window(256).to(x.device)) ** 2))
            Sxi = Sxi[:, :self.patch_embed.img_size[1]]

            # Normalize the spectrogram to fall within (-2, 2)
            Sxi = torch.maximum(Sxi, torch.max(Sxi) - 100)
            Sxi -= (torch.min(Sxi) + torch.max(Sxi))/2
            Sxi = Sxi / torch.max(Sxi) * scale

            Sx[i, 0, :, :Sxi.shape[1]] = Sxi[:-1, :]
        return Sx.type(torch.HalfTensor).to(x.device)
    
    def mel_forward(self, x, normalize=True):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        if normalize:
            x = (x - self.frame_mean[None, :, None]) / self.frame_std[None, :, None]
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2, 1)
        """
        assert imgs.shape[2] % self.patch_embed.grid_size[0] == 0 and imgs.shape[3] % self.patch_embed.grid_size[1] == 0

        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]
        p1 = self.patch_embed.patch_size[0]
        p2 = self.patch_embed.patch_size[1]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p1, w, p2))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p1*p2, 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2, 1)
        imgs: (N, 1, H, W)
        """
        p1 = self.patch_embed.patch_size[0]
        p2 = self.patch_embed.patch_size[1]
        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p1, w * p2))
        return imgs
    
    def mask_chunks_high_snr(self, x, x_unpatched, mask_ratio=0.1):
        def get_random_numbers():
            start_xs = torch.argsort(torch.rand((W*H,), device=x.device)) % W
            start_ys = torch.argsort(torch.rand((W*H,), device=x.device)) % H

            heights = torch.ones(W*H, dtype=torch.int32, device=x.device)
            heights.geometric_(1/avg_mask_height)
            heights = torch.minimum(heights, H - start_ys)

            widths = torch.ones(W*H, dtype=torch.int32, device=x.device)
            widths.geometric_(1/avg_mask_width)
            widths = torch.minimum(widths, W - start_xs)
        
            return start_xs, start_ys, widths, heights
        
        N = x.shape[0]
        D = x.shape[-1]
        H, W = self.patch_embed.grid_size

        avg_mask_height = self.chunked_mask_mean_height
        avg_mask_width = self.chunked_mask_mean_width

        n_masked = int(round(H*W*mask_ratio))
        ids_mask = torch.zeros(N, n_masked, dtype=torch.int32, device=x.device)
        ids_keep = torch.zeros(N, H*W-n_masked, dtype=torch.int32, device=x.device)
        masks = torch.ones(N, W*H, device=x.device)

        i = 0
        for k in range(N):
            mask = torch.ones(H, W, device=x.device)
            cur_mask_ratio = 0
            cutoff_coeff = 0.8
            cutoff = torch.min(x_unpatched) + (torch.max(x_unpatched) - torch.min(x_unpatched)) * cutoff_coeff
        
            start_xs, start_ys, widths, heights = get_random_numbers()

            while cur_mask_ratio <= mask_ratio:
                block = x_unpatched[k, 0, 16*start_ys[i]:16*(start_ys[i]+heights[i]), 16*start_xs[i]:16*(start_xs[i]+widths[i])]
                if torch.amax(block, (0,1)) >= cutoff:
                    mask[start_ys[i]:start_ys[i]+heights[i], start_xs[i]:start_xs[i]+widths[i]] = 0
                cur_mask_ratio = torch.sum(1 - mask) / (W*H)
                i += 1
                if i >= W*H:
                    cutoff_coeff *= 0.75
                    cutoff = torch.min(x_unpatched) + (torch.max(x_unpatched) - torch.min(x_unpatched)) * cutoff_coeff
                    i = 0
                    start_xs, start_ys, widths, heights = get_random_numbers()

            
            mask = mask.reshape(W*H)
            ids_mask[k, :] = torch.nonzero(1-mask, as_tuple=True)[0][:n_masked]
            masks[k, ids_mask[k, :]] = 0
            ids_keep[k, :] = torch.nonzero(masks[k, :], as_tuple=True)[0]

        ids_restore = torch.argsort(torch.cat((ids_keep, ids_mask), dim=1), dim=1)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D).type(torch.LongTensor).to(x.device))        

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking_block_size(self, x, mask_ratio, block_size=1):
        N, L, D = x.shape  # batch, length, dim
        H, W = self.patch_embed.grid_size

        W_rounded_up = (W + block_size - 1) // block_size * block_size
        H_rounded_down = H // block_size * block_size

        # rearrange the indices to group each block_size x block_size chunk
        # of patches, and only mask indices that fall into a full block
        idxs = torch.arange(H_rounded_down * W_rounded_up, dtype=torch.int64, device=x.device)
        block_num = idxs // (block_size ** 2)
        block_position = idxs % (block_size ** 2)

        row = block_size * (block_num // (W // block_size)) + block_position // block_size
        col = block_size * (block_num % (W // block_size)) + block_position % block_size
        rearranged_idxs = (row * W + col)[0:(W // block_size) * block_size * H_rounded_down]
        num_blocked_idxs = rearranged_idxs.shape[0]

        len_keep = int(num_blocked_idxs * (1 - mask_ratio)) + (H*W - num_blocked_idxs)
        rearranged_idxs = rearranged_idxs.reshape(rearranged_idxs.shape[0] // (block_size ** 2), block_size ** 2)

        noise = torch.rand(N, rearranged_idxs.shape[0], device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_shuffle = ids_shuffle.unsqueeze(-1).repeat(1, 1, block_size ** 2)

        ids_block_shuffle = torch.gather(rearranged_idxs.unsqueeze(0).repeat(N, 1, 1), 1, ids_shuffle).reshape(-1, num_blocked_idxs)

        unblocked_idxs = torch.zeros(H*W - num_blocked_idxs, dtype=torch.int64, device=x.device)

        if W % block_size:
            unblocked_idxs[0:(W%block_size)*H_rounded_down] = torch.arange(H_rounded_down*W).reshape(H_rounded_down, W)[:, -(W%block_size):].flatten()
        if H % block_size:
            unblocked_idxs[(W%block_size)*H_rounded_down:] = torch.arange(H*W).reshape(H, W)[H_rounded_down:, :].flatten()

        ids_block_shuffle = torch.cat((unblocked_idxs.unsqueeze(0).repeat(N, 1), ids_block_shuffle), dim=1)
        
        # now the rest is the same as in random_masking![]
        ids_restore = torch.argsort(ids_block_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_block_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # mask_block_size = int(torch.rand(1).item() * 4 // 1) + 1
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        target = target[:,:,:,0]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, lengths=None, mask_ratio=0.75, normalize=True):
        imgs = imgs.to(self.device)

        if self.specgram_type == 'mel':
            imgs = self.mel_forward(imgs, normalize=normalize)
        else:
            imgs = self.stft_forward(imgs, lengths)
            
        imgs = imgs[:, :, :self.patch_embed.img_size[0], :self.patch_embed.img_size[1]]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
