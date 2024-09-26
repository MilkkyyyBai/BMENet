import torch
import torch as th
import torch.nn as nn

from models.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    exists
)
from models.attention import SpatialTransformer

from models.unet import (
    TimestepEmbedSequential_c, ResBlock_dims, Downsample_c, AttentionBlock_c)

from models.unet_v2_conditioned import (
    TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
    ):
        super().__init__()

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv3d(in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])


        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels # 256
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        # self.output_blocks = nn.ModuleList([])
        # for level, mult in list(enumerate(channel_mult))[::-1]:
        #     for i in range(num_res_blocks + 1):
        #         ich = input_block_chans.pop()
        #         layers = [
        #             ResBlock(
        #                 ch + ich,
        #                 time_embed_dim,
        #                 dropout,
        #                 out_channels=model_channels * mult,
        #                 use_scale_shift_norm=use_scale_shift_norm,
        #             )
        #         ]
        #         ch = model_channels * mult
        #         if ds in attention_resolutions:
        #             dim_head = ch // num_heads
        #             layers.append(
        #                 AttentionBlock(
        #                     ch,
        #                     num_heads=num_heads_upsample,
        #                     num_head_channels=num_head_channels,
        #                 )
        #                 if not use_spatial_transformer
        #                 else SpatialTransformer(
        #                     ch,
        #                     num_heads,
        #                     dim_head,
        #                     depth=transformer_depth,
        #                     context_dim=context_dim,
        #                 )
        #             )
        #         if level and i == num_res_blocks:
        #             out_ch = ch
        #             layers.append(
        #                 ResBlock(
        #                     ch,
        #                     time_embed_dim,
        #                     dropout,
        #                     out_channels=out_ch,
        #                     use_scale_shift_norm=use_scale_shift_norm,
        #                     up=True,
        #                 )
        #                 if resblock_updown
        #                 else Upsample(ch, conv_resample, out_channels=out_ch)
        #             )
        #             ds //= 2
        #         self.output_blocks.append(TimestepEmbedSequential(*layers))
        #         self._feature_size += ch

        # self.out = nn.Sequential(
        #     Normalize(ch),
        #     nn.SiLU(),
        #     zero_module(nn.Conv3d(model_channels, out_channels, 3, padding=1)),
        # )
        # if self.predict_codebook_ids:
        #     self.id_predictor = nn.Sequential(
        #         Normalize(ch),
        #         nn.Conv3d(model_channels, n_embed, 1),
        #     )
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential_c(zero_module(conv_nd(3, channels, channels, 1, padding=0)))

    # hint ---> c_img context ------> c_text
    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []

        h = x.type(th.float32)
        
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

    # def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param context: conditioning plugged in via crossattn
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: an [N x C x ...] Tensor of outputs.
    #     """
    #     assert (y is not None) == (
    #         self.num_classes is not None
    #     ), "must specify y if and only if the model is class-conditional"
    #     assert timesteps is not None, "need to implement no-timestep usage"
    #     hs = []
    #     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #     emb = self.time_embed(t_emb)

    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)

    #     h = x
    #     for module in self.input_blocks:
    #         h = module(h, emb, context)
    #         hs.append(h)
    #     h = self.middle_block(h, emb, context)
    #     for module in self.output_blocks:
    #         h = th.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb, context)

    #     if self.predict_codebook_ids:
    #         # return self.out(h), self.id_predictor(h)
    #         return self.id_predictor(h)
    #     else:
    #         return self.out(h)
