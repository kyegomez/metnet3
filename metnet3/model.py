
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn
from torch.nn import Module, ModuleList, Sequential
from torchvision.models.resnet import resnet50
from zeta.nn.modules.unet import Unet

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# helper classes


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(dim * mult)
    return Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout),
    )


# MBConv


class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes


class Attention(Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7, num_registers=1):
        super().__init__()
        assert num_registers > 0
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        rel_pos_indices = F.pad(
            rel_pos_indices,
            (num_registers, 0, num_registers, 0),
            value=num_rel_pos_bias,
        )
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # combine heads out

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MaxViT(Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3,
        num_register_tokens=4,
    ):
        super().__init__()
        assert isinstance(
            depth, tuple
        ), "depth needs to be tuple if integers indicating number of transformer blocks at that stage"
        assert num_register_tokens > 0

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride=2, padding=1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1),
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2**i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(
            zip(dim_pairs, depth)
        ):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample=is_first,
                    expansion_rate=mbconv_expansion_rate,
                    shrinkage_rate=mbconv_shrinkage_rate,
                )

                block_attn = Attention(
                    dim=layer_dim,
                    dim_head=dim_head,
                    dropout=dropout,
                    window_size=window_size,
                    num_registers=num_register_tokens,
                )
                block_ff = FeedForward(dim=layer_dim, dropout=dropout)

                grid_attn = Attention(
                    dim=layer_dim,
                    dim_head=dim_head,
                    dropout=dropout,
                    window_size=window_size,
                    num_registers=num_register_tokens,
                )
                grid_ff = FeedForward(dim=layer_dim, dropout=dropout)

                register_tokens = nn.Parameter(
                    torch.randn(num_register_tokens, layer_dim)
                )

                self.layers.append(
                    ModuleList(
                        [
                            conv,
                            ModuleList([block_attn, block_ff]),
                            ModuleList([grid_attn, grid_ff]),
                        ]
                    )
                )

                self.register_tokens.append(register_tokens)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce("b d h w -> b d", "mean"),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x):
        b, w = x.shape[0], self.window_size

        x = self.conv_stem(x)

        for (
            conv,
            (block_attn, block_ff),
            (grid_attn, grid_ff),
        ), register_tokens in zip(self.layers, self.register_tokens):
            x = conv(x)

            # block-like attention

            x = rearrange(x, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=w, w2=w)

            # prepare register tokens

            r = repeat(
                register_tokens, "n d -> b x y n d", b=b, x=x.shape[1], y=x.shape[2]
            )
            r, register_batch_ps = pack_one(r, "* n d")

            x, window_ps = pack_one(x, "b x y * d")
            x, batch_ps = pack_one(x, "* n d")
            x, register_ps = pack([r, x], "b * d")

            x = block_attn(x) + x
            x = block_ff(x) + x

            r, x = unpack(x, register_ps, "b * d")

            x = unpack_one(x, batch_ps, "* n d")
            x = unpack_one(x, window_ps, "b x y * d")
            x = rearrange(x, "b x y w1 w2 d -> b d (x w1) (y w2)")

            r = unpack_one(r, register_batch_ps, "* n d")

            # grid-like attention

            x = rearrange(x, "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=w, w2=w)

            # prepare register tokens

            r = reduce(r, "b x y n d -> b n d", "mean")
            r = repeat(r, "b n d -> b x y n d", x=x.shape[1], y=x.shape[2])
            r, register_batch_ps = pack_one(r, "* n d")

            x, window_ps = pack_one(x, "b x y * d")
            x, batch_ps = pack_one(x, "* n d")
            x, register_ps = pack([r, x], "b * d")

            x = grid_attn(x) + x

            r, x = unpack(x, register_ps, "b * d")

            x = grid_ff(x) + x

            x = unpack_one(x, batch_ps, "* n d")
            x = unpack_one(x, window_ps, "b x y * d")
            x = rearrange(x, "b x y w1 w2 d -> b d (w1 x) (w2 y)")

        return self.mlp_head(x)


class TopographicalEmbeddingLayer(nn.Module):
    """
    A PyTorch layer that implements topographical embeddings.
    """

    def __init__(self, grid_size, stride, dim):
        """
        Initializes the topographical embedding layer.

        Parameters:
            grid_size (tuple): The size of the grid (height, width).
            stride (int): The stride of the grid in kilometers.
            dim (int): The dimensionality of the embeddings.
        """
        super().__init__()
        self.grid_size = grid_size
        self.stride = stride
        self.dim = dim

        # Create an embedding tensor for the grid
        self.embeddings = nn.Parameter(torch.randn(*grid_size, dim))

    def forward(self, x, coords):
        """
        Forward pass of the topographical embedding layer.

        Parameters:
            x (Tensor): The input tensor.
            coords (Tensor): The coordinates for bilinear interpolation.

        Returns:
            Tensor: The output tensor with topographical embeddings added.
        """
        batch_size, _, height, width = x.size()

        coords = coords.squeeze(1)  # Squeeze out the extra dimension if it's present

        # Perform bilinear interpolation
        topographical_embeddings = F.grid_sample(
            self.embeddings.permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1),
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return torch.cat((x, topographical_embeddings), dim=1)


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers, batch normalization, and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the residual block.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        Forward pass of the residual block.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the residual block.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MetNet(nn.Module):
    """
    MetNet: A neural weather model for precipitation forecasting.
    """

    def __init__(self, dim, dim_head=32, n_channels=3, n_classes=10, **kwargs):
        super(MetNet, self).__init__()
        self.topo_embed = TopographicalEmbeddingLayer(
            grid_size=(624, 624), stride=1, dim=dim
        )

        # Use the Bottleneck blocks from ResNet-50
        self.res_blocks_high = self._make_resnet_layer()
        self.res_blocks_low = self._make_resnet_layer()

        self.unet_backbone = Unet(n_channels=n_channels, n_classes=n_classes)

        # Assuming MaxViT is defined elsewhere with appropriate parameters
        self.maxvit_blocks = MaxViT(
            num_classes=n_classes,
            dim=dim,
            depth=(2, 2, 2, 2),
            dim_head=dim_head,
            
        )

        self.central_crop_768 = Rearrange(
            "b c (h p1) (w p2) -> b c (h w) p1 p2", p1=768, p2=768
        )
        self.central_crop_512 = Rearrange(
            "b c (h p1) (w p2) -> b c (h w) p1 p2", p1=512, p2=512
        )

        self.mlp_weather = nn.Sequential(
            nn.Linear(768, 128),  # Example sizes, needs to be adjusted
            nn.ReLU(),
            nn.Linear(128, 4),  # Example sizes, needs to be adjusted
        )

        self.upsample_precip = nn.UpsamplingBilinear2d(scale_factor=4)

    def _make_resnet_layer(self):
        # Get the layers from a pre-trained ResNet-50
        model = resnet50(pretrained=True)
        # We take the layers except the fully connected layer
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, high_res_inputs, low_res_inputs, coords):
        high_res_inputs = self.topo_embed(high_res_inputs, coords)

        high_res_features = self.res_blocks_high(high_res_inputs)
        high_res_features = F.interpolate(
            high_res_features, scale_factor=1 / 8, mode="bilinear", align_corners=False
        )

        low_res_features = F.pad(
            low_res_inputs, pad=(0, 0, 0, 0, 3, 3, 3, 3), mode="reflect"
        )
        low_res_features = self.res_blocks_low(low_res_features)

        combined_features = torch.cat((high_res_features, low_res_features), dim=1)
        features_unet = self.unet_backbone(combined_features)

        features_downsampled = F.interpolate(
            features_unet, scale_factor=1 / 2, mode="bilinear", align_corners=False
        )
        features_vit = self.maxvit_blocks(features_downsampled)

        features_cropped_768 = self.central_crop_768(features_vit).view(
            features_vit.size(0), -1, 768, 768
        )
        features_upsampled = F.interpolate(
            features_cropped_768, scale_factor=2, mode="bilinear", align_corners=False
        )

        features_cropped_512 = self.central_crop_512(features_upsampled).view(
            features_upsampled.size(0), -1, 512, 512
        )
        weather_states = self.mlp_weather(features_cropped_512.flatten(1))

        precipitation = self.upsample_precip(
            weather_states.view(weather_states.size(0), -1, 32, 32)
        )

        return precipitation


