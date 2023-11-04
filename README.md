[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Metnet3
Pytorch implementation of the model `MetNet-3` that utilizes a Unet -> MaxVit with topological embeddings

# Install
`pip install metnet3`


## Usage
```


```


## Architecture Overview

MetNet-3 is a neural network designed to process and predict spatial weather patterns with high precision. This sophisticated model incorporates a fusion of cutting-edge techniques including topographical embeddings, a U-Net backbone, and a modified MaxVit transformer to capture long-range dependencies. With a total of 227 million trainable parameters, MetNet-3 is at the forefront of meteorological modeling.

### Topographical Embeddings

Leveraging a grid of trainable embeddings, MetNet-3 can automatically learn and utilize topographical features relevant to weather forecasting. Each grid point, spaced with a stride of 4 km, is associated with 20 parameters. These embeddings are then bilinearly interpolated for each input pixel, enabling the network to effectively encode the underlying geography for each data point.

### Model Diagram

MetNet-3's architecture is complex, ingesting both high-resolution (2496 km² at 4 km resolution) and low-resolution (4992 km² at 8 km resolution) spatial inputs. The model processes these inputs through a series of layers and operations, as depicted in the following ASCII flow diagram:

```
Input Data
   │
   │ High-resolution inputs
   │ concatenated with current time
   │ (624x624x793)
   │
   ▼
 [Embed Topographical Embeddings]
   │
   ├─►[2x ResNet Blocks]───►[Downsampling to 8 km]
   │                            │
   │                            ├─►[Pad to 4992 km²]───►[Concatenate Low-res Inputs]
   │                            │
   ▼                            ▼
 [U-Net Backbone]            [2x ResNet Blocks]
   │                            │
   ├─►[Downsampling to 16 km]   │
   │                            │
   ▼                            │
 [Modified MaxVit Blocks]◄──────┘
   │
   │
 [Central Crop to 768 km²]
   │
   ├─►[Upsampling Path with Skip Connections]
   │
   │
 [Central Crop to 512 km²]
   │
   ├─►[MLP for Weather State Channels at 4 km resolution]
   │
   ├─►[Upsampling to 1 km for Precipitation Targets]
   │
   ▼
[Output Predictions]
```

#### Dense and Sparse Inputs

The model uniquely processes both dense and sparse inputs, integrating temporal information such as the time of prediction and the forecast lead time.

#### Target Outputs

MetNet-3 produces both categorical and deterministic predictions for various weather-related variables, including precipitation and surface conditions, using a combination of loss functions tailored to the nature of each target.

#### ResNet Blocks and MaxVit

Central to the network's ability to capture complex patterns are the ResNet blocks, which handle local interactions, and the MaxVit blocks, which facilitate global comprehension of the input data through attention mechanisms.

## Technical Specifications

- **Input Spatial Resolutions**: 4 km and 8 km
- **Output Resolutions**: From 1 km to 4 km depending on the variable
- **Embedding Stride**: 4 km
- **Topographical Embedding Parameters**: 20 per grid point
- **Network Parameters**: 227 million
- **Input Channels**: Various, including 617+1 channels from HRRR assimilation
- **Output Variables**: 6+617 for surface and assimilated state variables, respectively
- **Model Backbone**: U-Net with MaxVit transformer
- **Upsampling and Downsampling**: Implemented within the network to transition between different resolutions

## Low-Level Details and Optimization

Further technical details on architecture intricacies, optimization strategies, and hyperparameter selections are disclosed in Supplement B, providing an in-depth understanding of the model's operational framework.

This README intends to serve as a technical overview for researchers and engineers looking to grasp the functional composition and capabilities of MetNet-3. For implementation and collaboration inquiries, the supplementary materials should be referred to for comprehensive insights.


## Citation
```bibtex
@article{Andrychowicz2023DeepLF,
    title   = {Deep Learning for Day Forecasts from Sparse Observations},
    author  = {Marcin Andrychowicz and Lasse Espeholt and Di Li and Samier Merchant and Alexander Merose and Fred Zyda and Shreya Agrawal and Nal Kalchbrenner},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.06079},
    url     = {https://api.semanticscholar.org/CorpusID:259129311}
}

```


# License
MIT



