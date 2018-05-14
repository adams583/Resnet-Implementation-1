# Resnet Implementation 1
An implementation of a residual neural network in TensorFlow to classify CIFAR-10 images based off of the research papers "Identity Mappings in Deep Residual Networks" and "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun

### General Layout
- Original Residual Units: (3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU) + Identity Mapping
- Latest Residual Units (Better Performance): (BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 3x3 conv) + Identity Mapping
- SGD with mini-batching 
- In the [research paper,](https://arxiv.org/pdf/1512.03385.pdf) a single conv layer is followed by a bunch of residual units, which are then followed by a dense layer and output
- The width and height dimensions are halved every few residual units, and the number of filters is usually doubled when this happens (to preserve complexity)
#### What I want to implement
- Depth 16 network, downsampling performed by conv layers with stride 2
- 7x7 conv, 64 -> pool/2 -> Res unit (3x3 conv 64) x 2 -> Res unit (3x3 conv 128) x 2 -> Res unit (3x3 conv 256) x 3 -> avg pool -> dense layer
#### Considerations
- Wide networks that are shallower (more features in conv layers) have been shown to work better than deep networks that are narrower (I think 64 filters is considered relatively wide for a 16 depth network)
