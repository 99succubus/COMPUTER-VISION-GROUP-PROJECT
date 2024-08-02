import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov5.models.yolo import Model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.project(x)
        return x


class YOLOv5SegASPPModel(nn.Module):
    def __init__(self, num_classes, model_size='s'):
        super(YOLOv5SegASPPModel, self).__init__()
        self.yolo = Model(cfg=f'yolov5{model_size}-seg.yaml', ch=3, nc=num_classes)
        self.aspp = ASPP(in_channels=32, out_channels=32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        results = self.yolo(x)

        try:
            logger.info(f"YOLOv5 output structure: {type(results)}")
            logger.info(f"Number of elements in the tuple: {len(results)}")

            for i, item in enumerate(results):
                if isinstance(item, torch.Tensor):
                    logger.info(f"Tuple element {i} shape: {item.shape}")
                elif isinstance(item, list):
                    logger.info(f"Tuple element {i} is a list with {len(item)} elements")
                else:
                    logger.info(f"Tuple element {i} type: {type(item)}")

            # The segmentation mask is the second element of the tuple
            seg_features = results[1]

            if not isinstance(seg_features, torch.Tensor):
                raise AttributeError("Segmentation features are not a tensor")

            logger.info(f"Segmentation features shape: {seg_features.shape}")

            # Apply ASPP
            aspp_out = self.aspp(seg_features)

            # Final convolution
            output = self.final_conv(aspp_out)

            # Resize output to match input size
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)

            return output

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise


def get_model(in_channels, out_channels):
    model = YOLOv5SegASPPModel(num_classes=out_channels)

    # Load pre-trained weights
    try:
        ckpt = torch.load('yolov5s-seg.pt', map_location='cpu')
        state_dict = ckpt['model'].float().state_dict()

        # Filter out unnecessary keys and ignore size mismatches
        model_state_dict = model.yolo.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if
                               k in model_state_dict and v.shape == model_state_dict[k].shape}

        # Load the filtered state dict
        model.yolo.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loaded pre-trained YOLOv5s-seg weights (ignoring mismatched layers)")
    except Exception as e:
        logger.warning(f"Could not load pre-trained weights: {e}")

    return model


def print_model_summary(model):
    logger.info(f"Model structure:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")