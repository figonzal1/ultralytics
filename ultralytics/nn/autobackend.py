# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER, ROOT, yaml_load
from ultralytics.utils.checks import check_suffix, check_yaml
from ultralytics.utils.downloads import is_url


def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        # imagenet class codes, i.e. 'n01440764'
        if isinstance(names[0], str) and names[0].startswith("n0"):
            # human-readable names
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        try:
            return yaml_load(check_yaml(data))["names"]
        except Exception:
            pass
    # return default if above errors
    return {i: f"class{i}" for i in range(999)}


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            |-----------------------|-------------------|
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms.
    """

    @torch.no_grad()
    def __init__(
        self,
        weights="yolo11n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        Initialize the AutoBackend for inference.

        Args:
            weights (str | torch.nn.Module): Path to the model weights file or a module instance. Defaults to 'yolo11n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            batch (int): Batch-size to assume for inference.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        (
            pt,
        ) = [False]
        fp16 &= pt or nn_module  # FP16
        stride = 32  # default stride
        model, metadata, task = None, None, None

        # Set device
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if cuda and not any([nn_module, pt]):  # GPU dataloader formats
            device = torch.device("cpu")
            cuda = False

        # In-memory PyTorch model
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(
                model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # PyTorch
        elif pt:
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(
                model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # Any other format (unsupported)
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )

        # Load external metadata YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")

        elif not (pt or nn_module):
            LOGGER.warning(
                f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)
        names = check_class_names(names)

        # Disable gradients
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment,
                           visualize=visualize, embed=embed)

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(
                    im, training=False) if self.keras else self.model(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                details = self.input_details[0]
                # is TFLite quantized int8 or int16 model
                is_int = details["dtype"] in {np.int8, np.int16}
                if is_int:
                    scale, zero_point = details["quantization"]
                    # de-scale
                    im = (im / scale + zero_point).astype(details["dtype"])
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * \
                            scale  # re-scale
                    # if task is not classification, excluding masks (ndim=4) as well
                    if x.ndim == 3:
                        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                        if x.shape[-1] == 6:  # end-to-end model
                            x[:, :, [0, 2]] *= w
                            x[:, :, [1, 3]] *= h
                        else:
                            x[:, [0, 2]] *= w
                            x[:, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, 5::3] *= w
                                x[:, 6::3] *= h
                    y.append(x)
            # TF segment fixes: export is reversed vs ONNX export and protos are transposed
            if len(y) == 2:  # segment with (det, proto) output order reversed
                if len(y[1].shape) != 4:
                    # should be y = (1, 116, 8400), (1, 160, 160, 32)
                    y = list(reversed(y))
                if y[1].shape[-1] == 6:  # end-to-end model
                    y = [y[1]]
                else:
                    # should be y = (1, 116, 8400), (1, 32, 160, 160)
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

        # for x in y:
        #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
        if isinstance(y, (list, tuple)):
            # segments and names not defined
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):
                # y = (1, 32, 160, 160), (1, 116, 8400)
                nc = y[0].shape[1] - y[1].shape[1] - 4
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """
        Warm up the model by running one forward pass with a dummy input.

        Args:
            imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

        warmup_types = self.pt, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(
                *imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1):
                self.forward(im)  # warmup
