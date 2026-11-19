# DFO-CAM

The DFO-CAM implementation is not redistributed with this repository. It was obtained from the authors of the method, and the scripts here use it as an external component.

Scripts that evaluate DFO-CAM load it from:

```
third_party/dfo-cam/dfo-cam.py
```

Place the authors' implementation at that path before running those scripts. The file is expected to expose a `DFOCAM` class built on the `pytorch_grad_cam.base_cam.BaseCAM` interface, as in the authors' release. The comparison scripts also define a small compatibility wrapper around `get_cam_weights`; this wrapper only converts activations and gradients to single-precision NumPy arrays and does not change the method.

The faithfulness metrics (deletion/insertion, AD/AI/AG) are implemented in `code/cam_faithfulness/第十一步_忠实性指标.py` and are independent of the third-party code.

If you use DFO-CAM, cite the original work and follow the authors' terms of use.
