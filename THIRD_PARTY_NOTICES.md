# Third-party models, data and code

Pigeon's own code is MIT (see `LICENSE`), except `cockpit/`, which is AGPL-3.0 (see `cockpit/LICENSE`). These pieces come from others and keep their own terms.

## Models

- **YOLOv8n** (Ultralytics), AGPL-3.0. The cockpit prunes, quantizes and fine-tunes it through the `ultralytics` package, and the site's detector cage serves a converted copy of its weights (`site/public/coop/yolov8n/weights.bin`). Source and license: https://github.com/ultralytics/ultralytics. Under AGPL-3.0, anyone who uses those weights over a network is entitled to their source, which is this repository and the Ultralytics repository above.
- **MobileNetV3-Small** ImageNet weights from torchvision (BSD-3-Clause), the starting point that `train/` specializes into the Pigeon model (`site/public/model/`). https://github.com/pytorch/vision

## Data

- **Open Images V7** (Google). The box annotations are CC BY 4.0, and the images are listed as CC BY 2.0 by their authors. The site's test set (`site/public/testset/`) is 500 of those images. Their per-image attribution ships in `testset.json`. https://storage.googleapis.com/openimages/web/index.html
- **COCO** (bird images for the cockpit's fine-tuning and eval sets, built by `cockpit/build_finetune_set.py`; not committed). The annotations are CC BY 4.0, and the images remain under their Flickr licenses. https://cocodataset.org/#termsofuse

## Libraries the site loads

ONNX Runtime Web (MIT), TensorFlow.js (Apache-2.0) and Pyodide (MPL-2.0), each under its own license as distributed through npm or its CDN.
