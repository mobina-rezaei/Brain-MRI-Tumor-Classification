# Browser model files

This folder is the GitHub Pages site for the Brain MRI project.

## Required model artifact

The page expects a TensorFlow.js **LayersModel** at:

`docs/model/model.json`

and all binary weight shards referenced by that JSON in the same `docs/model/` folder.

The existing Keras model is not present in this repository, so the website cannot perform real predictions until the local file `outputs/brain_tumor_cnn.keras` is converted and its generated TensorFlow.js files are uploaded here.

The browser page mirrors the current Python inference setup as closely as possible:
- RGB input
- 224 × 224 resize (Keras `load_img` default nearest interpolation)
- float32 values divided by 255
- class order: glioma, meningioma, no tumor, pituitary

Before treating the browser result as equivalent, compare predictions and probabilities against the Python model on the same test images. Browser canvas decoding/resizing may introduce small differences.

This is an educational classification demo, not a medical diagnostic tool.
