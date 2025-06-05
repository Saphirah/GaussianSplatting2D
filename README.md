# GaussianSplatting2D

A Python library for image approximation via adaptive 2D Gaussian splatting. Iteratively optimizes a set of splats to reconstruct a target image with minimal memory footprint and allows quantization for shader use.

## Features

- Adaptive sample splitting and scaling based on gradient magnitude  
- Quantized export for ShaderToy as `uvec4` arrays  
- Checkpointing and multi-bit quantization of splat parameters

## Example Shader

https://www.shadertoy.com/view/Wc33W4

## Installation

Install PyTorch
```bash
pip install -r requirements.txt
```

## Usage

1. Place your target image in the project root.  
2. Adjust parameters in the `__main__` section of `main.py` if needed (epochs, sample count, thresholds).  
3. Run training:

   ```bash
   python main.py
   ```

4. Interrupt with `Ctrl+C` to save the latest checkpoint.  
5. Automatically copies splat parameters to clipboard, for use in ShaderGraph

## Configuration Parameters

- `num_samples` – Initial splat count  
- `num_max_samples` – Maximum allowed splats  
- `sigma_thre` – Minimum standard deviation for splitting  
- `grad_thre` – Gradient threshold for adaptive refinement  
- `num_epoch` – Total training epochs  
- `num_iter_per_epoch` – Iterations per epoch  

## Output

- Checkpoints saved under `training/<image_basename>/<image_basename>.pt`  
- Quantized variants:  
  - `16-bit-quantized-<name>.pt`  
  - `8-bit-quantized-<name>.pt`  
- Reconstruction and target side-by-side images in `training/<name>/images/`  
- Clipboard export of `uvec4` arrays for ShaderToy

## License

This project is released under the MIT License.
