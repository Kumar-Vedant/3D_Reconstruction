# 3D Model Reconstruction

A pipeline to convert single-view 2D images into full 3D model meshes using Monocular Depth Estimation, written in Python.

# Features

- Uses DepthAnything-v2 from HuggingFace Transformers for depth estimation from a single image.
- Conversion of depth maps to 3D point clouds using Orthographic projection.
- Estimates normals from Point Clouds using Poisson Surface Reconstruction (PSR).
- Conversion to a full 3D mesh using Open3D

## Demo

<p align="center">
  <img src="images/img_0.jpg" width="300"/>
  <img src="demo/cyberpunk.gif" width="300"/>
</p>

<p align="center">
  <img src="images/img_1.jpg" width="300"/>
  <img src="demo/car.gif" width="300"/>
</p>

<p align="center">
  <img src="images/img_2.jpg" width="300"/>
  <img src="demo/citystreet.gif" width="300"/>
</p>

## Tech Stack

- Python
- Depth estimation: HuggingFace transformers, DepthAnything-v2
- Open3D

## License
