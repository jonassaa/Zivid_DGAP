# Zivid_DGAP
Dataset generator and data preprocessor for zivid pointclouds for use with FCGF and DGR (Chris Choy, NVIDIA)

## Usage Example

```
python Preprocess.py    --dir path/to/zivid/files \
                        --zivid_camera_file path/to/camerafile
                        --val_fraction 0.1
                        --include_normals False
                        --include_color False
                        --subsample by2x2
                        --dataset_output_name NameOfOutput
                        --zivid False
                        --no_compress False
```

Note that if the --zivid flag is set to False (as by default) the program will ignore subsampling and normals as it is not vailable without the zivid SDK.

