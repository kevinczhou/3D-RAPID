# 3D-RAPID
We present 3D-RAPID (<ins>3D</ins> <ins>R</ins>econstruction with an <ins>A</ins>rray-based <ins>P</ins>arallelized <ins>I</ins>maging <ins>D</ins>evice), a highly parallelized computational 3D video microscopy technique that can capture long 3D videos of freely behaving model organisms at throughputs exceeding 5 gigapixels/sec. 3D-RAPID captures 9x6=54 synchronized raw videos using a multi-camera array microscope (MCAM) architecture, whose frames are computationally fused to form video streams of globally consistent photometric composites and coregistered 3D height maps. This repository features 3D-RAPID's scalable 3D reconstruction algorithm, which employs self-supervised learning to decouple reconstruction time from the video length and number of cameras.

The camera calibration part of the 3D-RAPID is based on our earlier work: https://github.com/kevinczhou/mesoscopic-photogrammetry

For more details, see our arXiv preprint: https://arxiv.org/abs/2301.08351.

## Data
Due to the large size of these videos (~50 GB/video), we have provided just three, one of each organism (fruit flies, ants, zebrafish), which can be downloaded from [here](https://doi.org/10.7924/r4db86b1q). Each raw video file (`raw_video.nc`) has an associated calibration file (`calibration_dataset.nc`), consisting of a single synchronized snapshot of a flat patterned target.

Please contact us if you're interested in other videos whose results feature in the paper.

## Setting up your environment
We recommend using a TensorFlow Docker image from NVIDIA (release 20.12) and the included dockerfile to install the relevant packages. You'll have to set up [Docker and NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian). After that, download the docker image with TensorFlow release 20.12:
```
 docker pull nvcr.io/nvidia/tensorflow:20.12-tf2-py3
```
You can also try other versions, but note that TensorFlow <=2.2 has a bug involving `tf.RaggedTensor` and `tf.data` that was fixed in 2.3. We've also tested and confirmed that 2.5 works.

Then, `cd` into the 'docker' directory and build a custom image from the `dockerfile` within:

```
docker build -t tensorflow-nvidia-custom .
```
Finally, run the Docker image using `docker run`.

Alternatively, instead of using Docker, you can manually install the packages listed in the dockerfile (plus jupyter and tensorflow 2.3), but we found the NVIDIA Docker version to be faster than the conda version.
We used a 24-GB GPU (RTX 3090), but if your GPU has less memory, you may need to reduce the batch or patch size.

## Usage
Download the raw video files and the calibration file from TBD and place according to the following file path structures:
- `/data/fruit_flies/raw_video.nc`
- `/data/fruit_flies/calibration_dataset.nc`
- `/data/harvester_ants/raw_video.nc`
- `/data/harvester_ants/calibration_dataset.nc`
- `/data/zebrafish/raw_video.nc`
- `/data/zebrafish/calibration_dataset.nc`
- `/data/camera_calibration_initial_guess.mat`

Adjust `directory` in the jupyter notebooks as needed. 

Start with the `calibration.ipynb` notebook to optimize the camera parameters, which generates new calibration files. Then, run `training_and_inference.ipynb` to perform self-supervised learning on the raw video data, and to generate the photometric composites and coregistered 3D height maps. More detailed instructions are contained in the notebooks.
