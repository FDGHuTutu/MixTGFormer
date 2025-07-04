# MixTGformer(Dual-stream Spatio-Temporal GCN-Transformer Network for 3D Human Pose Estimation)

The official code repository for the MixTGformer.
![mixtgformer.jpg](mixtgformer.jpg)

## Environment
* Python 3.8.10
* Pytorch 2.0.0
* CUDA 12.2
* For detailed environment configuration, see 'requirements.txt'(`pip install -r requirements.txt`)

## Dataset
### Human3.6M
* Preprocessing
  Download the fine-tuned Stacked Hourglass detections of [MotionBERT's](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) preprocessed H3.6M data [here](https://onedrive.live.com/?authkey=%21AMG5RlzJp%2D7yTNw&id=A5438CD242871DF0%21206&cid=A5438CD242871DF0&parId=root&parQt=sharedby&o=OneUp) and unzip it to 'data/motion3d'.
  Slice the motion clips by running the following python code in directory:data/preprocess
* Visualization
  Run the following command in the directory:data/preprocess
  python visualize.py --dataset h36m --sequence-number <AN ARBITRARY NUMBER>
