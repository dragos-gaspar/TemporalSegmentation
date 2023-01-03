#  SCCV - Temporal Segmentation

## How to run

```shell
pip3 install -r requirements.txt
python3 main.py [function]
```

The ```[function]``` argument can be either ```process```, ```train``` or ```predict```.

The paths in which user data is located can be modified in the ```config.py``` file.

## Architecture

The app uses a siamese network based on resnet-18, pretrained on imagenet, to extract features
from pairs of consecutive frames. The difference between the two feature maps produced
by the network is classified using an SVM.

![Architecture diagram](https://github.com/dragos-gaspar/TemporalSegmentation/blob/master/images/architecture.png?raw=true)

## Dataset

The dataset used for training and testing is the OpenVideoSceneDetectionDataset
https://research.ibm.com/haifa/projects/imt/video/Video_DataSetTable.shtml

The annotations were adjusted to match the exact frame timings of the transitions.
On the modified version of the dataset, the model obtains between 83% and 91% accuracy,
depending on the train-test split.