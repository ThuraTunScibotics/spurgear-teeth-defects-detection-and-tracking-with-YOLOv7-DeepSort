# spurgear-teeth-defects-detection-and-counting-YOLOv7

|     Detection       |      Detection + Tracking (Counting)       | 
|        :---:          |               :---:          | 
|          -            |    	<img src="https://github.com/ThuraTunScibotics/spurgear-teeth-defects-detection-and-tracking-with-YOLOv7-DeepSort/blob/main/assets/detect_track_count_2.gif" height="50%" width="50%">	     | 
|			|				|
|           -            |    <img src="https://github.com/ThuraTunScibotics/spurgear-teeth-defects-detection-and-tracking-with-YOLOv7-DeepSort/blob/main/assets/detect_track_count_1.gif" height="50%" width="50%">       | 


### Overview


-------------------------------------------------------------
### Contents

* [Inference Speed of _Detection_ and _Detection plus Tracking_](#inference-speed-of-detection-and-detection-plus-tracking)  
* [Dataset Preparation](#dataset-preparation)  

-------------------------------------------------------------
### Inference Speed of _Detection_ and _Detection plus Tracking_

**System Specification**
* **OS** : Ubuntu 22.04 LTS
* **CPU** : Intel Core i5 6200U 2.30GHz * 4
* **RAM** : DDR4, 4GB 
* **GPU** : NVIDIA GeForce 920Mx, 2GB
* **GPU Drvier & CUDA Version** : 515.105.01 | 11.7

|     Inferencing       |      time per frame       | 
|        :---:          |     :---:      | 
| YOLOv7 (Detection)                |       -        | 
| YOLOv7 + Deepsort (Detection+Tracking)    |       -        | 

-----------------------------------------------------------
### Dataset Preparation
The dataset that is being trained on is custom dataset and you can get here-> (link soon)

#### Step-1: Data Annotation

* Installing [labelImg](https://github.com/heartexlabs/labelImg) Data Annotation Tool;
```
git clone https://github.com/heartexlabs/labelImg.git   # first clone the original Repo on specific directory

conda create -n LabelImg python=3.9.13

conda activate LabelImg
pip install labelimg
```
* Creating pre-defined `classes.txt` file in images(dataset) directory with our interest class-names;
```
Good Teeth
Defect Teeth
```
* Starting Data Annotation for YOLOv7;

```
cd [the cloned LabelImg Repsitory]

python labelImg.py [/path/to/images-datasets] [/path/to/classes.txt]
```
`Make sure YOLO type` > `Create RectBox` > `Save` > `Next`

#### Step-2: Data Splitting
```
python split_dataset.py --folder [annotated-dataset-folder-name] --train [80] --validation [10] --test [10] --dest [/desinated-path-name-dataset]
```
#### Step-3: Making configuration file
* The parameters in [gear_detect.yaml](https://github.com/ThuraTunScibotics/spurgear-teeth-defects-detection-and-tracking-with-YOLOv7-DeepSort/blob/main/data/gear_detect.yaml) are needed to change.
	* add the training, validation and testing data path to `train`, `val`, and `test` respectively 
	* add the numbers of class to class to `nc`, and class names to `names` parameters
* The configuration parameter `nc` number of class in [yolov7-gear-detect.yaml](https://github.com/ThuraTunScibotics/spurgear-teeth-defects-detection-and-tracking-with-YOLOv7-DeepSort/blob/main/cfg/training/yolov7-gear-detect.yaml) is also needed to changed, and we chage to `2` for this case.
----------------------------------------------------

### Installation
Step-1: Install `NVIDIA GPU driver` & `CUDA` if it's not already installed on the machine (in Ubuntu)
```
# Installing the driver with your desired verion number
	sudo apt install libnvidia-common-515
	sudo apt-get -y install libnvidia-gl-515
	sudo apt install nvidia-driver-515-server
	
# After installing with all above commands, reboot ubuntu
	sudo reboot
	
# After reboot, check the smi table
	nvidia-smi	
```

Step-2: Creating environment & install dependencies
* Install [anaconda](https://www.anaconda.com/), create conda environment
```
conda create -n spur-gear-teeth-defects python=3.10.6 
```

* Install PyTorch & required packages
```
conda activate spur-gear-teeth-defects

pip3 install torch torchvision torchaudio  # Recommend: Go to pytorch.org and get the command according to desirements

pip install -r requirements_gpt.txt
```
For training YOLOv7 object detection model and running demo, we need to install [YOLOv7 Weights](https://drive.google.com/file/d/19zlzP6T3aBoR7ZvTXbrWaGquBa9Tx5q0/view?usp=sharing) in the the working directory. If you want to train with other YOLOv7 weights, you can check [here](). To do inferencing with my trained model, download it from [here](https://drive.google.com/file/d/1yfpfoOt8XkpwrSOkvZteVg3_iJoq51Gl/view?usp=share_link), and paste it to the directory `runs/train/yolov7-defect-detect-three2/weights/`

----------------------------------------------

### Run Demo
For running the demo of the project using the trained model, three main cases of demo can be ran including normal defect detection, detection and tracking of the product with counting inspection results, and the results can be saved as CSV file for visualizing inspection charts in Excel. The demo result files are saved in `runs/detect/` path after running each python script.

#### Run for detection
To run detection of spur gear teeth defects, we need to run `detect.py` python script.
* Argument lists;
   * `--weights` - path to the trained model, runs/train/.../last.pt
   * `--source` - path to inference data source
   * `--view-img` - display detecting frame
	
```
python detect.py --weights runs/train/yolov7-defect-detect-three2/weights/last.pt --source inference/videos/video_6.mp4 --view-img
```

#### Run for detection + tracking -> Counting
To run counting the inspecting result of gear product through `detection` and `tracking`, we need to run the implemented `detect_track_count.py` python script, and the argument are the same as detection. 
   
```
python detect_track_count.py --weights runs/train/yolov7-defect-detect-three2/weights/last.pt --source inference/videos/video_4.mp4 --view-img
```

#### Run for saving inspected data as `CSV` file
For saving the data of inspecting result as CSV file, `--save-csv` is needed to added to the command. This saved file is going to save in `./runs/csv/` path, and this csv file is used to visulize the qulity control plot and charts as following figure.
```
python detect_count.py --weights runs/train/yolov7-defect-detect-three2/weights/last.pt --source inference/videos/video_4.mp4 --view-img --save-csv
```

### Model Training

To train the model on your machine, make sure YOLOv7 model haved already downloaded in the project root directory.
```
python train.py --workers 0 --batch-size 1 --device 0 --data data/gear_detect.yaml --img 640 640 --cfg cfg/training/yolov7-gear-detect.yaml --weights yolov7.pt --name yolov7-defect-detect-three --hyp data/hyp.scratch.custom.yaml --epochs 200
```


------------------------------------------------------------

### Model Evaluation


-------------------------------------------------------

### References

