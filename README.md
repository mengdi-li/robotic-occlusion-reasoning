# Robotic Occlusion Reasoning for Efficient Object Existence Prediction

This is the code for our IROS2021 Paper [Robotic Occlusion Reasoning for Efficient Object Existence Prediction](https://arxiv.org/abs/2107.12095). The code is mainly (the model part) based on the PyTorch implementation by [kevinzakka](https://github.com/kevinzakka/recurrent-visual-attention) of [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247). 

## Requirements
* Create and activate the conda environment
    * `conda env create -f environment.yml`
    * `conda activate pyrep`
* Install [CoppeliaSim](https://www.coppeliarobotics.com/)
* Install [PyRep](https://github.com/stepjam/PyRep)


## Train a model

### Train on a specific level of data
Specify the configuration file through `--cfg_file`:

```
python code/trainer.py --cfg_file cfg/train-final-model/train-level1-1-vis.yml
```

Checkpoints and log files are stored in the `./data` folder. Visualization by tensorboard:
```
tensorboard --logdir ./data
```

### Curriculum training
```
./scripts/start-training.sh
```

### Troubleshooting
* `ImportError: libcoppeliaSim.so.1: cannot open shared object file: No such file or directory`  
Create a symbolic link named "libcoppeliaSim.so.1" to "libcoppeliaSim.so" manually:   
`ln -s /PATH/OF/COPPLELIASIM/libcoppeliaSim.so /PATH/OF/COPPLELIASIM/libcoppeliaSim.so.1`

## Running Headless
### **Method 1**: Xvfb
This method has no 3D hardware acceleration for rendering but is easy to make a start. To train the model:
```
xvfb-run python code/trainer.py --cfg_file cfg/train-final-model/train-level1-1-vis.yml
```
or
```
xvfb-run ./scripts/start-training.sh
```

### **Method 2**: Dummy X server
This method provides full 3D hardware acceleration.  
* If you have the `sudo` permission, run `sudo python scripts/startx.py`. By default, a dummy X server with the display number of "1" will be created. If the display number of "1" is already occupied, you need to assign other free display numbers. Check occupied display numbers by `ls /tmp/.X11-unix/`. 
* If you don't have the `sudo` permission, run `python scripts/generate-startx-conf.py` to generate a Xorg configuration file named `dummy-xorg.conf` by default at the working directory. You need to ask the administrator to copy `dummy-xorg.conf` to `/etc/X11/` and config `allowed_users = anybody` at `/etc/X11/Xwrapper.config`. Then you can run `Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config dummy-xorg.conf :1` to start a dummy X11 server. `:1` is the display number. 
* When a dummy X11 server is running, you can train the model by 
```
DISPLAY=:1.0 python code/trainer.py --cfg_file cfg/train-final-model/train-level1-1-vis.yml
```
or
```
DISPLAY=:1.0 ./scripts/start-training.sh
```
By using `DISPLAY=:1.0`, the first GPU will do the rendering work. If we want to use the second GPU, we should set `DISPLAY=:1.1`. 

### **Method 3**: VirtualGL
Similar to the dummy X server based approach, this method provides full 3D hardware acceleration. 
Please refer to [the README file of PyRep](https://github.com/stepjam/PyRep#running-headless). 

## Test a model

```
DISPLAY=:1.0 python code/trainer.py --cfg_file cfg/train-final-model/test.yml
```

## Cite
If you find this work useful, please cite our paper:
```
@InProceedings{LWKLZLW21,
  author       = "Li, Mengdi and Weber, Cornelius and Kerzel, Matthias and Lee, Jae Hee and Zeng, Zheni and Liu, Zhiyuan and Wermter, Stefan",
  title        = "Robotic Occlusion Reasoning for Efficient Object Existence Prediction",
  booktitle    = "2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)",
  month        = "Oct",
  year         = "2021",
  doi          = "10.1109/IROS51168.2021.9635947",
  url          = "https://www2.informatik.uni-hamburg.de/wtm/publications/2021/LWKLZLW21/LI_IROS2021.pdf"
}
```


## Contact
Mengdi Li - mengdi.li@studium.uni-hamburg.de
