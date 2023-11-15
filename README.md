# Robotic Manipulation project

## Installations
See https://drake.mit.edu/installation.html

Our recommended setup requires using a Linux machine, or using WSL on Windows, with the following requirements:
- `python` 3.8 or higher
- `pip` 20.3 or higher
- `pip install manipulation --upgrade --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/`


## Running
```
python3 main.py
```

## TODO
 - Fix missing collision geometry for iiwa and how to add model directives
 - Add cameras
 - Add flying object
 - Add skeleton for perception and grasping
 - Add hand

## Notes
 - Example for motion trajectory opt: https://deepnote.com/workspace/michael-zengs-workspace-61364779-69ef-470a-9f8e-02bf2b4f369c/project/06-Motion-Planning-Duplicate-c2fb7d28-4b8e-4834-ba5a-a1d69c1d218b/notebook/kinematic_trajectory_optimization-fdbbd9de17a44076a3321ed596648a24?
 - adding YCB objects (in YAML): 
    ```
    - add_model:
        name: ycb{i}
        file: package://manipulation/hydro/{ycb[object_num]}
    ```