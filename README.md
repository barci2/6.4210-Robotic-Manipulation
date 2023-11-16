# Robotic Manipulation project

## Meeting Notes

### 11/16
Agenda:
 - Review + discuss *Discussion of Architecture*

## Installations
See https://drake.mit.edu/installation.html

Our recommended setup requires using a Linux machine, or using WSL on Windows, with the following requirements:
- `python` 3.8 or higher
- `pip` 20.3 or higher
- `pip install manipulation --upgrade --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/`


## Running
```python
python3 main.py
```

## Discussion of Architecture

The overall architecture is based largely on this example: https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/05-Bin-Picking-8e10b301-2776-448f-be43-c7f6fb54fa1f/notebook/clutter_clearing-1723573aa07d4d709fa3d410a3b5d2fe?

 - The program contains 3 core "systems":
   1. Object Ballistic Trajectory Estimator
   2. Grasp Selector
   3. Motion Planner
 - Each individual "system" is implemented as LeafSystem. These systems will all be added to the same diagram builder, which will link their inputs and outputs together.
 - A State Machine detemines helps each leaf system determine what to do at all times. The State Machine has the following states that transition linearly:
   1. WAITING_FOR_OBJECT
   2. ESTIMATING_TRAJECTORY
   3. SELECTING_GRASP
   4. PLANNING_EXECUTING_TRAJECTORY
 - The "catching system" (including robot and cameras) itself will be implemented using a MultibodyPlant
 - The objects being thrown 


## TODO
 - How to add model drivers to MBP
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
