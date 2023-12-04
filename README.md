# Robotic Manipulation project

## Meeting Notes

### 11/16
Agenda:
 - Review + discuss *Discussion of Architecture*

## Installations
See https://drake.mit.edu/installation.html

Our recommended setup requires using a Linux machine, or using WSL2 on Windows, with the following requirements:
- `python` 3.8 or higher
- `pip` 20.3 or higher
- `pip install manipulation --upgrade --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/`

If you are running WSL2 on Windows, ensure you install the following to enable the graphics libraries to work:
 - `sudo apt install mesa-utils`
 - Install 
 - Install XcXsrv software on your Windows Machine: https://sourceforge.net/projects/vcxsrv/files/latest/download
 - Before running this code, start an instance of XcXsrv (by starting the XLaunch application). Leave all settings in XLaunch at the default, except, *disable Access Control*. You should only need to do this once (unless you kill XcXsrv or restart your machine).
 - Every time you open a new WSL terminal, you must run `export DISPLAY=<IP ADDRESS>:0.0`, where you can find `<IP ADDRESS>` by running `ipconfig` in a command prompt (and use the IPv4 address under WSL)
 - Test that the display forwarding is working by runing `glxgears` in your WSL terminal. You should see a new window appear with an animation of spinning gears.
 - If you ever run src/main.py, but see nothing happen in the meshcat Window (but also receive no error message), you likely do not have an instance of XcXsrv running.

## Running
```
python3 src/smain.py
```

## Discussion of Architecture

The overall architecture is based largely on this example: https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/05-Bin-Picking-8e10b301-2776-448f-be43-c7f6fb54fa1f/notebook/clutter_clearing-1723573aa07d4d709fa3d410a3b5d2fe?

 - The program contains 3 core "systems":
    1. Object Ballistic Trajectory Estimator
        - Input ports:
        - Output ports:
            - Trajectory object
            - A downsampled PointCloud object (containing just the object) in Object frame
    2. Grasp Selector
        - Input ports: 
            - Trajectory object
            - A downsampled PointCloud object (containing just the object) in Object frame
        - Output ports:

    3. Motion Planner
        - Input ports: 
        - Output ports: 
 - Each individual "system" is implemented as LeafSystem. These systems will all be added to the same diagram builder, which will link their inputs and outputs together.
 - A State Machine detemines helps each leaf system determine what to do at all times. The State Machine has the following states that transition linearly:
    1. `WAITING_FOR_OBJECT`
    2. `ESTIMATING_TRAJECTORY`
    3. `SELECTING_GRASP`
    4. `PLANNING_EXECUTING_TRAJECTORY`
    5. `RETURNING_HOME`
 - The "catching system" (including robot and cameras) itself will be implemented using a MultibodyPlant
 - The objects being thrown 


## Future Directions
 - Allow thrown objects to go directly at iiwa (this makes catching harder)
 - Use more 
 
## Notes

 - Example for motion trajectory opt: https://deepnote.com/workspace/michael-zengs-workspace-61364779-69ef-470a-9f8e-02bf2b4f369c/project/06-Motion-Planning-Duplicate-c2fb7d28-4b8e-4834-ba5a-a1d69c1d218b/notebook/kinematic_trajectory_optimization-fdbbd9de17a44076a3321ed596648a24?

 - adding YCB objects (in YAML): 
    ```
    - add_model:
        name: ycb{i}
        file: package://manipulation/hydro/{ycb[object_num]}
    ```
