# Robotic Manipulation project

## Meeting Notes

### 11/16
Agenda:
 - Review + discuss *Discussion of Architecture*

## Installations
See https://drake.mit.edu/installation.html

Our recommended setup requires using a Linux machine, or using WSL2 on Windows, with the following requirements:
- `python` 3.8 or higher
- `pip` 23.3.1 or higher

Necessary installs:
- `pip install manipulation
- `pip install --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly/ 'drake==0.0.20231210'` (or any newer version of drake)
- `pip install ipython`
- `pip install pyvirtualdisplay`

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

## Notes
 - `station.py` is a modified of `station.py` directly from drake. We imported it so that we could modify it such that we can export the `desired_acceleration` input port and also so that we could modify the PID gains of the WSG gripper.
 - In the SDF file of the object to be caught, the origin in the SDF file MUST roughly be at the object's centroid.