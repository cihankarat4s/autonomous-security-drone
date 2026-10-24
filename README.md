# Autonomous Security Drone

A low-cost autonomous drone that flies a pre-planned mission over a protected area, detects intruders with an onboard object detector and raises an alarm.

This started as my undergraduate graduation project (TÜBİTAK 2209-A supported) and is now being **rebuilt in simulation** (ArduPilot SITL + Gazebo), because the original hardware is no longer available.

> **Status:** Work in progress. The simulation environment is set up; the companion application is being rewritten from scratch.

## Original system (hardware, tested during the project)

1. A mission planned in Mission Planner is flown autonomously by a Pixhawk running ArduPilot.
2. A Raspberry Pi 5 companion computer listens to the flight controller over MAVLink and starts detection when the drone reaches the surveillance waypoint (`MISSION_ITEM_REACHED`).
3. Frames from a Camera Module 3 NoIR are enhanced (histogram equalization, sharpening) and passed to a YOLOv8n detector looking for people.
4. On a detection, a buzzer is triggered and a `STATUSTEXT` alert is sent to the ground station.
5. The drone returns home with RTL when the mission ends.

![Hardware wiring diagram](media/images/hardware_wiring_diagram.png)

**Hardware:** Pixhawk, Raspberry Pi 5 (8 GB), Camera Module 3 NoIR, 3DR 433 MHz telemetry, 3S 6200 mAh Li-Po, MINI560 step-down regulator, buzzer, FlySky FS-i6X.

The original companion script and the custom-trained detection model were not preserved. Earlier 2024 prototypes (DroneKit flight helpers, OpenCV STOP-sign and face tracking) are available in the initial commit `dbd549e`.

## Rebuild in simulation

```
ArduPilot SITL (ArduCopter) <-- MAVLink --> companion app (Python, pymavlink)
        ^                                      - mission trigger (MISSION_ITEM_REACHED)
        | ardupilot_gazebo (JSON)              - camera -> YOLOv8n person detection
Gazebo Harmonic: drone + gimbal camera         - non-blocking alarm + STATUSTEXT
Mission Planner (Windows) <-- UDP 14550
```

Environment: Windows 11 + WSL2 (Ubuntu 22.04), ArduPilot SITL, Gazebo Harmonic with `ardupilot_gazebo`, NVIDIA GTX 1650 Ti.

The original custom model was not preserved. The rebuilt version uses YOLOv8n fine-tuned on VisDrone persons plus Gazebo hard negatives (`training/`), loaded from `yolov8n_visdrone_person_neg.pt` in the repository root. The weights are not in git: train them with `training/colab_train.ipynb`, or run with the COCO weights instead (`--model yolov8n.pt`, set `detection.confidence` to 0.4 in `config/default.yaml`).

## Repository structure

- `media/images/` – wiring diagram and test footage frame

## Safety

Flight code in this repository is experimental. Validate everything in SITL before considering real hardware.
