# Autonomous Security Drone

An undergraduate autonomous drone project combining computer vision, autonomous flight, and MAVLink-based flight control for aerial monitoring and security-oriented applications.

## Overview

This project was developed during my undergraduate engineering studies and explores the integration of a Pixhawk-based flight controller with Python computer vision and autonomous control.

The development process included:

- Pixhawk / MAVLink communication
- DroneKit-based flight control
- Autonomous takeoff, landing and movement
- ArduPilot SITL simulation
- OpenCV-based visual detection
- STOP-sign detection and tracking experiments
- Face detection and vision-guided yaw control
- Vision-guided flight-control prototypes
- A TUBITAK 2209-A research proposal on an image-processing-based autonomous area-control drone

## Flight Control

The flight-control modules include:

- Vehicle connection and telemetry
- Arming and disarming
- Autonomous takeoff
- Landing
- Return-to-Launch (RTL)
- Yaw control
- Local movement commands

## Computer Vision

OpenCV experiments include:

- HSV-based red object segmentation
- Octagonal STOP-sign detection
- Contour-based object detection
- Haar Cascade face detection
- Target position calculation relative to the camera center

## Vision-Guided Flight

Experimental modules combine visual detections with MAVLink flight commands. These prototypes explore yaw, roll and pitch control according to a detected target's position in the camera frame.

## Simulation

ArduPilot / MAVLink simulation experiments include SITL connection, autonomous arming and takeoff, yaw control, movement and automated landing.

## Repository Structure

- src/flight_control - Core drone-control functions
- src/vision - Computer-vision experiments
- src/autonomy - Vision-guided control prototypes
- simulation - ArduPilot and MAVLink simulation experiments
- tests/flight - Historical flight tests
- docs/thesis - Undergraduate project reports
- docs/research_project - Research-project documentation
- docs/presentations - Project presentations
- media/images - Project images
- archive/early_experiments - Early development experiments

## Technology Stack

- Python
- OpenCV
- NumPy
- DroneKit
- PyMAVLink / MAVLink
- Pixhawk
- ArduPilot
- ArduPilot SITL

## Documentation

The repository contains documentation from the original engineering process, including undergraduate project reports, presentations and TUBITAK 2209-A research-project material.

## Project Status

This is a historical undergraduate engineering project reconstructed and organized from the original development files. The repository separates core source code, simulation, flight tests and experimental prototypes.

## Safety

The flight-control software in this repository is experimental. Review and validate the code in simulation before considering operation on real hardware.
