# TRI Drone Project, NAVLab

*Author: Torstein Ørbeck Eliassen*

## Introduction

The software in this repository processes drone images/ web cam images in order to obtain relative distance measurements.
Multiple approaches for detection have been investigated, such as apriltag detection, feature detection and color segmentation.

## Tracking videos

The following videos are tracked using 1 apriltag detection and then SIFT detection to generate the relative vector visualized below. 


*Flightroom tracking*

![Flightroom: Tracking with SIFT](plots/gifs/flightroom.gif)

*Thunderhill tracking*

![Thunderhill: Tracking with SIFT](plots/gifs/thunderhill.gif)


## Apriltag tag comparisons


*Tag family 16h5: Id comparison*

![Tag family 16h5, tag id comparison](plots/16h5_comparison.png)

*Tag families at low resolution*

![Tag family comparison](plots/16h5_vs_36h11/uniq_fam_detections.png)


## SIFT detections

SIFT detect and match on arbitrary image: 

![SIFT matches](images/sift/matches_sift.png)