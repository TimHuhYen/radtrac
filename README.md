# radtrac: Aircraft Tracking with Kalman Filters

A radar-style tracker. Built with Python, NumPy. Tested on live flight data over Huntsville, AL (OpenSky).

A Kalman filter repeats two steps: predict where the object should be using a motion model, then corrects that guess using a noisy measurement, weighting each by how much it can be trusted. 

## What it does

    - Linear Kalman filter and constant velocity plane model
    - Extended Kalman filter (ekf) for radar style readings (range and bearing), including hangling of the +/-100 degree angle wrap around (pi-issue)
    - Fusing two radars into one position estimate
    - Flight data from OpenSky Network (ADS-B tracks), converted from latitude/longitude to local x/y meters around Huntsville, AL area

## Results
...