# AI Grand Prix — Technical Specification  
**Document ID:** VADR-TS-001  
**Issue:** 00.01  
**Date:** 2026-03-09  

---

# 1. Document Control

## 1.1 Revision History
| Issue | Date       | Author | Summary          |
|------|-----------|--------|------------------|
| 00.01 | 2026-03-09 | KH     | First release    |

## 1.2 Audience
- Competition participants ("Teams")
- Developers building autonomous control software
- Engineers interfacing with simulator APIs

---

# 2. Purpose and Scope

## 2.1 Purpose
Defines the interface between contestant control software and the race simulator.

## 2.2 Scope
Includes:
- Communication interfaces
- Control input requirements
- Telemetry interfaces
- Vision data interfaces
- Simulation timing constraints
- Virtual environment definition
- Qualification requirements

## 2.3 Out of Scope
- Internal simulator architecture
- Event operations
- Commercial/contractual aspects

---

# 3. Simulation Environment

## 3.1 General Environment
- Start gate
- Sequential race gates
- Finish gate
- Vertical/horizontal obstacles
- Boundary elements
- Terrain + environmental structures

## 3.2 Physical Simulation Model
- Rigid-body drone physics
- Includes thrust, drag, gravity, and collisions
- Physics Update Rate: 120 Hz

## 3.3 Spatial Reference Model
- Local Cartesian coordinate system
- No GPS
- No global position exposure

## 3.4 Visual Environment
- Forward-facing first-person camera
- Includes gates, structures, static objects, dynamic lighting

## 3.5 Environmental Determinism
- Identical course geometry
- Identical physics parameters
- Fully deterministic conditions

---

# 4. Communication Protocol — MAVLink Interface

## 4.1 Overview
- Protocol: MAVLink v2
- Interface: MAVSDK-compatible

## 4.2 Transport
- UDP

## 4.3 Supported MAVLink Messages

| Message | Direction | Purpose |
|--------|----------|--------|
| HEARTBEAT | Simulator → Client | Connection status |
| ATTITUDE | Simulator → Client | Vehicle attitude |
| HIGHRES_IMU | Simulator → Client | Sensor data |
| TIMESYNC | Simulator → Client | Time sync |
| ODOMETRY | Simulator → Client | Position/velocity |
| SET_POSITION_TARGET_LOCAL_NED | Client → Simulator | Position control |
| SET_ATTITUDE_TARGET | Client → Simulator | Attitude control |

## 4.4 Timing Constraints

| Parameter | Value |
|----------|------|
| Physics Rate | 120 Hz |
| Command Rate | 50–120 Hz |
| Heartbeat Rate | 2 Hz minimum |

## 4.5 Telemetry Data
- Attitude
- Orientation
- Linear velocities
- System status flags
- Navigation reference data

## 4.6 Vision Stream
- Provided separately

## 4.7 SITL Bridge
- Low-latency UDP bridge for external controllers

---

# 5. Contestant Software Environment

## 5.1 Runtime Environment
- Python supported (3.14.2 verified)
- Other environments allowed

## 5.2 Client Responsibilities
- Establish MAVLink communication
- Maintain heartbeat
- Send control commands
- Process telemetry
- Process vision data

## 5.3 Intended Control Architecture

```
Vision + Telemetry
        ↓
    Perception
        ↓
     Planning
        ↓
      Control
        ↓
  Pilot Commands
        ↓
Stabilized Controller
```

---

# 6. Example Control Session

```
1. Initialize MAVSDK
2. Connect to simulator
3. Receive HEARTBEAT
4. Send control commands
5. Simulator applies commands
6. Receive telemetry + vision
```

---

# 7. Compliance

- Fully autonomous operation required
- No human interaction during runs

---

# 8. Qualification Phase

## 8.1 Objective
- Navigate racecourse successfully

## 8.2 Course Structure
- Start gate
- Intermediate gates
- Finish gate

## 8.3 Constraints

| Parameter | Value |
|----------|------|
| Max Duration | 8 minutes |
