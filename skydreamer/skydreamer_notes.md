# SkyDreamer: Implementation-Oriented Summary

## Objective
An end-to-end vision-based drone racing system:
- Input: segmentation mask (64x64), IMU body rates, motor RPM, flight plan
- Output: direct motor commands (4 motors, normalized [0,1])
- Runs fully onboard, no calibration, no PID controller

---

## Core Idea
Use model-based reinforcement learning (Dreamer-style):
- Learn a world model
- Train actor in latent space (imagination)
- Use privileged state info during training only

---

## Architecture

Pipeline:
Camera → Segmentation → World Model → Latent → Actor → Motor Commands

World Model (RSSM):
- Encoder: z_t ~ q(z | h_t, o_t)
- Sequence (GRU): h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
- Dynamics: ẑ_t ~ p(z | h_t)
- Decoder: predicts privileged state
- Reward + continuation predictors

Actor-Critic:
- Actor: a_t ~ π(a | h_t, z_t)
- Critic: V(h_t, z_t)
- Gaussian policy, deterministic at inference

---

## Observations

Input:
- Segmentation mask (64x64)
- IMU body rates
- Motor RPM

Training-only (privileged):
- Position, velocity, orientation
- Motor speeds
- Camera extrinsics
- Dynamics parameters

---

## Actions
- 4 motor commands in [0,1]
- Direct control (no PID)

---

## Reward Function

r_t = 5 * progress - rate_penalty + 30 * gate_reward

- Progress: distance reduction to next gate
- Rate penalty: discourages high angular velocity
- Gate reward: centered gate crossing bonus

Episode ends on collision or instability.

---

## Flight Plan Logic

Adds structure to avoid ambiguity:
- Encodes positions/yaws of next 3 gates
- Updates when drone passes gate

---

## Dynamics Model

Includes:
- Position, velocity, rotation (quaternion)
- Motor dynamics
- Disturbances

Simulated with Runge-Kutta integration.

---

## Vision System

Segmentation:
- U-Net model
- Output binary mask

Augmentations:
- GAN-based sim-to-real (CycleGAN)
- Mask erosion
- Rolling shutter simulation

---

## Training

- 17M steps
- Replay buffer: 10M
- Batch length increases during training
- Learning rate + entropy decay

Smoothness regularization:
L = λ ||μ_t - μ_{t-1}||²

---

## Deployment

- ~90 Hz control loop
- ~4 ms total inference time

Pipeline:
JAX → PyTorch → ONNX → TensorRT

---

## Performance

- Speed: ~21 m/s
- Acceleration: ~6g
- Fully onboard
- Robust to noise and sim-to-real gap

---

## Key Takeaways

- Latent world models enable efficiency
- Privileged decoding enables state estimation
- Flight plan solves visual ambiguity
- Direct motor control improves agility
- Vision augmentation is critical for sim-to-real
