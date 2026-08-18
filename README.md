# 🩸 AI-Driven Blood Inventory Optimization Using TFT and PPO

## 📌 Project Overview

Blood banks need to maintain sufficient blood inventory to meet patient
demand while minimizing shortages and unnecessary excess stock.

This project presents an AI-driven blood inventory optimization system
that combines **Temporal Fusion Transformer (TFT)** for blood demand
forecasting with **Proximal Policy Optimization (PPO)** for inventory
optimization.

The system forecasts future blood demand for different blood groups and
uses reinforcement learning to determine the recommended quantity of
blood units to order while considering inventory levels, demand,
shortage costs, and excess inventory.

---

## 🎯 Objectives

- Forecast blood demand for different blood groups.
- Optimize blood inventory using reinforcement learning.
- Minimize blood shortages.
- Reduce unnecessary excess inventory.
- Generate recommended blood order quantities.
- Evaluate inventory management performance.
- Provide an interactive dashboard for visualization.

---

## 🧠 Proposed Methodology

The project consists of the following major stages:

### 1. Data Preprocessing

Historical blood donation and inventory data are cleaned and converted
into a suitable time-series format for machine learning.

### 2. Demand Forecasting using TFT

A **Temporal Fusion Transformer (TFT)** model is trained on historical
blood demand data to predict future demand for different blood groups.

### 3. Inventory Optimization using PPO

The TFT predictions are provided to a **Proximal Policy Optimization
(PPO)** reinforcement learning agent.

The PPO agent learns an inventory ordering policy based on:

- Current inventory
- Predicted demand
- Safety stock
- Ordering cost
- Shortage penalty
- Excess inventory penalty

### 4. Result Generation

The trained PPO model generates recommended order quantities for each
blood group and date.

### 5. Evaluation

The system evaluates the optimized inventory using:

- Total predicted demand
- Total recommended order
- Total shortage
- Total excess
- Total reward
- Service level
- Order efficiency

### 6. Interactive Dashboard

A Streamlit dashboard is used to visualize demand forecasts,
recommended orders, inventory levels, shortages, excess inventory, and
overall system performance.

---

## 🔄 System Architecture

```text
                 Historical Blood Data
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
                Time-Series Dataset
                         │
                         ▼
              TFT Demand Forecasting
                         │
                         ▼
               Predicted Blood Demand
                         │
                         ▼
              PPO Reinforcement Learning
                         │
                         ▼
             Inventory Optimization
                         │
                         ▼
              Recommended Order
                         │
                         ▼
                   Evaluation
                         │
                         ▼
               Interactive Dashboard
