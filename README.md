# Context-Adaptive Multi-Objective Construction Project Scheduling

## Overview

This repository contains the computational framework developed for the research project:

**A Context-Adaptive Multi-Objective Framework for Sustainable Multi-Mode Construction Project Scheduling Considering Dynamic Environmental Context**

The framework extends multi-mode resource-constrained project scheduling with context-aware adaptation, sustainability objectives, and adaptive evolutionary search.

The computational architecture integrates:

- Multi-Mode Resource-Constrained Project Scheduling
- Multi-objective optimization
- NSGA-II
- Context-aware adaptation
- Adaptive crossover and mutation operators
- Sustainable objective evaluation
- Renewable and non-renewable resource constraints
- PSPLIB benchmark validation
- Multi-seed experimental evaluation

---

## Research Framework

The proposed computational framework contains the following main components:

1. PSPLIB/MM project parsing
2. Project and resource modeling
3. Multi-mode activity representation
4. Serial Schedule Generation Scheme (SSGS)
5. Schedule feasibility validation
6. Sustainability objective generation
7. Multi-objective evaluation
8. NSGA-II optimization
9. Context management
10. Context-adaptive operator control
11. Common normalization and reference-set construction
12. Hypervolume and IGD+ evaluation
13. Multi-seed experimental analysis

---

## Objectives

The optimization framework evaluates four objectives:

1. Makespan
2. Total cost
3. Total carbon emissions
4. Total energy consumption

All objectives are treated within a multi-objective minimization framework.

---

## Context-Adaptive NSGA-II

The proposed CA-NSGA-II incorporates a context vector containing:

- Carbon pressure
- Energy pressure
- Resource pressure
- Cost pressure
- Schedule pressure
- Uncertainty

The context is used to dynamically adjust evolutionary operator probabilities.

The current implementation adapts:

- Crossover probability
- Mutation probability

The baseline NSGA-II uses fixed operator probabilities for experimental comparability.

---

## Benchmark Dataset

The framework supports PSPLIB multi-mode benchmark instances.

The current validation includes:

- 640 PSPLIB MM instances
- 32 activities per j30 instance
- Renewable and non-renewable resources
- Multiple execution modes

Parser validation currently reports:

**640 / 640 instances passed**

Resource validation currently reports:

**640 / 640 instances passed**

---

## Experimental Protocol

The experimental framework supports:

- Multiple random seeds
- Baseline NSGA-II
- Context-Adaptive NSGA-II
- Common objective normalization
- Common reference set
- Common reference point
- Hypervolume evaluation
- IGD+ evaluation
- Persistent experimental results

The current protocol validation uses:

- Instance: `j3010_1.mm`
- Seeds: `42, 43, 44`
- Population size: `10`
- Generations: `5`

Pilot validation results:

- Baseline mean HV: `0.350155`
- CA mean HV: `0.373140`
- HV improvement: `6.5643%`
- Baseline mean IGD+: `0.351433`
- CA mean IGD+: `0.344636`
- IGD+ improvement: `1.9340%`

These values are **protocol-validation results**, not the final empirical results of the research paper.

---

## Reproducibility

The implementation includes deterministic seed handling.

Current reproducibility validation confirms:

- Identical seed → identical NSGA-II result
- Different seed → different NSGA-II result
- Identical context seed → identical context trajectory
- Different context seed → different context trajectory

---

## Validation Status

The current automated validation suite reports:

```text
9 passed