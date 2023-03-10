---
draft: false
date: 2021-11-05
authors:
- mbenesty
categories:
- Optimization
- Transformers
tags:
- Hugging Face
- Nvidia Triton
links:
- Source code: https://github.com/ELS-RD/triton_transformers
- Project description: posts/2021/hugging-face-transformer-inference-under-1-millisecond-latency/index.md
---

# Optimization of Hugging Face Transformer models to get Inference < 1 Millisecond Latency + deployment on production ready inference server

Hi,

I just released a project showing how to optimize big NLP models and deploy them on Nvidia Triton inference server.

<!-- more -->

source code: [https://github.com/ELS-RD/triton_transformers](https://github.com/ELS-RD/triton_transformers)

project
description: [Hugging Face Transformer Inference Under 1 Millisecond Latency](../hugging-face-transformer-inference-under-1-millisecond-latency/index.md)

Please note that it is for **real life large scale NLP model deployment**. It's only based on open source softwares.
It's using tools not very often discussed in usual NLP tutorial.

Performance have been benchmarked and compared with recent Hugging Face Infinity inference server (commercial product @
20K$ for a single model deployed on a single machine).

Our open source inference server with carefully optimized models get better latency times that the commercial product in
both scenarios they have shown during the demo (GPU based).

Don't hesitate if you have any question...