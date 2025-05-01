# MrSteve: Instruction-Following Agents in Minecraft with What-Where-When Memory

Author: {Junyeong Park, Junmo Cho}, Sungjin Ahn<br/>
***Published to ICLR 2025***<br/>
[[paper]](https://openreview.net/forum?id=CjXaMI2kUH) [[project]](https://sites.google.com/view/mr-steve)

## Abstract

Significant advances have been made in developing general-purpose embodied AI in environments like Minecraft through the adoption of LLM-augmented hierarchical approaches. While these approaches, which combine high-level planners with low-level controllers, show promise, low-level controllers frequently become performance bottlenecks due to repeated failures. In this paper, we argue that the primary cause of failure in many low-level controllers is the absence of an episodic memory system. To address this, we introduce MrSteve (**M**emory **R**ecall Steve), a novel low-level controller equipped with Place Event Memory (PEM),  a form of episodic memory that captures what, where, and when information from episodes. This directly addresses the main limitation of the popular low-level controller, Steve-1. Unlike previous models that rely on short-term memory, PEM organizes spatial and event-based data, enabling efficient recall and navigation in long-horizontal tasks. Additionally, we propose an Exploration Strategy and a Memory-Augmented Task Solving Framework, allowing agents to alternate between exploration and task-solving based on recalled events. Our approach significantly improves task-solving and exploration efficiency compared to existing methods, and we are releasing our code to support further research.


## Quick Start

### Step 1. Set up Python Environment

```bash
uv sync
python -m pip install git+https://github.com/MineDojo/MineCLIP
python -m pip install gym==0.21.0
```

And, follow the environment installation guide from [here](https://github.com/PKU-RL/MCEnv).

### Step 2. Download Pre-trained Model Weights

We provide a script for downloading pre-trained weights of MineCLIP, Steve-1, and VPT-Nav.

```bash
uv run bash prepare_models.sh
```

### Step 3. Run and Evaluation

You can execute episodes with the following commands. For the task, please refer to [task_specs.yaml](mrsteve/env/minedojo/task/minedojo/task_specs.yaml).

```bash
# MrSteve
uv run main.py task=log_water_bucket_aba_randinit agent=mrsteve n_episodes=100

# MrSteve-EM
uv run main.py task=log_water_bucket_aba_randinit agent=mrsteve agent.epmem.memory_option=event_memory

# MrSteve-PM
uv run main.py task=log_water_bucket_aba_randinit agent=mrsteve agent.epmem.memory_option=place_memory

# MrSteve-FM
uv run main.py task=log_water_bucket_aba_randinit agent=mrsteve agent.epmem.memory_option=fifo_memory

# Steve-1
uv run main.py task=log_water_bucket_aba_randinit agent=steve1
```

Episode and agent logs will be stored in the `outputs` directory, and you can calculate statistics using the script as follows:

```bash
uv run scripts/get_stats.py outputs/log_water_bucket_aba_randinit/mrsteve/*
```

## Citation
```
@inproceedings{
    park2025mrsteve,
    title={{M}r{S}teve: Instruction-Following Agents in Minecraft with What-Where-When Memory},
    author={Junyeong Park and Junmo Cho and Sungjin Ahn},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=CjXaMI2kUH}
}
```
