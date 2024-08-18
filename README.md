# Topics Causal Inference

Replication repository for Causal Inference Topics Course Summer Semester 2024.

## Instructions

To run the replication, Python has to be installed on the system.

First, after cloning the repository, create the virtual environment by typing

```shell
mamba env create
```

in the shell. Note the current path of the shell needs to be the cloned repository.

For building the project, I use a workflow management system called
[`pytask`](https://pytask-dev.readthedocs.io/en/stable/). `pytask` facilitates
reproducible analysis by automatically discovering tasks and evaluating tasks only when
the code or dependencies have changed.

To build the project, first activate the virtual environment and then run the build:

```shell
conda activate topics_causal_inf
pytask
```

However, it is advised to run the tasks collected by `pytask` in parallel by specifying

```shell
pytask -n [n_workers]

# Alternative: Starts os.cpu_count() - 1 workers.
pytask -n auto
```

With 11 workers on my private computer, execution took about 90 minutes. To reduce
runtime, the simulation settings in `config.py` might be changed.

In particular,

```python
N_SIM = 40
N_SPLITS = 10
```

control the number of total simulation runs as well as the number of splits for the
generic ML procedure.

Note that I generated the inputs to the WGAN-based simulations externally on a Google
Colab server. The results have to be stored under
`src\topics_causal_inf\wgan_generated\`. Alternatively,
`DG_TO_RUN = ["standard", "wgan"]` in `task_wa_replication.py` has to be changed (i.e.
remove `"wgan"`).
