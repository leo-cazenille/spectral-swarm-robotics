# spectral-swarm-robotics

This is the main repository of the "Hearing the Shape of an Arena with Spectral Swarm Robotics" article, available on Arxiv [here](https://arxiv.org/pdf/2403.17147.pdf).

## Abstract
Swarm robotics promises adaptability to unknown situations and robustness against failures. However, it still struggles with global tasks that require understanding the broader context in which the robots operate, such as identifying the shape of the arena in which the robots are embedded. Biological swarms, such as shoals of fish, flocks of birds, and colonies of insects, routinely solve global geometrical problems through the diffusion of local cues. This paradigm can be explicitly described by mathematical models that could be directly computed and exploited by a robotic swarm.

Diffusion over a domain is mathematically encapsulated by the Laplacian, a linear operator that measures the local curvature of a function. Crucially the geometry of a domain can generally be reconstructed from the eigenspectrum of its Laplacian. Here we introduce spectral swarm robotics where robots diffuse information to their neighbors to emulate the Laplacian operator -enabling them to "hear" the spectrum of their arena.

We reveal a universal scaling that links the optimal number of robots (a global parameter) with their optimal radius of interaction (a local parameter). We validate experimentally spectral swarm robotics under challenging conditions with the one-shot classification of arena shapes using a sparse swarm of Kilobots. 

Spectral methods can assist with challenging tasks where robots need to build an emergent consensus on their environment, such as adaptation to unknown terrains, division of labor, or quorum sensing. 

Spectral methods may extend beyond robotics to analyze and coordinate swarms of agents of various natures, such as traffic or crowds, and to better understand the long-range dynamics of natural systems emerging from short-range interactions.


## Install and Compile binaries
We use singularity/apptainer to easily launch simulations on clusters.
We assume that singularity is already installed on your Linux computer. If not, please follow [this guide](https://docs.sylabs.io/guides/3.0/user-guide/installation.html).

To compile a singularity image:
```shell
sudo singularity build -F  spectral-kilo.simg singularity.def
```

To compile a Kilombo simulation using this singularity image:
```shell
make clean; singularity exec spectral-kilo.simg make -j 20
```

To launch a simulation using the parameters of regime r3 (cf Figs. 2 and 3 of main text) using only 32 runs:
```shell
./runOneSingularityExpe.sh final-ag350-fop85-expe36900-32runs.yaml
```
Final results will be found in directory "results/"


