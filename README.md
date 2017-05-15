# darwin
darwin is a library for distributed blackbox optimization with [Evolutionary Strategies](https://arxiv.org/abs/1703.03864). 

## Installation


## Guiding Principles and Contribution Guidelines
* Modular: write once, use often. The library will be split in three somewhat independent submodules. The [*core*](https://github.com/nasimrahaman/darwin/tree/master/darwin/core) should handle server-worker communication, [*models*](https://github.com/nasimrahaman/darwin/tree/master/darwin/models) implement Keras models and [*metrics*](https://github.com/nasimrahaman/darwin/tree/master/darwin/metrics) define the optimization objectives and/or RL-environments. 
* Scalability: this project is intended to scale to clusters with at least 500+ nodes. Integration with cluster management tools like [Kubernetes](https://kubernetes.io) is a future goal.
* PEP 20: "Readability counts." 

## Getting in Touch
This project is managed by Nasim Rahaman and Lukas Schott at the [Image Analysis and Learning](https://hci.iwr.uni-heidelberg.de/mip) Lab @ [Heidelberg Collaboratory for Image Processing](https://hci.iwr.uni-heidelberg.de/), [University of Heidelberg](https://www.uni-heidelberg.de/). Get in touch by opening an Github issue or by email, firstname.lastname [at] iwr.uni-heidelberg.de.
