
The code presented was used in the following publication [(preprint here)](https://arxiv.org/abs/2304.10860).

[1] S. Gracla, C. Bockelmann and A. Dekorsy,
"On the Importance of Exploration for Real Life Learned Algorithms",
in *Proc. 2022 IEEE 23rd International Workshop on Signal Processing Advances in Wireless Communication (SPAWC)*,
Oulu, Finland, 28. - 30. July 2022,
pp. 1-5, doi: [10.1109/SPAWC51304.2022.9834009](https://doi.org/10.1109/SPAWC51304.2022.9834009).

Email: {**gracla**, bockelmann, dekorsy}@ant.uni-bremen.de

The project structure is as follows:

```
/project/
├─ exploration_project_imports/     | python modules
├─ .gitignore                       | .gitignore
├─ a_config.py                      | contains configurable parameters
├─ requirements.txt                 | project dependencies
├─ runner.py                        | orchestrates training & testing
├─ testing_wrapper.py               | wrapper for testing different configurations
├─ training_wrapper.py              | wrapper for training different configurations
```