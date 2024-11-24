# Grid and Road Expressions Are Complementary for Trajectory Representation Learning

**[KDD2025(August Cycle)]** The pytorch implementation of accepted paper "Grid and Road Expressions Are Complementary for Trajectory Representation Learning"

## Framework
<div align=center>
<img src="framework.png"/>
</div>


## Dataset
We only provide Porto dataset. Due to privacy, we cannot provide the Chengdu dataset.

To use dataset, unzip dataset in directory `./data/porto`.

- `./data/porto/rn/...` is the road network data.

- `./data/porto/traj.csv` is the raw trajectory data.

- `./data/porto/traj/*_od.csv` is the trajectory data for similarity.
  
- To get road trajectories, please refer to the [FMM](https://github.com/cyang-kth/fmm).

## Hyper-parameters

The hyper-parameters are in `./config/config.py`. You can modify it according to your needs.

## Pretraining

```
# Chengdu
python main.py --dataset chengdu

# Porto
python main.py --dataset porto 
```
You can set **exp_id** in the `main.py`.

When running the model for the first time, it preprocesses the data, which will take some time, so be patient.



## Downstream Tasks

When run the model for the downstream tasks, set the same **exp_id** in `main_<task>.py` as for pre-training.

### Travel Time Estimation (Fine-tuning)
```
# Chengdu
python main_tte.py --dataset chengdu

# Porto
python main_tte.py --dataset porto 
```

### Trajectory Classification (Fine-tuning)
```
# Chengdu
python main_cls.py --dataset chengdu

# Porto
python main_cls.py --dataset porto 
```

### Most Similar Trajectory Search (No Fine-tuning)
```
# Chengdu
python main_sim.py --dataset chengdu

# Porto
python main_sim.py --dataset porto 
```
