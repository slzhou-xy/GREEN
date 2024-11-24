# Grid and Road Expressions Are Complementary for Trajectory Representation Learning

**[KDD2025(August Cycle)]** The pytorch implementation of accepted paper "Grid and Road Expressions Are Complementary for Trajectory Representation Learning"

## Framework
<div align=center>
<img src="framework.pdf"/>
</div>


## Download data
**The repo is now incomplete. Full readme and data are coming soon**


The hyper-parameters are in `./config/config.py`

## Pretraining

```
# Chengdu
python main.py --dataset chengdu

# Porto
python main.py --dataset porto 
```

## Downstream Tasks

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
