# Grid and Road Expressions Are Complementary for Trajectory Representation Learning

The pytorch implementation of KDD2025 accepted paper "Grid and Road Expressions Are Complementary for Trajectory Representation Learning"


## Download data
Data is coming soon


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
