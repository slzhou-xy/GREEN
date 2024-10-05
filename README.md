# GREEN

## Download data
For privacy, we only publish the sampled data (`./data/`) at the review, and if the paper is accepted, we will open source all the data and code.


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
