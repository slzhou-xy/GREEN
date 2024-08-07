# GREEN

## Download data
Please download data from this [website](https://drive.google.com/drive/folders/1DZQIpoVy4TC9bGTnsNAbodMHC3GnfN7C?usp=drive_link), then move data to the folder `GREEN/data/`

The hyper-parameters are in `GREEN/config/config.py`

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
