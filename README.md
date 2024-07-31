# GREEN

## Download data
Please download data from this [website](https://drive.google.com/drive/folders/1DZQIpoVy4TC9bGTnsNAbodMHC3GnfN7C?usp=drive_link), then move data to the folder `GREEN/data/`

## Pretraining

### Chengdu
```
python main.py --dataset chengdu
```
### Porto
```
python main.py --dataset porto 
```

## Fine-tune

### Travel Time Estimation
```
python main_tte.py --dataset chengdu 
python main_tte.py --dataset porto 
```

### Trajectory Classification
```
python main_tte.py --dataset chengdu 
python main_tte.py --dataset porto 
```

### Most Similar Trajectory Search
```
python main_sim.py --dataset chengdu 
python main_sim.py --dataset porto 
```
