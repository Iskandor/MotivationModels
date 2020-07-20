#python main.py --env lunar_lander --model baseline --trials 1 --episodes 4000 --batch_size 64 --memory_size 10000
python main.py --env half_cheetah --model fm --trials 3 --episodes 4000 --batch_size 64 --memory_size 100000
python main.py --env half_cheetah --model su --trials 3 --episodes 4000 --batch_size 64 --memory_size 100000
