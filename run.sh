# cd model
# learning_rates=(0.001 0.0005 0.0008)  
# lr_decay_rates=(0.05 0.03 0.1)

# for learning_rate in "${learning_rates[@]}"
# do
#     for lr_decay_rate in "${lr_decay_rates[@]}"
#     do
#         echo "Running script for learning rate: $learning_rate"
#         echo "Running script for lr decay rates: $lr_decay_rate"
#         python main.py --lr_init $learning_rate --lr_decay_rate $lr_decay_rate --dataset='CHITaxi' --model='GSSG' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --weight_decay=1e-3 --epochs=200 --tensorboard --comment="" --device=3
#     done
# done

python main.py --dataset='CHITaxi' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --tensorboard --comment="" --device=4