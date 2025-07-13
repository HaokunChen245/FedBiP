DATASET='domainnet'
if [[ "$DATASET" == "officehome" ]]; then
    domains=("Art" "Clipart" "Product" "Real")
elif [[ $DATASET == "domainnet" ]]; then
    domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
elif [[ $DATASET == "pacs" ]]; then
    domains=("photo" "sketch" "art_painting" "cartoon")
else
    domains=("client_0"  "client_1" "client_2" "client_3" "client_4")
fi

idx=0
TRAIN_TYPE=prompt
for domain in "${domains[@]}"
do
    echo "Training domain: $domain"
    CUDA_VISIBLE_DEVICES=$idx python train.py \
        --output_dir "exps_$DATASET/$TRAIN_TYPE"_d_"$domain"_multiclient --evaluation_type generate \
        --num_train_epochs 50 --train_batch_size 8 \
        --domain "$domain" --train_type $TRAIN_TYPE \
        --dataset "$DATASET" --num_shot 16 \
        --learning_rate 0.1 &
    idx=$((idx+1))
done
