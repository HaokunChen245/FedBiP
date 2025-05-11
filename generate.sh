#!/bin/bash
DATASET="domainnet"
if [[ "$DATASET" == "officehome" ]]; then
    domains=("Art" "Clipart" "Product" "Real")
elif [[ $DATASET == "domainnet" ]]; then
    domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
elif [[ $DATASET == "pacs" ]]; then
    domains=("photo" "sketch" "art_painting" "cartoon")
else
    domains=("client_0" "client_1" "client_2" "client_3" "client_4")
fi

idx=5
for domain in "${domains[@]}"
do
    echo "Testing domain: $domain"
    CUDA_VISIBLE_DEVICES=$idx python generate.py \
        --output_dir "datasets_$DATASET" --evaluation_type generate_prompt \
        --domain "$domain" --test_type syn_wnoise_0.1_test_interp \
        --dataset "$DATASET"
    idx=$((idx+1))
done