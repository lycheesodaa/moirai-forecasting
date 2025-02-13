gpu_id=1

for news_type in content headlines
do
  for emotion_type in historical sentiment emotion
  do
    run_name=${news_type}_${emotion_type}
    folder_path=./data/stocks/candle_w_emotion/day_average_${news_type}/
    output_dir=./results/stocks_fyp/stocks_${run_name}/

    # Zero-shot evaluation
    python MOIRAI_stocks_all.py \
    --folder_path $folder_path \
    --output_dir $output_dir \
    --run_name $run_name \
    --gpu_id $gpu_id

    # Prepare all stocks' training data
#    python prepare_train_data_stocks.py \
#    --dataset_name Stocks_${run_name} \
#    --folder_path $folder_path
#
#    # Finetuning
#    python -m cli.train \
#      -cp conf/finetune \
#      run_name=${run_name} \
#      model=moirai_1.0_R_large \
#      data=stocks_${news_type} \
#      val_data=stocks_${news_type} \
#      trainer.devices=[$gpu_id]
#
#    # Finetuned Evaluation
#    python MOIRAI_stocks_all.py \
#    --folder_path $folder_path \
#    --output_dir $output_dir \
#    --gpu_id $gpu_id \
#    --finetuned 1 \
#    --yaml_cfg stocks_${news_type} \
#    --run_name $run_name
  done
done
