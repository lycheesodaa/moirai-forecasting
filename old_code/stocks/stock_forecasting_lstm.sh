gpu_id=0

for news_type in content headlines
do
  for emotion_type in historical sentiment emotion
  do
    run_name=${news_type}_${emotion_type}
    folder_path=./data/stocks/candle_w_emotion/day_average_${news_type}/
    output_dir=./results/stocks_fyp/stocks_${run_name}/

    python lstm_stocks.py \
    --folder_path $folder_path \
    --output_dir $output_dir \
    --gpu_id $gpu_id \
    --run_name $run_name
  done
done
