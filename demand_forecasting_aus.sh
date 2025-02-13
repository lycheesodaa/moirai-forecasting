for run_name in top0 top5 top9 aus
do
  if [ "$run_name" = "aus" ]; then
    csv_path=./data/demand_data_all_nsw_numerical.csv
    output_dir=./results/demand_aus/
  else
    csv_path=./data/demand_data_all_nsw_${run_name}.csv
    output_dir=./results/demand_aus/${run_name}/
    run_name="aus_${run_name}"
  fi
  yaml_prefix='demand'
  target='actual'
  gpu_id=1
  echo "Running ${run_name}..."

  # Pre-evaluation
#  python MOIRAI.py \
#  --csv_path $csv_path \
#  --run_name $run_name \
#  --target $target \
#  --yaml_prefix $yaml_prefix \
#  --gpu_id $gpu_id \
#  --output_dir $output_dir

  # Process dataset
  python prepare_train_data.py \
  --dataset_name Demand_${run_name} \
  --data_path $csv_path

  # Finetuning
  python -m cli.train \
    -cp conf/finetune \
    run_name=$run_name \
    model=moirai_1.0_R_large \
    data=demand_${run_name} \
    val_data=demand_${run_name} \
    trainer.devices=[$gpu_id]

  # Finetuned Evaluation
  python MOIRAI.py \
  --csv_path $csv_path \
  --run_name $run_name \
  --target $target \
  --yaml_prefix $yaml_prefix \
  --gpu_id $gpu_id \
  --output_dir $output_dir \
  --finetuned 1
done