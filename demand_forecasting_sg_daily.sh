for run_name in top0 featsel all
do
  if [ "$run_name" = "all" ]; then
    csv_path=./data/demand_data_all_cleaned_daily.csv
    run_name="sg_daily"
  else
    csv_path=./data/demand_data_all_cleaned_daily_${run_name}.csv
    run_name="sg_daily_${run_name}"
  fi
  output_dir=./results/demand/daily/${run_name}/
  moirai_ver=moirai_1.1_R_small
  yaml_prefix='demand'
  target='actual'
  gpu_id=0
  echo "Running ${run_name}..."

  # Pre-evaluation
 python MOIRAI_demand_daily.py \
 --csv_path $csv_path \
 --run_name $run_name \
 --target $target \
 --yaml_prefix $yaml_prefix \
 --gpu_id $gpu_id \
 --output_dir $output_dir \
 --moirai_ver $moirai_ver

  # Process dataset
  python prepare_train_data.py \
  --dataset_name Demand_${run_name} \
  --data_path $csv_path

  # Finetuning
  python -m cli.train \
    -cp conf/finetune \
    run_name=$run_name \
    exp_name=${run_name}_${moirai_ver} \
    model=$moirai_ver \
    model.patch_size=32 \
    model.context_length=512 \
    model.prediction_length=365 \
    data=demand_${run_name} \
    val_data=demand_${run_name} \
    trainer.devices=[$gpu_id]

  # Finetuned Evaluation
  python MOIRAI_demand_daily.py \
  --csv_path $csv_path \
  --run_name $run_name \
  --target $target \
  --yaml_prefix $yaml_prefix \
  --gpu_id $gpu_id \
  --output_dir $output_dir \
  --finetuned 1
done