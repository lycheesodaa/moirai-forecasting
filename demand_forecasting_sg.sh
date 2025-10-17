for run_name in top0 featsel all
do
  if [ "$run_name" = "all" ]; then
    csv_path=./data/demand_data_all_cleaned.csv
    output_dir=./results/demand/
    run_name="sg"
  else
    csv_path=./data/demand_data_all_cleaned_${run_name}.csv
    output_dir=./results/demand/${run_name}/
    run_name="sg_${run_name}"
  fi
  yaml_prefix='demand'
  target='actual'
  gpu_id=0
  echo "Running ${run_name}..."

 # Pre-evaluation
 python MOIRAI_demand_hourly.py \
 --csv_path $csv_path \
 --run_name $run_name \
 --target $target \
 --yaml_prefix $yaml_prefix \
 --gpu_id $gpu_id \
 --output_dir $output_dir \
 --moirai_ver moirai_2.0_R_small

  # # Process dataset
  # python prepare_train_data.py \
  # --dataset_name Demand_${run_name} \
  # --data_path $csv_path

  # # Finetuning
  # python -m cli.train \
  #   -cp conf/finetune \
  #   run_name=$run_name \
  #   model=moirai_1.1_R_small \
  #   data=demand_${run_name} \
  #   val_data=demand_${run_name} \
  #   trainer.devices=[$gpu_id] \
  #   model.lr=1e-7 

  # # Finetuned Evaluation
  # python MOIRAI_demand_hourly.py \
  # --csv_path $csv_path \
  # --run_name $run_name \
  # --target $target \
  # --yaml_prefix $yaml_prefix \
  # --gpu_id $gpu_id \
  # --output_dir $output_dir \
  # --finetuned 1
done