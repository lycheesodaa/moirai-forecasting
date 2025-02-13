run_name=sg_daily
csv_path=./data/demand_data_all_cleaned_numerical_daily.csv
output_dir=./results/demand/daily/
yaml_prefix='demand'
target='actual'
gpu_id=2

# Pre-evaluation
#  python MOIRAI_daily.py \
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
python MOIRAI_daily.py \
--csv_path $csv_path \
--run_name $run_name \
--target $target \
--yaml_prefix $yaml_prefix \
--gpu_id $gpu_id \
--output_dir $output_dir \
--finetuned 1
