run_name=fast_eval
csv_path=./data/demand_data_all_cleaned_numerical.csv
output_dir=./results/demand/
yaml_prefix='demand'
target='actual'
gpu_id=0

# Pre-evaluation
python MOIRAI.py \
--csv_path $csv_path \
--run_name $run_name \
--target $target \
--yaml_prefix $yaml_prefix \
--gpu_id $gpu_id \
--output_dir $output_dir

# Process dataset
python prepare_train_data.py \
--dataset_name Demand \
--data_path $csv_path

# Finetuning
python -m cli.train \
  -cp conf/finetune \
  run_name=$run_name \
  model=moirai_1.0_R_large \
  data=demand \
  val_data=demand \
  trainer.devices=[$gpu_id]

# Finetuned Evaluation
python MOIRAI.py \
--csv_path $csv_path \
--finetuned 1 \
--run_name $run_name \
--target $target \
--yaml_prefix $yaml_prefix \
--gpu_id $gpu_id \
--output_dir $output_dir

