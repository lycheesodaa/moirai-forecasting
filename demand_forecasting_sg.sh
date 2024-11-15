run_name=fast_eval
csv_path=./data/demand_data_all_cleaned_numerical.csv
gpu_id=0

# Pre-evaluation
#python MOIRAI.py \
#--csv_path $csv_path \
#--gpu_id $gpu_id

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
  val_data=demand

# Finetuned Evaluation
#python MOIRAI.py \
#--csv_path $csv_path \
#--gpu_id $gpu_id \
#--finetuned 1 \
#--yaml_cfg demand \
#--run_name $run_name
