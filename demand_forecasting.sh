run_name=top9
csv_path=./data/demand_data_all_cleaned_top9.csv
gpu_id=1

# Pre-evaluation
python MOIRAI.py \
--csv_path $csv_path \
--desc_prefix _${run_name} \
--gpu_id $gpu_id

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
  val_data=demand_${run_name}

# Finetuned Evaluation
python MOIRAI.py \
--csv_path $csv_path \
--desc_prefix _${run_name} \
--gpu_id $gpu_id \
--finetuned 1 \
--yaml_cfg demand_${run_name} \
--run_name $run_name
