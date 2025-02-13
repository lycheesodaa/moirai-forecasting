run_name=daily
yaml_prefix='carbon'
target='Price'
gpu_id=0

for feats in 0 2 4 6 8 12
do
  csv_path="./data/carbon/res_daily/merged_data_imputed_top${feats}.csv"
  output_dir="./results/carbon_daily/feat${feats}/"

  # Pre-evaluation
#  python MOIRAI.py \
#  --csv_path $csv_path \
#  --run_name $run_name \
#  --target $target \
#  --yaml_prefix $yaml_prefix \
#  --gpu_id $gpu_id \
#  --output_dir $output_dir

  # Process dataset - dataset_name must be the same as in the YAML config file
#  python prepare_train_data.py \
#  --dataset_name Carbon_${run_name} \
#  --data_path $csv_path \
#  --target $target

  # Finetuning
  python -m cli.train \
    -cp conf/finetune \
    run_name=$run_name \
    model=moirai_1.0_R_large \
    data=carbon_${run_name} \
    val_data=carbon_${run_name} \
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
