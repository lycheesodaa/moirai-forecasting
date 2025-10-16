# note: remember to change the forecast horizon list in MOIRAI.py or MOIRAI_daily.py
run_name=monthly
yaml_prefix='carbon'
target='Price'
gpu_id=0

for feats_pct in 0 20
do
  csv_path="./data/carbon/res/merged_data.csv"
  train_csv_path="./data/carbon/res/merged_data_top${feats_pct}pct.csv"
  output_dir="./results/carbon_monthly/feat0.${feats_pct}/"

  # Pre-evaluation
  python MOIRAI_carbon_monthly.py \
  --csv_path $csv_path \
  --run_name $run_name \
  --target $target \
  --feats_pct $feats_pct \
  --yaml_prefix $yaml_prefix \
  --gpu_id $gpu_id \
  --output_dir $output_dir

  # Process dataset - dataset_name must be the same as in the YAML config file
#  python prepare_train_data.py \
#  --dataset_name Carbon_${run_name} \
#  --data_path $train_csv_path \
#  --target $target
#
#  # Finetuning
#  python -m cli.train \
#    -cp conf/finetune \
#    run_name=$run_name \
#    model=moirai_1.0_R_large \
#    data=carbon_${run_name} \
#    val_data=carbon_${run_name} \
#    train_dataloader.num_batches_per_epoch=2 \
#    trainer.devices=[$gpu_id]

  # Finetuned Evaluation
  python MOIRAI_carbon_monthly.py \
  --csv_path $csv_path \
  --run_name $run_name \
  --target $target \
  --feats_pct $feats_pct \
  --yaml_prefix $yaml_prefix \
  --gpu_id $gpu_id \
  --output_dir $output_dir \
  --finetuned 1
done
