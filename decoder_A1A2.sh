stage=5
stop_stage=5

export CUDA_VISIBLE_DEVICES="0"
train_config=./conf/train_conformer.yaml
checkpoint=
dir=exp_gpu_A1_fw_A2
testdata=/data/experiment/wenet/examples/aishell/s1/test
# aishell2019/data/aishell2/eval-dataset/datalist/2019_data.list
# aishell1/data/experiment/wenet/examples/aishell/s1/test
# aishell2018/data/wenet/examples/aishell/s1/utils/test/data.list
# commonvoice/data/experiment/wenet/examples/aishell/s1/utils/commonvoice_test

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5 eval"
  python3 recognize.py \
      --gpu 1 \
      --mode "attention_rescoring" \
      --config /data/experiment/wenet/examples/aishell/s2/exp_gpu_A1_fw_A2/train.yaml \
      --data_type 'raw' \
      --test_data $testdata/data.list\
      --checkpoint /data/experiment/wenet/examples/aishell/s2/exp_gpu_A1_fw_A2/103.pt \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict /data/wenet/examples/aishell/s1/utils/get_dict \
      --ctc_weight 0.5 \
      --reverse_weight 0.0 \
      --result_file /data/experiment/wenet/examples/aishell/s2/decoder_A1A2/103pt_A1_text
  
fi
