stage=4
stop_stage=4

export CUDA_VISIBLE_DEVICES="0"
train_config=./conf/train_conformer.yaml
checkpoint=
dir=exp_gpu_whisper_A2

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  mkdir -p $dir
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE=$dir/ddp_init
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"

  for ((i = 0; i < $num_gpus; ++i)); do
  {
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python train.py --gpu 0 \
      --config $train_config \
      --data_type 'raw' \
      --symbol_table '/data/wenet/examples/aishell/s1/utils/get_dict' \
      --train_data /data/wenet/examples/aishell/s1/utils/faster-whisper-aishell2/data.list  \
      --cv_data /data/wenet/examples/aishell/s1/utils/dev/data.list  \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir exp_gpu_whisper_A2/ \
      --num_workers 1 \
      --cmvn /data/wenet/examples/aishell/s1/utils/faster-whisper-aishell2/global_cmvn \
      --pin_memory
  } &
  done
  wait
fi

# ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5 eval"
  python3 recognize.py \
      --gpu 1 \
      --mode "attention_rescoring" \
      --config /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/train.yaml \
      --data_type 'raw' \
      --test_data /home/luo/hungarian-asr/data/prefile/eval_spont/data.list \
      --checkpoint /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/209.pt \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict /home/luo/hungarian-asr/data/dict/get_dict \
      --ctc_weight 0.5 \
      --reverse_weight 0.0 \
      --result_file /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/decoder/eval_spont_text
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6 export jit"
  python3 ./export_jit.py\
    --config /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/train.yaml \
    --checkpoint /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/209.pt \
    --output_file /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/export/final.zip \
    --output_quant_file /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/export/final_quant.zip
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 6 compute wer"
python3 tools/compute-wer.py --char=1 --v=1 /home/luo/hungarian-asr/data/prefile/eval_spont/text /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/decoder/attention_rescoring/eval_spont_text
# /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/decoder/rec_text_devre
fi

# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#   # 1) Prepare dict
#   # unit_file=/home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/units.txt
#   tools/fst/prepare_dict.py \
#     /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/units.txt \
#     /home/luo/hungarian-asr/data/prefile/eval_spont/lexicon.txt \
#     /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/lexicon.txt
#   fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # 2) Train lm
  /home/luo/hungarian-asr/srilm/bin/i686-m64/ngram-count \
    -text /home/luo/hungarian-asr/data/prefile/dev_spont/only_text \
    -order 3 \
    -write /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/G/gram.count

  /home/luo/hungarian-asr/srilm/bin/i686-m64/ngram-count \
    -read /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/G/gram.count \
    -order 3 \
    -lm /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/G/lm.arpa -interpolate -kndiscount
  fi

# TLG构图
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    tools/fst/compile_lexicon_token_fst.sh \
    tlg/dict tlg/tmp tlg/lang
    tools/fst/make_tlg.sh tlg tlg/lang tlg/lang_test

# 直接结果+测试集/ 用train集/用导师给的lm

  # 1）T
  # /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/tools/fst/compile_lexicon_token_fst.sh \
  #   /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/ \
  #   /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/tmp \
  #   /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/lang 

    #2)TLG(token lexicon lm)
  # tools/fst/make_tlg.sh \
  #   /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/G \
  #   /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/lang \
  #   /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/lang_test

fi

# decoder
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  chunk_size=-1
  ./tools/decode.sh --nj 1 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/lang_test/TLG.fst \
    --dict_path home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/lang_test/words.txt \
    /home/luo/hungarian-asr/data/prefile/eval_spont/wav.scp /home/luo/hungarian-asr/data/prefile/eval_spont/text /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/export/final.zip \
    home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/lang_test/units.txt /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/LM_decoder/lm_with_runtime


  # /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/tools/decode.sh --nj 1 \
  #     --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
  #       --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
  #       --fst_path /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/lang_test/TLG.fst \
  #       /home/luo/hungarian-asr/data/prefile/eval_spont/wav.scp /home/luo/hungarian-asr/data/prefile/eval_spont/text \
  #       /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/export/final_quant.zip \
  #       /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/lang_test/units.txt \
  #       /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/LM_decoder/lm_with_runtime

  # decoder_main --rescoring_weight 1.0 \
  #     --ctc_weight 0.5 \
  #     --rescoring_weight 0.0 \
  #     --chunk_size -1 \
  #     --wav_scp /home/luo/hungarian-asr/data/prefile/eval_spont/wav.scp \
  #     --model_path /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/exp_gpu/export/final.zip \
  #     --unit_path /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/tt/tlg/lang_test/units.txt \
  #     --fst_path /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/tt/tlg/lang_test/TLG.fst --beam 15.0 --dict_path /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/model_TLG/tt/tlg/lang_test/words.txt --lattice_beam 7.5 --max_active 7000 --min_active 200 --acoustic_scale 1.0 --blank_skip_thresh 0.98 --length_penalty 0.0 \
  #     --result /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/LM_decoder/decoder_tlg.txt
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "stage 6 compute wer"
python3 tools/compute-wer.py --char=1 --v=1 /home/luo/hungarian-asr/data/prefile/dev_spont/text /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/train-114_spok_4gram/decoder_tlg4.txt
# python3 tools/compute-cer.py --char=1 --v=1 /home/luo/hungarian-asr/data/prefile/dev_spont/text /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/train-114_spok_4gram/decoder_tlg4.txt
# python3 tools/speech_to_text_eval.py 
# --char=1 --v=1 /home/luo/hungarian-asr/data/prefile/dev_spont/text /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/luo/tlg/train-114_spok_4gram/decoder_tlg4.txt
fi


# stage=4
# stop_stage=4
# export CUDA_VISIBLE_DEVICES="0,1"
# train_config=./conf/train_conformer.yaml

# dir=exp_gpu

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   mkdir -p $dir
#   # You have to rm `INIT_FILE` manually when you resume or restart a
#   # multi-machine training.
#   INIT_FILE=$dir/ddp_init
#   init_method=file://$(readlink -f $INIT_FILE)
#   echo "$0: init method is $init_method"
#   num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
#   # Use "nccl" if it works, otherwise use "gloo"
#   dist_backend="gloo"
#   world_size=`expr $num_gpus \* $num_nodes`
#   echo "total gpus is: $world_size"

#   for ((i = 0; i < $num_gpus; ++i)); do
#   {
#     gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
#     # Rank of each gpu/process used for knowing whether it is
#     # the master of a worker.
#     rank=`expr $node_rank \* $num_gpus + $i`
#     python wenet/bin/train.py --gpu $gpu_id \
#       --config $train_config \
#       --data_type 'raw' \
#       --symbol_table '/home/luo/hungarian-asr/data/dict/get_dict' \
#       --train_data /home/luo/hungarian-asr/data/prefile/train/data.list  \
#       --cv_data /home/luo/hungarian-asr/data/prefile/dev_spont/data.list  \
#       ${checkpoint:+--checkpoint $checkpoint} \
#       --model_dir exp_gpu/ \
#       --num_workers 1 \
#       --cmvn ./global_cmvn \
#       --pin_memory
#   } &
#   done
#   wait
# fi
