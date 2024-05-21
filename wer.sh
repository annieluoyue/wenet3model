stage=7
stop_stage=7

export CUDA_VISIBLE_DEVICES="0"
train_config=./conf/train_conformer.yaml
checkpoint=
dir=exp_gpu
testdir=/data/experiment/wenet/examples/aishell/s1/utils/commonvoice_test
#aishell2019/data/aishell2/eval-dataset/trans
#aishell1/data/experiment/wenet/examples/aishell/s1/utils/test

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 6 compute wer"
python3 tools/compute-wer.py --char=1 --v=1 $testdir/text /data/experiment/wenet/examples/aishell/s2/decoder/91_CV_text
#/data/experiment/wenet/examples/aishell/s2/decoder/103_A2018_text
#/data/experiment/wenet/examples/aishell/s2/decoder_WA2/40pt_test_decode_text
# /home/luo/hungarian-asr/wenet/wenet/examples/aishell/s2/decoder/rec_text_devre
fi


