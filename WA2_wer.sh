stage=7
stop_stage=7

export CUDA_VISIBLE_DEVICES="0"
train_config=./conf/train_conformer.yaml
checkpoint=
dir=exp_gpu
testdir=/data/experiment/wenet/examples/aishell/s1/utils/commonvoice_test/text
#/data/experiment/wenet/examples/aishell/s1/utils/commonvoice_test
#aishell1 /data/experiment/wenet/examples/aishell/s1/utils/test
#asihell2018 /data/wenet/examples/aishell/s1/utils/test/text
#aishell2019 /data/aishell2/eval-dataset/trans/2019_trans.txt
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 6 compute wer"
python3 tools/compute-wer.py --char=1 --v=1 $testdir /data/experiment/wenet/examples/aishell/s2/decoder_WA2/108pt_CV_text
fi
