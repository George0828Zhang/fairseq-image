#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

fairseq_dir=$(realpath ../../../fairseq)
spm_train=${fairseq_dir}/scripts/spm_train.py
spm_encode=${fairseq_dir}/scripts/spm_encode.py

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <download_dir> <out_dir>"
  echo "e.g.: $0 /tmp/librispeech_raw/ ~/data/librispeech_final"
  exit 1
fi

PREP_JSON=$(pwd)/img_prep_json.py
download_dir=$(realpath ${1%/})
out_dir=${2%/}

mkdir -p ${out_dir}
cd ${out_dir} || exit

if [ ! -f $PREP_JSON ]; then
  echo "$PREP_JSON not found!" 
  exit 1
fi

nbpe=582
bpemode=word

# dict=data-tmp/train_${bpemode}${nbpe}_units.txt
encoded=data-tmp/train_${bpemode}${nbpe}_encoded.txt
fairseq_dict=data-tmp/train_${bpemode}${nbpe}_fairseq_dict.txt
bpemodel=data-tmp/train_${bpemode}${nbpe}
# echo "dictionary: ${dict}"
echo "Dictionary preparation"
mkdir -p data-tmp/
# echo "<unk> 3" > ${dict}
# echo "</s> 2" >> ${dict}
# echo "<pad> 1" >> ${dict}
cp ${download_dir}/im2latex_formulas.norm.lst data-tmp/input.txt
# spm_train --input=data-tmp/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --unk_id=3 --eos_id=2 --pad_id=1 --bos_id=-1 --character_coverage=1
python $spm_train --input=data-tmp/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1
python $spm_encode --model=${bpemodel}.model --output_format=piece < data-tmp/input.txt > ${encoded}
# echo "Generating dict"
# cat ${encoded} | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+3}' >> ${dict}
# echo "Generating fairseq-dict"
# cat ${encoded} | tr ' ' '\n' | sort | uniq -c | awk '{print $2 " " $1}' > ${fairseq_dict}
# cut -f1 ${bpemodel}.vocab | tail -n +4 | sed "s/$/ 100/g" > ${fairseq_dict}
cut -f1 ${bpemodel}.vocab | tail -n +4 | sed "s/$/ 100/g" > ${fairseq_dict}
wc -l ${fairseq_dict}

if [ ! -f ${download_dir}/im2latex_valid_filter.lst ]; then
    ln -s ${download_dir}/im2latex_validate_filter.lst ${download_dir}/im2latex_valid_filter.lst
fi

for part in train valid test; do
    echo "Prepare $part json"
    python $PREP_JSON --data-path ${download_dir} --split $part --spm-model ${bpemodel}.model --dictionary ${fairseq_dict} --output ${part}.json
done

cp ${fairseq_dict} ./dict.txt
cp ${bpemodel}.model ./spm.model
