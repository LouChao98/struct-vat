Our code is based on the public repo: https://github.com/yzhangcs/parser.git

Run with command   `python -m supar.cmds.crf_dep train -b -d 0 -c config/example.ini -p log/model.log -f char --no-ckpt --proj --tree --train data/ptb/fewshot/10percent.sampled.conllu --unsup data/ptb/fewshot/10percent.res.conllu`

See supar/models/dep.py:BiaffineDependencyModel for arguments.

