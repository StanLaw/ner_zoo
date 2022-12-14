
target=bert-base-chinese

mkdir $target
cd $target

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz

tar -zxvf bert-base-chinese.tar.gz

wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt

