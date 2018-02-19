

KERAS_BACKEND="theano" THEANO_FLAGS="device=gpu0,floatX=float32" python train.py \
--emb ../preprocessed_data/beer/w2v_embedding \
--domain beer \
-o output_dir \

