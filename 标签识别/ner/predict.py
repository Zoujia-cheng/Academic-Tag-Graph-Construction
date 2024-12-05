import tensorflow as tf
import numpy as np
import os

from model_multitask_bert_wlf_wol2 import MyModel
from bert import modeling as bert_modeling
from utils import load_vocabulary
from utils import extract_kvpairs_in_bioes

# 加载词汇表和其他必要的配置
bert_vocab_path = "./bert_model/chinese_L-12_H-768_A-12/vocab.txt"
bert_config_path = "./bert_model/chinese_L-12_H-768_A-12/bert_config.json"
saved_model_checkpoint = "./ckpt/model.ckpt.batch100"  # 更新为实际的保存的检查点路径

w2i_char, i2w_char = load_vocabulary(bert_vocab_path)
w2i_word, i2w_word = load_vocabulary("./data3/vocab_word.txt")
w2i_bio, i2w_bio = load_vocabulary("./data3/vocab_bio.txt")
w2i_attr, i2w_attr = load_vocabulary("./data3/vocab_attr.txt")

# 构建模型架构
bert_config = bert_modeling.BertConfig.from_json_file(bert_config_path)
model = MyModel(bert_config=bert_config,
                vocab_size_word=len(w2i_word),
                vocab_size_bio=len(w2i_bio),
                vocab_size_attr=len(w2i_attr),
                O_tag_index=w2i_bio["O"],
                use_crf=False)

# 从检查点恢复模型
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, saved_model_checkpoint)

    input_data = {
    }

    # 执行推断
    feed_dict = {model.inputs_seq: [input_data["inputs_seq"]],
                 model.inputs_seq_word: [input_data["inputs_seq_word"]],
                 model.inputs_mask: [input_data["inputs_mask"]],
                 model.inputs_segment: [input_data["inputs_segment"]]}

    preds_seq_bio, preds_seq_attr = sess.run(model.outputs, feed_dict)


    preds_seq_bio_labels = [i2w_bio[i] for i in preds_seq_bio[0]]
    preds_seq_attr_labels = [i2w_attr[i] for i in preds_seq_attr[0]]

    # 如果需要，进行进一步的处理（例如，提取键值对）
    kv_pairs = extract_kvpairs_in_bioes(preds_seq_bio_labels, input_data["inputs_seq"], preds_seq_attr_labels)

    # 打印或根据需要使用预测结果
    print("预测的键值对：", kv_pairs)
