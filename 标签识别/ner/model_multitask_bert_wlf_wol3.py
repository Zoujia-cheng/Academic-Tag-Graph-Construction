import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, BertModel

class MyModel(object):

    def __init__(self,
                 vocab_size_wordpiece,
                 vocab_size_bio,
                 vocab_size_attr,
                 O_tag_index,
                 use_crf):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.inputs_text = tf.placeholder(tf.string, [None], name="inputs_text")
        self.inputs_seq_wordpiece = tf.placeholder(tf.int32, [None, None], name="inputs_seq_wordpiece")
        self.inputs_mask = tf.placeholder(tf.int32, [None, None], name="inputs_mask")  # B * (S+2)
        self.outputs_seq_bio = tf.placeholder(tf.int32, [None, None], name='outputs_seq_bio')  # B * (S+2)
        self.outputs_seq_attr = tf.placeholder(tf.int32, [None, None], name='outputs_seq_attr')  # B * (S+2)

        bert_model = BertModel.from_pretrained("bert-base-chinese")
        bert_outputs = bert_model(input_ids=self.inputs_seq_wordpiece, attention_mask=self.inputs_mask)[0]  # B * S * D

        with tf.variable_scope('bio_projection'):
            if not use_crf:
                logits_bio = tf.layers.dense(bert_outputs, vocab_size_bio)  # B * S * V
                probs_bio = tf.nn.softmax(logits_bio, axis=-1)
                preds_bio = tf.argmax(probs_bio, axis=-1, name="preds_bio")  # B * S

        with tf.variable_scope('attr_projection'):
            logits_attr = tf.layers.dense(bert_outputs, vocab_size_attr)  # B * S * V
            probs_attr = tf.nn.softmax(logits_attr, axis=-1)
            preds_attr = tf.argmax(probs_attr, axis=-1, name="preds_attr")  # B * S

        self.outputs = (preds_bio, preds_attr)

        with tf.variable_scope('loss'):
            if not use_crf:
                loss_bio = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bio,
                                                                          labels=self.outputs_seq_bio)  # B * S
                #masks_bio = tf.sequence_mask(inputs_seq_len, dtype=tf.float32)  # B * S
                #loss_bio = tf.reduce_sum(loss_bio * masks_bio, axis=-1) / tf.cast(inputs_seq_len, tf.float32)  # B
                masks_of_entity = tf.cast(tf.not_equal(self.outputs_seq_bio, O_tag_index), tf.float32)  # B * S
                weights_of_loss = masks_of_entity + 0.5  # B  *S
                loss_bio = loss_bio * weights_of_loss  # B * S

            loss_attr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_attr,
                                                                       labels=self.outputs_seq_attr)  # B * S
            masks_attr = tf.cast(tf.not_equal(preds_bio, O_tag_index), tf.float32)  # B * S
            loss_attr = tf.reduce_sum(loss_attr * masks_attr, axis=-1) / (tf.reduce_sum(masks_attr, axis=-1) + 1e-5)  # B
            loss_attr_expanded = tf.expand_dims(loss_attr, axis=-1)  # Expand dimensions to (?, 1)
            loss = loss_bio + loss_attr_expanded  # B

        self.loss = tf.reduce_mean(loss)

        with tf.variable_scope('opt'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            gradients = optimizer.compute_gradients(self.loss)
            gradients_clipped, _ = tf.clip_by_global_norm([grad for grad, var in gradients], 5.0)
            self.train_op = optimizer.apply_gradients(zip(gradients_clipped, [var for grad, var in gradients]))
