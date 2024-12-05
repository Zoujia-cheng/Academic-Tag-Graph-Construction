import numpy as np
import tensorflow as tf
from bert import modeling as bert_modeling


class MyModel(object):

    def __init__(self,
                 bert_config,
                 vocab_size_word,
                 vocab_size_bio,
                 use_lstm,
                 O_tag_index,
                 use_crf):


        self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq")  # B * (S+2)
        self.inputs_mask = tf.placeholder(tf.int32, [None, None], name="inputs_mask")  # B * (S+2)
        self.inputs_segment = tf.placeholder(tf.int32, [None, None], name="inputs_segment")  # B * (S+2)
        self.inputs_seq_word = tf.placeholder(tf.int32, [None, None], name="inputs_seq_word")
        self.outputs_seq = tf.placeholder(tf.int32, [None, None], name='outputs_seq')  # B * (S+2)


        inputs_seq_len = tf.reduce_sum(self.inputs_mask, axis=-1)  # B

        bert_model = bert_modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.inputs_seq,
            input_mask=self.inputs_mask,
            token_type_ids=self.inputs_segment,
            use_one_hot_embeddings=False
        )

        bert_outputs = bert_model.get_sequence_output()  # B * (S+2) * D

        with tf.variable_scope('embedding_layer'):
            embedding_matrix_word = tf.get_variable("embedding_matrix_word", [vocab_size_word, 300], dtype=tf.float32)
            embedded_word = tf.nn.embedding_lookup(embedding_matrix_word, self.inputs_seq_word) # B * S * D

        if not use_lstm:
            hiddens = tf.concat([bert_outputs, embedded_word], axis=2)

        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(hiddens, vocab_size_bio)  # B * (S+2) * V
            probs_seq = tf.nn.softmax(logits_seq)

            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq")  # B * S

        self.outputs = preds_seq

        with tf.variable_scope('loss'):
            if not use_crf:
                loss_bio = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq,
                                                                          labels=self.outputs_seq)  # B * S
                #masks_bio = tf.sequence_mask(inputs_seq_len, dtype=tf.float32)  # B * S
                #loss_bio = tf.reduce_sum(loss_bio * masks_bio, axis=-1) / tf.cast(inputs_seq_len, tf.float32)  # B
                masks_of_entity = tf.cast(tf.not_equal(self.outputs_seq, O_tag_index), tf.float32)  # B * S
                weights_of_loss = masks_of_entity + 0.5  # B  *S
                loss= loss_bio * weights_of_loss  # B * S

        self.loss = tf.reduce_mean(loss)

        with tf.variable_scope('opt'):
            params_of_bert = []
            params_of_other = []
            for var in tf.trainable_variables():
                vname = var.name
                if vname.startswith("bert"):
                    params_of_bert.append(var)
                else:
                    params_of_other.append(var)
            opt1 = tf.train.AdamOptimizer(1e-4)
            opt2 = tf.train.AdamOptimizer(1e-3)
            gradients_bert = tf.gradients(loss, params_of_bert)
            gradients_other = tf.gradients(loss, params_of_other)
            gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
            gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
            train_op_bert = opt1.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
            train_op_other = opt2.apply_gradients(zip(gradients_other_clipped, params_of_other))

        self.train_op = (train_op_bert, train_op_other)



