import tensorflow as tf


class PolicyNet:
    """Policy Function Approximator Network."""

    def __init__(self, obs_shape, n_actions, fc1_units=200, lr=1e-3, scope="policy_net"):

        print("STATE SHAPE: {}, NUM ACTIONS: {}, FC1 UNITS: {}, LEARNING RATE: {}".format(obs_shape, n_actions, fc1_units, lr))

        with tf.variable_scope(scope):
            # batch env states
            self.env_state = tf.placeholder(tf.float32,
                                            shape=[None, obs_shape[0]],
                                            name="env_state")
            # batch discounted returns
            self.discounted_rewards = tf.placeholder(tf.float32,
                                                     shape=[None, ],
                                                     name="discounted_rewards")
            # batch index labels
            self.action_index = tf.placeholder(tf.int32,
                                               shape=[None, ],
                                               name="target")
            # Forward
            self.a = tf.layers.dense(self.env_state,
                                     units=fc1_units,
                                     activation=tf.nn.relu, name="a1")

            self.a = tf.layers.dense(self.a,
                                     units=fc1_units // 2,
                                     activation=tf.nn.relu, name="a2")

            # uncomment if 3 layers required
            # self.a = tf.layers.dense(self.a,
            #                          units=fc1_units // 4,
            #                          activation=tf.nn.relu, name="a3")

            self.logits = tf.layers.dense(self.a,
                                          units=n_actions,  # 6 Actions
                                          activation=None,
                                          name="logits")

            # batch probability of taking actions
            self.all_probs = tf.nn.softmax(self.logits, name="probab_distb")

            # Backward
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_index,
                                                                                logits=self.logits,
                                                                                name="loss_to_be_modified")

            # modulate the gradient with advantage (PG magic happens right here.)
            self.modulated_loss = tf.reduce_sum(cross_entropy_loss * self.discounted_rewards, name="loss")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

            self.train_op = self.optimizer.minimize(self.modulated_loss,
                                                    global_step=tf.contrib.framework.get_global_step())

    def magic_update(self, env_state, one_hot_picked_actions_index, discounted_rewards, sess=None):
        """Calculates the loss--> Modulates it using discounted_rewards--> Updates the policy"""
        sess = sess or tf.get_default_session()

        _, loss, shape1, shape2 = sess.run([self.train_op,
                                            self.modulated_loss,
                                            tf.shape(self.modulated_loss),
                                            tf.shape(self.discounted_rewards)],
                                            feed_dict={self.discounted_rewards: discounted_rewards.ravel(),
                                                      self.env_state: env_state,
                                                      self.action_index: one_hot_picked_actions_index.ravel()})

        return loss

    def predict(self, env_state, sess=None):
        """Predict Actions probab.."""
        sess = sess or tf.get_default_session()
        prob = sess.run(self.all_probs, feed_dict={self.env_state: env_state})
        return prob
