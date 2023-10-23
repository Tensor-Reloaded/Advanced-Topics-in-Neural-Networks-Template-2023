import torch


class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, t, dw, db):
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)
        # *** biases *** #
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db ** 2)

        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        v_db_corr = self.v_db / (1 - self.beta2 ** t)

        w_modif = 0.001 * (m_dw_corr / (torch.sqrt(v_dw_corr) + self.epsilon))
        b_modif = 0.001 * (m_db_corr / (torch.sqrt(v_db_corr) + self.epsilon))
        return w_modif, b_modif


# Source which helped me implement the class:
# https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
