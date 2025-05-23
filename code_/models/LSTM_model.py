import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3, clip=5.0):
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.lr, self.clip = lr, clip
        def xavier(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6/(fan_in+fan_out))
            return np.random.uniform(-limit, limit, size=shape)
        D, H = input_dim+hidden_dim, hidden_dim
        self.Wf, self.bf = xavier((D,H)), np.zeros(H)
        self.Wi, self.bi = xavier((D,H)), np.zeros(H)
        self.Wc, self.bc = xavier((D,H)), np.zeros(H)
        self.Wo, self.bo = xavier((D,H)), np.zeros(H)
        self.Wy, self.by = xavier((hidden_dim, output_dim)), np.zeros(output_dim)
        self.m, self.v = {}, {}
        for name in ['Wf','bf','Wi','bi','Wc','bc','Wo','bo','Wy','by']:
            self.m[name] = np.zeros_like(getattr(self,name))
            self.v[name] = np.zeros_like(getattr(self,name))
        self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0

    @staticmethod
    def sigmoid(x): return 1/(1+np.exp(-x))

    def forward(self, x):
        """
        x: (seq_len, batch, input_dim)
        Returns y_pred: (seq_len, batch, output_dim)
        """
        seq_len, B, _ = x.shape
        h, c = np.zeros((B,self.hidden_dim)), np.zeros((B,self.hidden_dim))
        self.cache = []
        y_seq = np.zeros((seq_len, B, self.output_dim))
        for t in range(seq_len):
            xt = x[t]
            z = np.hstack([h, xt])  # (B, D+H)
            f = self.sigmoid(z.dot(self.Wf) + self.bf)
            i = self.sigmoid(z.dot(self.Wi) + self.bi)
            c_tilde = np.tanh(z.dot(self.Wc) + self.bc)
            c = f*c + i*c_tilde
            o = self.sigmoid(z.dot(self.Wo) + self.bo)
            h = o * np.tanh(c)
            y = h.dot(self.Wy) + self.by
            y_seq[t] = y
            self.cache.append((z, f, i, c_tilde, c, o, h))
        return y_seq

    def backward(self, x, dy):
        """
        x: (seq_len, B, input_dim), dy: (seq_len, B, output_dim)
        """
        grads = {n: np.zeros_like(getattr(self,n)) for n in self.m}
        dh_next = np.zeros((x.shape[1], self.hidden_dim))
        dc_next = np.zeros_like(dh_next)
        for t in reversed(range(x.shape[0])):
            z, f, i, c_tilde, c, o, h = self.cache[t]
            dy_t = dy[t]
            grads['Wy'] += h.T.dot(dy_t)
            grads['by'] += dy_t.sum(axis=0)
            dh = dy_t.dot(self.Wy.T) + dh_next
            do = dh * np.tanh(c)
            do_raw = do * o*(1-o)
            # cell
            dc = dh * o * (1 - np.tanh(c)**2) + dc_next
            di = dc * c_tilde
            di_raw = di * i*(1-i)
            df = dc * self.cache[t-1][4] if t>0 else dc * 0
            df_raw = df * f*(1-f)
            dc_tilde = dc * i
            dc_tilde_raw = dc_tilde * (1 - c_tilde**2)

            for name, dgate in [('Wf',df_raw),('bf',df_raw),
                                ('Wi',di_raw),('bi',di_raw),
                                ('Wc',dc_tilde_raw),('bc',dc_tilde_raw),
                                ('Wo',do_raw),('bo',do_raw)]:
                if name.startswith('W'):
                    grads[name] += z.T.dot(dgate)
                else:
                    grads[name] += dgate.sum(axis=0)

            dz = (df_raw.dot(self.Wf.T)
                  + di_raw.dot(self.Wi.T)
                  + dc_tilde_raw.dot(self.Wc.T)
                  + do_raw.dot(self.Wo.T))
            dh_next = dz[:, :self.hidden_dim]
            dc_next = f * dc

        for name, grad in grads.items():
            np.clip(grad, -self.clip, self.clip, out=grad)
            self._adam(name, grad)

    def _adam(self, name, grad):
        self.t += 1
        m, v = self.m[name], self.v[name]
        m[:] = self.beta1*m + (1-self.beta1)*grad
        v[:] = self.beta2*v + (1-self.beta2)*(grad**2)
        m_hat = m/(1-self.beta1**self.t)
        v_hat = v/(1-self.beta2**self.t)
        setattr(self, name, getattr(self,name) - self.lr * m_hat/(np.sqrt(v_hat)+self.eps))

    def train(self, X, Y, epochs=10, batch_size=32, val_data=None):
        """
        X: (N, seq_len, input_dim)
        Y: (N, seq_len, output_dim)
        """
        N = X.shape[0]
        for ep in range(1, epochs+1):
            perm = np.random.permutation(N)
            X, Y = X[perm], Y[perm]
            loss = 0
            for i in range(0, N, batch_size):
                xb = X[i:i+batch_size].transpose(1,0,2)
                yb = Y[i:i+batch_size].transpose(1,0,2)
                preds = self.forward(xb)
                dy = (preds - yb) / yb.shape[1]
                loss += np.mean((preds - yb)**2)
                self.backward(xb, dy)
            loss /= (N//batch_size)
            msg = f"Epoch {ep}/{epochs} — train MSE: {loss:.4f}"
            if val_data:
                Xv,Yv = val_data
                pv = self.forward(Xv.transpose(1,0,2))
                vloss = np.mean((pv - Yv.transpose(1,0,2))**2)
                msg += f" — val MSE: {vloss:.4f}"
            print(msg)

    def predict(self, X):
        """
        X: (N, seq_len, input_dim)
        returns (N, seq_len, output_dim)
        """
        out = self.forward(X.transpose(1,0,2))
        return out.transpose(1,0,2)

