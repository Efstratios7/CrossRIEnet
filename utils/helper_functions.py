import tensorflow as tf
from keras import backend as K

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@tf.function(reduce_retracing=True)
def _symmetrize(M):
    return 0.5 * (M + tf.linalg.matrix_transpose(M))

@tf.function(reduce_retracing=True)
def svd_via_eigh_full(C, eps=None, jitter_eigh=0.0):
   """
   Batched SVD via eigh(CC^T) con V_full costruita in modo coerente con U_k e s_k.
   C: [B, n, m]
   Ritorna:
     s_k    [B, r]        (r = min(n,m), singolari in ordine decrescente)
     U_k    [B, n, r]
     V_full [B, m, m]     (prime r colonne = right singular vectors coerenti)
   """
   C = tf.convert_to_tensor(C)
   if eps is None:
       eps = tf.cast(K.epsilon(), C.dtype)

   B = tf.shape(C)[0]
   n = tf.shape(C)[1]
   m = tf.shape(C)[2]
   r = tf.minimum(n, m)
   m_minus_r = m - r

   # 1) U_k e s_k da eigh(CC^T), simmetrizzato
   A = tf.matmul(C, C, transpose_b=True)            # [B, n, n]
   A = _symmetrize(A)

   if jitter_eigh != 0.0:
       I_n = tf.eye(n, batch_shape=[B], dtype=C.dtype)
       A = A + tf.cast(jitter_eigh, C.dtype) * I_n  # piccolo jitter sulla diagonale

   evals_u, U_full = tf.linalg.eigh(A)              # autovalori in ordine crescente

   # Ordina in ordine decrescente
   idx_u = tf.argsort(evals_u, direction="DESCENDING")
   evals_u = tf.gather(evals_u, idx_u, batch_dims=1, axis=1)    # [B, n]
   U_full = tf.gather(U_full, idx_u, batch_dims=1, axis=2)      # [B, n, n]

   # Singolari = sqrt(max(evals, 0))
   zeros_evals = tf.zeros_like(evals_u)
   s_all = tf.sqrt(tf.maximum(evals_u, zeros_evals))            # [B, n]

   U_k = U_full[:, :, :r]                                       # [B, n, r]
   s_k = s_all[:, :r]                                           # [B, r]
   s_safe = tf.maximum(s_k, eps)

   # 2) prime r colonne di V: V1 = C^T U_k / s_k
   V1 = tf.matmul(C, U_k, transpose_a=True)                     # [B, m, r]
   V1 = V1 / tf.expand_dims(s_safe, axis=1)                     # [B, m, r]

   # 3) normalizza colonne di V1 e compensa in s_k
   norms = tf.maximum(tf.linalg.norm(V1, axis=1), eps)          # [B, r]
   V1 = V1 / tf.expand_dims(norms, axis=1)                      # [B, m, r]
   s_k = s_k * norms                                            # [B, r]

   I_m = tf.eye(m, batch_shape=[B], dtype=C.dtype)              # [B, m, m]
   W0 = I_m[:, :, r:]                                           # [B, m, m-r]

   # Proietta W0 sul complemento di span(V1):
   # V1^T W0: [B, r, m-r]
   V1tW0 = tf.matmul(V1, W0, transpose_a=True)
   # W1 = W0 - V1 (V1^T W0): [B, m, m-r]
   W1 = W0 - tf.matmul(V1, V1tW0)

   # QR di W1: le colonne di Q sono ortonormali e (idealmente) nel complemento di span(V1)
   # Funziona anche quando m_minus_r == 0 (dimensione nulla).
   V2, _ = tf.linalg.qr(W1, full_matrices=False)                # [B, m, m-r]

   # V_full = [V1 | V2]
   V_full = tf.concat([V1, V2], axis=2)                         # [B, m, m]

   return s_k, U_full, V_full

@tf.function(reduce_retracing=True)
def diag_vtCv(C, V):
    """
    C: [B, n, n]
    V: [B, n, r]  (colonne = vettori v_j)
    return: [B, r] con diag(V^T C V) per batch
    """
    CV = tf.matmul(C, V)           # [B, n, r]
    d  = tf.reduce_sum(V * CV, axis=1)  # somma su dimensione n -> [B, r]
    return d

def pad_A_to_B_simple(A, B):
    n_a = tf.shape(A)[1]
    n_b = tf.maximum(tf.shape(B)[1], tf.shape(B)[2])
    
    pad_len = n_b - n_a
    paddings = tf.stack([[0, 0], [0, pad_len], [0, 0]], axis=0)
    return tf.pad(A, paddings, mode="CONSTANT", constant_values=0)

@tf.function(reduce_retracing=True)
def svd_reconstruct_from_full(s_k, U_full, V_full):
    """
    Ricostruisce C_hat = U_k @ diag(s_k) @ (V_full[:,:,:r])^T
    """
    r = tf.shape(s_k)[1]
    V_k = V_full[:, :, :r]   
    U_k = U_full[:, :, :r]   # [B, m, r]
    S = tf.linalg.diag(s_k)                             # [B, r, r]
    return tf.matmul(U_k, tf.matmul(S, V_k, transpose_b=True))
