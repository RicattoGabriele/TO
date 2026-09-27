# In the OC update loop:
# l1, l2, move = 1e-9, 1e9, 0.2
# while (l2 - l1) / (l1 + l2) > 1e-4:
#     lmid = 0.5 * (l2 + l1)
#     Be = np.sqrt(np.maximum(0.0, -dc_filtered / (lmid * np.maximum(1e-12, dv_filtered))))
#     xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * Be))))
#     if np.mean(xnew) > self.volfrac:
#         l1 = lmid
#     else:
#         l2 = lmid

# The user is correct. xnew are the design variables. But the volume fraction constraint is on xPhys_new!
# The volume is np.mean(xPhys_new).
# We must compute xPhys_new from xnew inside the loop to check the volume constraint correctly.

# x_tilde_new = np.array(self.H @ xnew).flatten() / self.Hs
# xPhys_new = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde_new - eta))) / denom
# if np.mean(xPhys_new) - self.volfrac > 0:
#     l1 = lmid
# else:
#     l2 = lmid
print("I will modify the code to check the volume on xPhys_new inside the bisection loop.")
