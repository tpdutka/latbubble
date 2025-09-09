# src/latticesim/core/counterterms.py

def get_counterterms(order):
    LO = 1 if order >= 0 else 0
    NLO = 1 if order >= 1 else 0
    NNLO = 1 if order >= 2 else 0
    return LO, NLO, NNLO

def get_counterterms_constants(improved):
    if improved:
        Sigma = 2.75238391130752
        x_i   = -0.083647053040968
        C1    = 0.0550612
        C2    = 0.0334416
        C3    = -0.86147916
    else:
        Sigma = 3.17591153562522
        x_i   = 0.152859324966101
        C1    = 0
        C2    = 0
        C3    = 0.08848010
    return Sigma, x_i, C1, C2, C3