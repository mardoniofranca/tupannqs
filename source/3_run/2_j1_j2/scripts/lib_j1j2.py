# Standard library imports
import sys
import os
import warnings
import math
import time
import json
import glob
import random
from pathlib import Path
from fractions import Fraction

# Scientific computing
import numpy as np
import scipy
from scipy.stats import pearsonr, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# JAX ecosystem
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
import optax

# NetKet
import netket as nk
import netket.exact as nke
import netket.nn as nknn
import netket.experimental as nkx

# Data handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px

# Configuration
warnings.filterwarnings('ignore')
os.environ["JAX_PLATFORM_NAME"] = "cpu"

def n(f):
    return str(f).replace(".","_")

def create(data_path):
    path = Path(data_path)
    path.mkdir(parents=True, exist_ok=True)

def digts(number):
    if number < 10:
        return '0' + str(number)
    else:
        return str(number)

def theta(w):
    frac = Fraction(w, 180)
    return frac.numerator, frac.denominator

def ftheta(w):
    num, den = theta(w)
    if num !=0:
        if num != 1:
            theta_txt = "$\\theta = " + str(num)
        else:
            theta_txt = "$\\theta = "
        if den !=1 :
            theta_txt =  theta_txt + "\\pi/" + str(den) + "$"
        else:
            theta_txt =  theta_txt + "\\pi/" + str(den) + "$"
    else:
        theta_txt = "$\\theta = 0$"
    return theta_txt


trained_params_list = []; parameters_list = [];iii = []

def conf(J,L):
    # Define custom graph
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i + 1) % L, 1])
        edge_colors.append([i, (i + 2) % L, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)

    sigmaz = [[1, 0], [0, -1]]
    mszsz = np.kron(sigmaz, sigmaz)

    # Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),
        (J[1] * exchange).tolist(),
    ]

    bond_color = [1, 2, 1, 2]

    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    op = nk.operator.GraphOperator(
        hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color
    )

    return g,hi,op

class Jastrow(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs):
        k1, k2 = jax.random.split(rngs.params())
        self.J = nnx.Param(0.01 * jax.random.normal(k1, (N, N),
                                                    dtype=jnp.complex128))

        self.v_bias = nnx.Param(0.01 * jax.random.normal(k2, (N, 1),
                                                         dtype=jnp.complex128))

    def __call__(self, x):
        x = x.astype(jnp.complex128)              # keep the dtypes aligned
        quad = jnp.einsum('...i,ij,...j->...', x, self.J, x)
        lin  = jnp.squeeze(x @ self.v_bias, -1)   # (...,N) @ (N,1) → (...,1)
        return quad + lin

class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=2 * x.shape[-1],
            use_bias=True,
            param_dtype=np.complex128,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.normal(stddev=0.01),
        )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

class N4FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=4 * x.shape[-1],
            use_bias=True,
            param_dtype=np.complex128,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.normal(stddev=0.01),
        )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x



class Model2(nnx.Module):

    def __init__(self, N: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=N, out_features=2 * N, dtype=jnp.complex128, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=2 * N, out_features=N, dtype=jnp.complex128, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear1(x)
        x = nk.nn.activation.log_cosh(x)
        x = self.linear2(x)
        x = nk.nn.activation.log_cosh(x)
        return jnp.sum(x, axis=-1)

model    = FFNN()
modelN4  = N4FFNN()

def calc_exac_lanczos_ed(op,L,wT,data_path):
    E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
    exact_gs_energy = E_gs[0]

    exact_df = pd.DataFrame();    exact_v  = []
    exact_v.append(float(exact_gs_energy)); exact_row_df = pd.DataFrame([exact_v])
    exact_df = pd.concat([exact_row_df])
    exact_df.insert(0, 'id', range(1, 1 + len(exact_df)))
    exact_df.columns = ['id','value']

    create(data_path)

    e_path = data_path + "/exact_" + str(L)  + "_"  + str(wT) +  ".csv"; print(e_path)

	
    exact_df.to_csv(e_path)

    return float(exact_gs_energy), e_path

def calc_jastrow(hi,g,op,L,it,wT,run_id,data_path):
    print(it)
    ma  = Jastrow(N=hi.size, rngs=nnx.Rngs(0))
    sa  = nk.sampler.MetropolisExchange(hilbert=hi,graph=g)
    sr  = nk.optimizer.SR(diag_shift=0.1, holomorphic=True)
    opt = nk.optimizer.Sgd(learning_rate=0.01)
    vs  = nk.vqs.MCState(sa, ma, n_samples=1008)
    gs  = nk.VMC(hamiltonian=op, optimizer=opt, preconditioner=sr, variational_state=vs)

    create(data_path)
    v_out = data_path + '/jastrow_ID' + str(run_id)  + '_L_' + str(L) +  '_W_' + str(wT) + '_IT_' + str(it)
    print(v_out); print("---------------------")

    gs.run(out=v_out, n_iter=it)
    final_energy = float(gs.energy.mean.real)
    return final_energy,v_out

def calc_ffnn(hi,g,op,model,L,wt,it,run_id,data_path):
    paths = []

    print(it)
    sa    = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=2)
    vs    = nk.vqs.MCState(sa, model, n_samples=1008)
    opt   = nk.optimizer.Sgd(learning_rate=0.01)
    sr    = nk.optimizer.SR(diag_shift=0.01)
    gs    = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)

    create(data_path)

    r_out =  data_path + '/ffnn_ID_' + str(run_id)  + '_L_' + str(L) +  '_W_' + str(wt) + '_IT_' + str(it)
    print(r_out)
    print("---------------------")
    gs.run(out=r_out, n_iter=it, save_params_every=1, callback=save_params)

    head, body, bias_list,kernel_list = info(parameters_list[-1])
    real_df = pd.DataFrame()
    img_df  = pd.DataFrame()
    for param in parameters_list:
        head, body, bias_list,kernel_list = info(param)
        real_v = [];img_v = []
        for bias in bias_list:
            nr, ni = r_i(bias);
            real_v.append(nr)
            img_v.append(ni)

        real_row_df = pd.DataFrame([real_v])
        img_row_df  = pd.DataFrame([img_v])

        real_df = pd.concat([real_df,real_row_df])
        img_df  = pd.concat([img_df,img_row_df])


    real_df.insert(0, 'id', range(1, 1 + len(real_df)))
    img_df.insert(0, 'id', range(1, 1 + len(img_df)))

    path_nm = data_path + "/ffnn_ID_" + str(run_id)  + '_L_' + str(L) +  '_W_' + str(wt) + '_IT_' + str(it)

    real_df.to_csv(path_nm + "_bias_real.csv",index=None);  paths.append(path_nm + "_bias_real.csv")
    img_df.to_csv(path_nm  + "_bias_img.csv",index=None) ;  paths.append(path_nm + "_bias_img.csv")

    real_kernel_df = pd.DataFrame()
    img_kernel_df  = pd.DataFrame()
    for param in parameters_list:
        head, body, bias_list,kernel_list = info(param)
        real_v = [];img_v = []
        for ks in kernel_list:
            for k in ks.flatten():
                nr, ni = r_i(k);
                real_v.append(nr)
                img_v.append(ni)
        real_row_df = pd.DataFrame([real_v])
        img_row_df  = pd.DataFrame([img_v])

        real_kernel_df = pd.concat([real_kernel_df,real_row_df])
        img_kernel_df  = pd.concat([img_kernel_df,img_row_df])

    real_kernel_df.insert(0, 'id', range(1, 1 + len(real_kernel_df)))
    img_kernel_df.insert(0 , 'id', range(1, 1 + len(img_kernel_df)))

    real_kernel_df.to_csv(path_nm + "_kernel_real.csv",index=None); paths.append(path_nm + "_kernel_real.csv")
    img_kernel_df.to_csv(path_nm  + "_kernel_img.csv" ,index=None); paths.append(path_nm + "_kernel_img.csv")

    end = time.time()
    final_energy = float(gs.energy.mean.real)

    return final_energy,r_out,paths

def calc_nffnn(hi,g,op,model,L,wt,it,run_id,data_path):
    paths = []

    print(it)
    sa    = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=2)
    #model = Model2(N=hi.size, rngs=nnx.Rngs(1))
    vs    = nk.vqs.MCState(sa, model, n_samples=1008)
    opt   = nk.optimizer.Sgd(learning_rate=0.01)
    sr    = nk.optimizer.SR(diag_shift=0.01)
    gs    = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)



    create(data_path)
    r_out = data_path + '/nffnn_ID_' + str(run_id)  + '_L_' + str(L) +  '_W_' + str(wt) + '_IT_' + str(it)
    print(r_out)
    print("---------------------")
    gs.run(out=r_out, n_iter=it, save_params_every=1, callback=save_params)


    head, body, bias_list,kernel_list = info(parameters_list[-1])
    real_df = pd.DataFrame()
    img_df  = pd.DataFrame()
    for param in parameters_list:
        head, body, bias_list,kernel_list = info(param)
        real_v = [];img_v = []
        for bias in bias_list:
            nr, ni = r_i(bias);
            real_v.append(nr)
            img_v.append(ni)

        real_row_df = pd.DataFrame([real_v])
        img_row_df  = pd.DataFrame([img_v])

        real_df = pd.concat([real_df,real_row_df])
        img_df  = pd.concat([img_df,img_row_df])


    real_df.insert(0, 'id', range(1, 1 + len(real_df)))
    img_df.insert(0, 'id', range(1, 1 + len(img_df)))

    path_nm = data_path + "/nffnn_ID_" + str(run_id)  + '_L_' + str(L) +  '_W_' + str(wt) + '_IT_' + str(it)

    real_df.to_csv(path_nm + "_bias_real.csv",index=None);  paths.append(path_nm + "_bias_real.csv")
    img_df.to_csv(path_nm  + "_bias_img.csv",index=None) ;  paths.append(path_nm + "_bias_img.csv")

    real_kernel_df = pd.DataFrame()
    img_kernel_df  = pd.DataFrame()
    for param in parameters_list:
        head, body, bias_list,kernel_list = info(param)
        real_v = [];img_v = []
        for ks in kernel_list:
            for k in ks.flatten():
                nr, ni = r_i(k);
                real_v.append(nr)
                img_v.append(ni)
        real_row_df = pd.DataFrame([real_v])
        img_row_df  = pd.DataFrame([img_v])

        real_kernel_df = pd.concat([real_kernel_df,real_row_df])
        img_kernel_df  = pd.concat([img_kernel_df,img_row_df])

    real_kernel_df.insert(0, 'id', range(1, 1 + len(real_kernel_df)))
    img_kernel_df.insert(0 , 'id', range(1, 1 + len(img_kernel_df)))

    real_kernel_df.to_csv(path_nm + "_kernel_real.csv",index=None); paths.append(path_nm + "_kernel_real.csv")
    img_kernel_df.to_csv(path_nm  + "_kernel_img.csv" ,index=None); paths.append(path_nm + "_kernel_img.csv")

    end = time.time()
    final_energy = float(gs.energy.mean.real)

    return final_energy,r_out,paths

def info(e):
    head   = list(e.keys())[0]
    body   = list(e[head].keys())
    bias   = e[head][body[0]]
    kernel = e[head][body[1]]
    return  head, body, list(bias), list(kernel)
def real(c):
    return float(np.real(c))
def img(c):
    return float(np.imag(c))
def r_i(c):
    return real(c),img(c)

def save_params(step, params, energy):
    trained_params_list.append(params.copy())
    parameters_list.append(energy.state.parameters.copy())
    iii.append(1)
    return True