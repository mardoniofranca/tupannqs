import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import math
import json
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


log_t = "data/log/"
dfs_t = "data/df/"

#### a: angle in degrees; t: number of decimal places after truncation
def j_coupl(a,t):
    theta = math.radians(a); sin = math.sin(theta);cos = math.cos(theta)
    sin_t = math.trunc(cos * 10**t) / 10**t; cos_t = math.trunc(cos * 10**t) / 10**t
    J = [sin_t,cos_t]
    return J

class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=2*x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x
    
def config(J,L):
    # Define custom graph
    edge_colors = []
    for i in range(L):
        edge_colors.append([i, (i+1)%L, 1])
        edge_colors.append([i, (i+2)%L, 2])

    # Define the netket graph object
    g = nk.graph.Graph(edges=edge_colors)
    
    #Sigma^z*Sigma^z interactions
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz))

    #Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    bond_operator = [
        (J[0] * mszsz).tolist(),
        (J[1] * mszsz).tolist(),
        (-J[0] * exchange).tolist(),
        (J[1] * exchange).tolist(),
    ]

    bond_color = [1, 2, 1, 2]
    
    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)
    # Custom Hamiltonian operator
    return  g, hi, nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)

def st_fact(hi,L):
    sf = []
    sites = []
    structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(0, L):
        for j in range(0, L):
            structure_factor += (nk.operator.spin.sigmaz(hi, i)*nk.operator.spin.sigmaz(hi, j))*((-1)**(i-j))/L
    return structure_factor

# Loss function
def loss(params, structure_factor):
    output = model.apply({'params': params}, structure_factor)
    return jnp.mean(output)

# Callback Function to save params while train
def save_params(step, params, energy):
    trained_params_list.append(params.copy())
    parameters_list.append(energy.state.parameters.copy())
    iii.append(1)
    return True

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


def g_run(g,hi,model,n_s,op):
    # We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max = 2)

    # Construct the variational state
    vs = nk.vqs.MCState(sa, model, n_samples=n_s)

    # We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
    opt = nk.optimizer.Sgd(learning_rate=0.01)

    # Stochastic Reconfiguration
    sr = nk.optimizer.SR(diag_shift=0.01)

    # We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
    gs = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)   
    return gs


def run(A,T,L,N_IT,N_S,LOGP):
    J = j_coupl(A,T); g,hi,op = config(J,L);
    n_s = N_S; structure_factor = st_fact(hi,L)

    model = FFNN()

    gs = g_run(g,hi,model,n_s,op)
    path = LOGP + '_'+ str(L) + '_' + str(A) + '_' + str(N_IT) + '_' + str(N_S)
    print(path)
    gs.run(out=LOGP + '_'+ str(L) + '_' + str(A) + '_' + str(N_IT) + '_' + str(N_S), 
           n_iter=N_IT,  obs={'Structure Factor': structure_factor}, callback=save_params)
    return op, structure_factor, path

def result(path, N_IT):
    path = path + ".log"
    import json
    data=json.load(open(path))
    iters = data['Energy']['iters']
    energy=data['Energy']['Mean']['real']
    sf=data['Structure Factor']['Mean']['real']
    pos = (-1) * N_IT

    print(len(sf))

    print(r"Energy = {0:.3f}({1:.3f})".format(np.mean(energy[-1]), np.std(energy)/(np.sqrt(N_IT))))

    print(r"Structure factor = {0:.3f}({1:.3f})".format(np.mean(sf[-1]),
                                              np.std(np.array(sf))/np.sqrt(N_IT)))

    calc_energy,calc_struct_fac           = np.mean(energy[-1]),np.mean(sf[-1])
    std_calc_energy                       = np.std(energy)/(np.sqrt(N_IT))
    std_calc_struct_fac                   = np.std(np.array(sf))/np.sqrt(N_IT)

    return calc_energy,calc_struct_fac, std_calc_energy, std_calc_struct_fac


def exact_diag(op,structure_factor, run_exact):
    if run_exact :
         E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
         structure_factor_gs = (ket_gs.T.conj()@structure_factor.to_linear_operator()@ket_gs).real[0,0]
         exact_energy,exact_struct_fac = E_gs[0], structure_factor_gs
         print("Exact ground state energy = {0:.3f}".format(E_gs[0]))
         print("Exact Ground-state Structure Factor: {0:.3f}".format(structure_factor_gs))
         return exact_energy,exact_struct_fac

    else :
         return 0, 0



def main(A,T,L,N_IT,N_S,LOGP,TO_EXACT):

    op,structure_factor,path = run(A,T,L,N_IT,N_S,LOGP)

    calc_energy,calc_struct_fac,std_calc_energy, std_calc_struct_fac = result(path,N_IT)

    exact_energy,exact_struct_fac = exact_diag(op,structure_factor, TO_EXACT)

    df = pd.DataFrame({
    'degree'              : [A],
    'l'                   : [L],
    'n_it'                : [N_IT],
    'n_s'                 : [N_S],
    'calc_energy'         : [calc_energy],
    'exact_energy'        : [exact_energy],
    'calc_struct_fac'     : [calc_struct_fac],
    'exact_struct_fac'    : [exact_struct_fac],
    'std_calc_energy'     : [std_calc_energy],
    'std_calc_struct_fac' : [std_calc_struct_fac],
    'path'                : [path + ".csv"]
    })

    return df

#NN_IT = [30,50,100,200,400,600,1200,2400]
#NN_S  = [504,1008,2016,4032]

NN_IT = [100,300,600,1200]
NN_S  = [1200]
LL    = [32]
T     = 10
I     = 0
P     = log_t + 't'
LOGP  = dfs_t + 't'

for N_IT in NN_IT:
    for N_S in NN_S:
        for L in LL:
            for A in range (I,360,15):
                trained_params_list = []
                parameters_list     = []
                iii                 = []
                df = main(A,T,L,N_IT,N_S,LOGP, False)
                path_df = P + '_'+ str(L) + '_' + str(A) + '_' + str(N_IT) + '_' + str(N_S) + '.csv'
                r_path_df = P + '_'+ str(L) + '_' + str(A) + '_' + str(N_IT) + '_' + str(N_S) + '_R' + '.csv'
                i_path_df = P + '_'+ str(L) + '_' + str(A) + '_' + str(N_IT) + '_' + str(N_S) + '_I' + '.csv'


                df.to_csv(path_df, index=None)


                real_kernel_df = pd.DataFrame()
                img_kernel_df  = pd.DataFrame()

                for param in parameters_list:
                    head, body, bias_list,kernel_list = info(param)
                    real_v = [];img_v = []
                    for ks in kernel_list:
                        for k in ks:
                            nr, ni = r_i(k);
                            real_v.append(nr)
                            img_v.append(ni)
                        real_row_df = pd.DataFrame([real_v])
                        img_row_df  = pd.DataFrame([img_v])
                        real_kernel_df = pd.concat([real_kernel_df,real_row_df])
                        img_kernel_df  = pd.concat([img_kernel_df,img_row_df]) 

                real_kernel_df.to_csv(r_path_df, index=None)
                img_kernel_df.to_csv(i_path_df, index=None)



