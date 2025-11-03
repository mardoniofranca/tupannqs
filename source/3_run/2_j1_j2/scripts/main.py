from lib_j1j2 import *
#it = 300; L = 14; I_W = 356; L_W = 360; L_E = 5
it = 10; L = 8
#wTs = [0,15,20,22,23,24,25,30,45,60,90,120,150,170,171,172,175,176,177,180,210,225,269,270,271,300,315,330]; L_E = 100

wTs  = [0,30,45,60,120,150,210,225,245,300,315,330]; L_E = 100

#for wT in range(I_W,L_W):
for wT in wTs:
    for e in range(0,L_E):
        print(wT,e)
        data_path = "data/it_" + str(it) + "_l_" + str(L) + "/exec_" + digts(e); print(data_path); run_id    = digts(e)
        create(data_path)
        theta_txt = ftheta(wT); rad  = math.radians(wT); sin = math.sin(rad); cos = math.cos(rad)
        j1        = sin; j2 = cos; J = [j1,j2]; trained_params_list  = []; parameters_list = [];iii = []
        g,hi,op = conf(J,L)
        if e == 0 :
            exact_gs_energy1,e_path1 = calc_exac_lanczos_ed(op,L,wT,data_path)
            calc_j_gs_energy1,j_out1 = calc_jastrow(hi,g,op,L,it,wT,run_id,data_path)
        final_energy,r_out,paths = calc_ffnn (hi,g,op,model,L,wT,it,run_id,data_path)