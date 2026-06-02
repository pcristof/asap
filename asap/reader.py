import numpy as np

def read_res(filename):
    '''This function is designed to parse the results in a typical raw output
    file.'''
    f = open(filename, 'r')
    for i, line in enumerate(f.readlines()):
        if line.strip()=="------:": continue
        sl = line.split()
        if ':' in line:
            if line.split(":")[-1].strip()=="": line+='0.0 0.0'
        if i==0:
            coeffs = [float(sl[i]) for i in range(len(sl))]
        elif i==1:
            ecoeffs = [float(sl[i]) for i in range(len(sl))]
        elif i==2:
            T = float(sl[0]); L = float(sl[1]); M = float(sl[2]); A = float(sl[3])
        elif i==3:
            eT = float(sl[0]); eL = float(sl[1]); eM = float(sl[2]); eA = float(sl[3])
        elif i==4:
            substr = line.split(':')[-1]
            bf = float(substr.split()[0])
            dbf = float(substr.split()[1])
        elif i==5:
            substr = line.split(':')[-1]
            avb = float(substr.split()[0])
            davb = float(substr.split()[1])
        elif i==6:
            if 'vb' not in line: ## sanity check
                print('read_res issue; vb not found in line {}'.format(i))
            substr = line.split(':')[-1]
            vb = float(substr.split()[0])
            dvb = float(substr.split()[1])
        elif i==7:
            if 'RV' not in line: ## sanity check
                print('read_res issue; RV not found in line {}'.format(i))
            substr = line.split(':')[-1]
            guessrv = float(substr.split()[0].replace('[', '').replace(']', ''))
        elif i==8:
            if 'RV' not in line: ## sanity check
                print('read_res issue; RV not found in line {}'.format(i))
            substr = line.split(':')[-1]
            rv = float(substr.split()[0])
            drv = float(substr.split()[1])
        elif i==9:
            if 'vsini' not in line: ## sanity check
                print('read_res issue; vsini not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            vsini = float(substr.split()[0])
            dvsini = float(substr.split()[1])
        elif i==10:
            if 'vmac' not in line: ## sanity check
                print('read_res issue; vmac not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            vmac = float(substr.split()[0])
            dvmac = float(substr.split()[1])
            vmacMode = line.split(':')[0].split('[')[-1].replace(']', '')
        elif i==13:
            if 'nb. of points' not in line.lower(): ## sanity check
                print('read_res issue; nb. of points not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            nbOfPoints = round(float(substr.strip()))
        elif i==14:
            if 'normfactor' not in line.lower(): ## sanity check
                print('read_res issue; normfactor not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            normfactor = float(substr.strip())
        elif i==15:
            if 'veilingfac' not in line.lower(): ## sanity check
                print('read_res issue; veilingfac not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            strvalues = substr.split()
            veilingfac = []
            for value in strvalues:
                veilingfac.append(float(value))
            veilingfac = np.array(veilingfac)
        elif i==16:
            if 'veilingfac' not in line.lower(): ## sanity check
                print('read_res issue; veilingfac not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            strvalues = substr.split()
            eveilingfac = []
            for value in strvalues:
                eveilingfac.append(float(value))
            eveilingfac = np.array(eveilingfac)
        elif i==17:
            if 'lum' not in line.lower(): ## sanity check
                print('read_res issue; lum not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            lum = float(substr.split()[0])
            dlum = float(substr.split()[1])
        elif i==18:
            if 'Mk' not in line: ## sanity check
                print('read_res issue; Mk not found in line {}'.format(i))
            substr = line.split(':')[-1]
            absmk = float(substr.split()[0])
            dabsmk = float(substr.split()[1])
        elif i==19:
            if 'dist' not in line: ## sanity check
                print('read_res issue; dist not found in line {}'.format(i))
            substr = line.split(':')[-1]
            dist = float(substr.split()[0])
            ddist = float(substr.split()[1])
    f.close()

    output = {"coeffs":coeffs, "ecoeffs":ecoeffs,
              "teff":T, "logg":L, "mh":M, "alpha":A,
              "dteff":eT, "dlogg":eL, "dmh":eM, "dalpha":eA,
              "vb":vb, "guessrv":guessrv, "rv":rv, "vsini":vsini, "vmac":vmac,
              "evb":dvb, "erv":drv, "evsini":dvsini, "evmac":dvmac,
              "lum":lum, "elum":dlum, "absmk":absmk, "eabsmk":dabsmk,
              "dist":dist, "edist":ddist,
              "bf":bf, "dbf":dbf,
              "avb":avb, "davb":davb,
              "nbpoints": nbOfPoints,
              "normfactor": normfactor,
              "veilingfac": veilingfac,
              "eveilingfac": eveilingfac,
              }
    return output


def read_res_v2(filename):
    '''New function aimed at replacing the old function.
    This function reads data from the results.txt file using type and keys.
    This allows for more flexibility while ensuring self.consistency.
    Input filename used : as a seperator between type, key and attributes'''
    supported_types = ['str', 'flt', 'cst', 'int', 'arr']
    data = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.strip()[0]=='#': continue ## Comments handling
            if line.strip()[0]=='': continue ## Empty line handling
            sl = line.split(':')
            ## Check file consistency
            if len(sl)!=3: raise Exception('Error reading file; '
                                        +'should contain 3 :-seperated columns')
            _type = sl[0].strip(); _var = sl[1].strip(); _value = sl[2].strip()
            if _type not in supported_types: 
                raise Exception('Error reading file; supported types are:'
                                        +' '.join(supported_types))
            ## Read for each type:
            if _type=='str': data[_var] = _value.strip()
            elif _type=='cst': data[_var] = float(_value) 
            elif _type=='int': data[_var] = int(_value) 
            elif _type=='flt':
                _val, _val_err = _value.split()
                data[_var] = float(_val) 
                data[_var+'_err'] = float(_val_err) 
            elif _type=='arr':
                _val_arr = _value.split()
                data[_var] = [float(_val_arr[i]) for i in range(len(_val_arr))] 
    return data