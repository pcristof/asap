def read_vald_online(filename):
    ## These are the keys in which the output is ordered
    keys = ['wvl', 'element', 'ion', 'lggf', 'chi_low', 'j_low', 'chi_up', 'j_up', 'lande_eff', 
            'lande_low', 'lande_up', 'gamma_rad', 'gamma_stark', 'gamma_waals', 'transilow', 'transihigh',
            'coupling_low', 'config_low', 'coupling_up', 'config_up']

    data = []
    f = open(filename)
    for line in f.readlines():
        dataline = {}
        sl = line.split(",")
        ## The last two are the transition strings. Let's get the coupling:
        subsl = sl[-2].replace("'", "").strip().split()
        couplinglow = subsl[0] ## Low coupling
        transilow = " ".join(subsl[1:]) ## The rest is the transition
        ## Now for the upper level:
        subsl = sl[-1].replace("'", "").strip().split()
        couplingup = subsl[0] ## up coupling
        transiup = " ".join(subsl[1:]) ## The rest is the transition
        ## Add those things to the list
        sl.append(couplinglow)
        sl.append(transilow)
        sl.append(couplingup)
        sl.append(transiup)
        ## Populate a dictionary
        for ik in range(len(keys)):
            dataline[keys[ik]] = sl[ik]
        ## Add the spec_ion key:
        dataline['spec_ion'] = " ".join([dataline['element'], dataline['ion']])
        data.append(dataline)
    f.close()
    return data


def read_vald(filenames):
    '''Reading the extended VALD extract stellar format
    First two lines are ignored
    Third line contains headers
    Of course the table is not comma seperated everywhere, 
    that would be too easy.
    
    This function assumes datablocks of 4 lines. If the length of the first line in the datablock
    is lower than expected, I assume we have reached the end of the file and break the loop.
    The function returns a list of dictionaries containing the data from each block.
    '''

    blockline = 0 ## Iterator to help know where we are at

    datalist = []
    header = ""
    tail = ""

    if isinstance(filenames, list):
        filelist = filenames
    else:
        filelist = [filenames]

    ## What is the file format?
    ## VALD provide multiple options, included extract stellar and extract all
    ## If if is extract all there will be one less column (the depth) but also one less line in the header
    ## To decide what the format of the file is, I get the first 3 lines and see which line contains
    ## Lande factors       Damping parameters
    for filename in filelist:
        f = open(filename)
        lines = f.readlines()
        if "WARNING" in lines[0]:
            print(lines[0])
            # lines = lines[1:]
            idxstart = 1
        else:
            idxstart = 0
        if 'Lande factors' in lines[idxstart]:
            filetype = 'extract_all'
        elif 'Lande factors' in lines[idxstart+1]:
            filetype = 'extract_stellar'
        else:
            raise Exception('File type not recognized')

    if filetype=='extract_stellar':
        header_length = 3
        keys = ['spec_ion', 'wvl', 'lggf', 'chi_low', 'j_low', 'chi_up', 'j_up', 'lande_low', 'lande_up', 'lande_mean', 'gamma_rad', 'gamma_stark', 'gamma_waals', 'depth']
    elif filetype=='extract_all':
        header_length = 2
        keys = ['spec_ion', 'wvl', 'lggf', 'chi_low', 'j_low', 'chi_up', 'j_up', 'lande_low', 'lande_up', 'lande_mean', 'gamma_rad', 'gamma_stark', 'gamma_waals']
    # refkeys = ['ref_wvl', 'ref_logg', 'ref_3', 'ref_4', 'ref_5', 'ref_6', 'ref_7', 'ref_8', 'elcheck']

    lines = lines[idxstart:]

    for filename in filelist:
        f = open(filename)
        storetail = False
        ignoreblock = False
        for i, line in enumerate(lines):
            storeblock = False
            if storetail:
                tail+=line
                continue  
            if i<=(header_length-1): 
                header+=line
                continue
            if i>(header_length-1):
                if blockline==4: 
                    ignoreblock = False
                    blockline = 0 ## new block

                if blockline==0: ## first line of the block
                    full_block_string = line
                    _linedata = {} ## Define the data dictionary
                    ## For each element of the first line, add it to the dictionary
                    sl = line.split(',')
                    ## Is this a datablock?
                    if len(sl)<len(keys): ## this is not a datablock 
                        # print('break')
                        storetail=True
                        continue
                        # break ## We have reached the end of the data, this was not a data block
                    for j in range(len(keys)):
                        _linedata[keys[j]] = sl[j].strip()
                    blockline+=1
                elif blockline==1: ## second line is lower energy configuration
                    full_block_string+=line
                    ## This leads to issues sometimes, some coupling are missing.
                    # sl = line.replace("'", "").split() 
                    sl = []
                    sl.append(line[:30].strip())
                    sl.append(line[30:].strip())
                    _linedata['coupling_low'] = sl[0].strip() ## That is the first coupling
                    _linedata['config_low'] = " ".join(sl[1:]).strip() ## All the rest is the configuration
                    blockline+=1
                elif blockline==2: ## third line is upper energy configuration
                    full_block_string+=line
                    # ## PIC: I observed some files with a missing line here (empyty line)
                    # if line.strip()=="":
                    #     ignoreblock = True
                    #     blockline+=1
                    #     continue ## Simply ignore the whole block of code
                    ## This leads to issues sometimes, some coupling are missing.
                    # sl = line.replace("'", "").split() 
                    sl = []
                    sl.append(line[:30].strip())
                    sl.append(line[30:].strip())
                    _linedata['coupling_up'] = sl[0].strip() ## That is the first coupling
                    _linedata['config_up'] = " ".join(sl[1:]).strip() ## All the rest is the configuration
                    blockline+=1
                elif blockline==3: ## fourth line contains reference
                    full_block_string+=line
                    _linedata['full_block_string'] = full_block_string
                    _linedata['refs'] = line.replace("'", "").strip()
                    # comment = line.replace("'", "")[:30]
                    # _linedata['comment'] = comment
                    # sl = line[31:].replace("'", "").replace('_', "").split()
                    # j = 0
                    # print(sl)
                    # while j<len(sl)-1:
                    #     _linedata[refkeys[j//2]] = "{} {}".format(sl[j], sl[j+1])
                    #     j+=2
                    blockline+=1
                    storeblock = True ## Store block to avoid duplicates
            if not ignoreblock:
                if _linedata: ## Ensure that the dictionary is not empty
                    if storeblock: ## Avoid storing duplicates
                        datalist.append(_linedata)
        f.close()
    return datalist, header, tail

def write_vald(datalist, header, tail, outputfile='test.vald'):
    ostr=header
    ## Go through the blocks
    for block in datalist:
        lines = block['full_block_string'].split('\n')
        ## Go through lines in block:
        ostr+=f"{block['spec_ion']},{block['wvl']:>17},{block['lggf']:>7},{block['chi_low']:>8},{block['j_low']:>5},{block['chi_up']:>5},{block['j_up']:>5},{block['lande_low']:>7},{block['lande_up']:>7},{block['lande_mean']:>7},{block['gamma_rad']:>6},{block['gamma_stark']:>6},{block['gamma_waals']:>7},{block['depth']:>6},\n"
        ostr+=lines[1]+'\n'
        ostr+=lines[2]+'\n'
        ostr+=lines[3]+'\n'
    ostr+=tail
    with open(outputfile, 'w') as f:
        f.write(ostr)

def is_atom(el):
    ## How many upper cases do we have in this?
    # nbUpperCase = sum([el[i].isupper() for i in range(len(el))])
    # if nbUpperCase<2:
    #     newdata.append(block)    
    species = get_species_codes() ## Those are all the species we know
    if el in species.keys():
        isAtom = True
    else:
        isAtom = False
    return isAtom

def get_atoms(data):
    '''Extract the data comming from atoms only'''
    ## Go through the data and only retain atoms
    newdata = []
    for block in data:
        el = block['spec_ion'].replace("'", "").split()[0] ## The species without ionization
        ## We say it is a molecule if the element name contains more than one upper case
        # nbUpperCase = sum([el[i].isupper() for i in range(len(el))])
        # if nbUpperCase<2:
        #     newdata.append(block)
        if is_atom(el):
            newdata.append(block)
    return newdata

def get_molecules(data):
    '''Extract the data comming from atoms only'''

    ## Go through the data and only retain atoms
    newdata = []
    for block in data:
        el = block['spec_ion'].replace("'", "").split()[0] ## The species without ionization
        ## We say it is a molecule if the element name contains more than one upper case
        # nbUpperCase = sum([el[i].isupper() for i in range(len(el))])
        # if nbUpperCase>1:
        #     newdata.append(block)
        if not is_atom(el):
            newdata.append(block)
    return newdata

def get_species_codes():
    ## This is a list of all the elements
    lele_list = [
        'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca',
        'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr'
    ]
    ## I am turning this into a dictionary
    species_code = {}
    for i in range(len(lele_list)):
        species_code[lele_list[i].strip()] = i+1
    return species_code

def get_species_data(data, species):
    ## Go through the data and only retain atoms
    newdata = []
    for block in data:
        el = block['spec_ion'] ## The species and ionization state
        if el in species:
            newdata.append(block)
    return newdata


def get_orb(configstr):
    low_orb = ''
    for i in range(len(configstr)-1, 0, -1):
        letter = configstr[i]
        if letter.isalpha():
            if letter.islower():
                low_orb = letter
                break
    if not low_orb:
        low_orb = 'x'
    return low_orb

def zeeturbo_format_line(block, ZeeTurboFormat=True):
    ## We need the first and last letters of the orbitals in the transition
    config_low = block['config_low'].strip().split()[0]
    low_orb = get_orb(config_low)
    if block['config_up'].strip()=="":
        config_up = ""
    else:
        config_up = block['config_up'].strip().split()[0]
    up_orb = get_orb(config_up)

    ## We want the full string of the transition
    element, ion = block['spec_ion'].split()
    element = element.replace("'", "")
    ion = ion.replace("'", "")
    newelement = "{} {} ".format(element, toRoman(int(ion)))
    transistr = newelement+block['coupling_low'].replace("'", '').strip()+':'+block['config_low'].replace("'", '').strip()\
        +' '+block['coupling_up'].replace("'", '').strip()+':'+block['config_up'].replace("'", '').strip()
    transistr = ' '.join(transistr.split())
    ## Then we make a list for that line
    line = "{:>10.4f} {:6.3f} {:>6.3f} {:8.3f}  {:6.1f} {:.2e} {:0.3f} '{}' '{}' {} {} "
    datalist = [float(block['wvl']), float(block['chi_low']), float(block['lggf']), float(block['gamma_waals']), 
                2*float(block['j_up'])+1, float(block['gamma_rad']), float(block['gamma_stark']), 
                low_orb, up_orb, 0, 1]
    ## If it is ZeeTurbo, we need addtional things
    if ZeeTurboFormat:
        line2 = "'{}' {:8.3f} {:8.3f} {:8.3f} {:8.3f}"
        secondlist = [transistr, float(block['j_low']), float(block['j_up']), 
                    float(block['lande_low']), float(block['lande_up'])]
    ## Create the final line
    line = line.format(*datalist)
    if ZeeTurboFormat:
        line2 = line2.format(*secondlist)
        line = line+line2

    return line

def CamelSplitMethod(string_value):
    result = [string_value[0]] 
    for char in string_value[1:]:
        if char.isupper():
            result.extend([' ', char])
        else:
            result.append(char)
    return ''.join(result).split()

def toRoman(number):
    if number==1: output = 'I'
    elif number==2: output = 'II'
    elif number==3: output = 'III'
    elif number==4: output = 'IV'
    elif number==5: output = 'V'
    elif number==6: output = 'VI'
    else:
        raise Exception('Conversion to Roman number failed for {}'.format(number))
    return output

def expend_element(element):
    newelement = ""
    value = 1
    for i in range(len(element)-1, -1, -1):
        letter = element[i]
        if letter.isnumeric():
            value = int(letter)
        else:
            for i in range(value):
                newelement+=letter
            value = 1
    newelement = newelement[::-1]
    return newelement

def zeeturbo_format_block(eldata, ZeeTurboFormat=True):
    lines = []
    el = eldata[0]['spec_ion']
    for block in eldata:
        ## sanity check
        el2 = block['spec_ion']
        if el2!=el:
            raise Exception('The list of data in zeeturbo_format_block should be for the same species')
        ## Format the line
        line = zeeturbo_format_line(block, ZeeTurboFormat)
        lines.append(line)
    ## For this block, we need information:
    nblines = len(lines) ## Number of lines for that species
    ## Get the element codes
    element, ion = el.replace("'", '').split()
    ## If molecule, we need to split the string
    element = expend_element(element)
    elements = CamelSplitMethod(element)
    ## Get the element codes
    elcodes = []
    isotopes = []
    for element in elements:
        converter = get_species_codes()
        elcode = converter[element]## The code
        elcodes.append(elcode)
        isotopes.append('000')
    elcodes.sort()
    elcodestr = "".join(map(str, elcodes))
    isotopesstr = "".join(map(str, isotopes))
    elcodestr = elcodestr+'.'+isotopesstr
    ## Convert this element to a string
    line1 = "' {:12}'   {}     {} ".format(elcodestr, ion, nblines)
    line2 = "' {} {} '".format(element, toRoman(float(ion)))
    ## Join everything into a block of line list
    strblock = line1+'\n'+line2+'\n'+"\n".join(lines)
    return strblock


def get_list_species(data):
    species = []
    for block in data:
        el = block['spec_ion']
        if el not in species:
            species.append(el)
    return species

def zeeturbo_format(data, ZeeTurboFormat=True):
    species = get_list_species(data)
    datablocks = []
    for el in species:
        eldata = get_species_data(data, el)
        strblock = zeeturbo_format_block(eldata, ZeeTurboFormat)
        datablocks.append(strblock)
    ostr = "\n".join(datablocks)
    return ostr

def set_default_value(data, column, fill, default=0.0):
    for block in data:
        _floatval = float(block[column])
        if _floatval==default:
            block[column] = str(fill)
    return data

def convert_units(data):
    for block in data:
        block['gamma_rad'] = "{:.2e}".format(10**float(block['gamma_rad']))
    return data

def to_electronvolts(data):
    hplank = 4.135667696E-15 ## eV Hz-1
    speedoflight = 2.99792458E+10 ## in cm/s
    for block in data:
        block['chi_low'] = "{:.4f}".format(hplank*speedoflight*float(block['chi_low']))
        block['chi_up'] = "{:.4f}".format(hplank*speedoflight*float(block['chi_up']))
    return data

def set_defaults(data):
    for block in data:
        if float(block['gamma_rad']) < 3:
            block['gamma_rad'] = "{:.2e}".format(10**5)
        if float(block['gamma_waals'])==0.:
            fdamp = get_default_gamma_waals(block['spec_ion'])
            block['gamma_waals'] = "{:0.3f}".format(fdamp)         
    return data

def get_default_gamma_waals(spec_ion):

    spec, ion = spec_ion.replace("'", "").split()
    atom = is_atom(spec)
    ion = int(ion)
    ## Atomic species
    ## get orbital types for atomic lines
    ## Apply FDAMP, for references See BDP (A&A 275,101) or notes below
    if atom:
        converter = get_species_codes()
        iel = converter[spec.strip()]
        if ion==1:
            if iel==11:
                ## Holweger 1971 A&A 10, 128
                fdamp=2.0
            elif iel==14:
                ## Holweger 1973 A&A 26, 275
                fdamp=1.3
            elif iel==20:
                ## O'Neill & Smith 1980 A&A 81, 100
                fdamp=1.8
            elif iel==26:
                ## Simmons & Blackwell 1982 A&A 112, 209
                ## Magain & Zhao 1996 A&A 305, 245
                fdamp=1.4
            else:
                ## Mackle et al. 1975 A&A 38, 239
                fdamp=2.5
        else: 
            if iel==20:
                fdamp=1.4
            ## from fit of H&K in the HM model to the fts intensity spectrum
            elif iel==38:
                fdamp=1.8
            ## from fit of Sr II 4077.724 in the HM model to the fts intensity spectrum
            elif iel==56:
            ## Holweger & Muller 1974 Solar Physics 39, 19
                fdamp=3.0
            else:
            ## Mackle et al. 1975 A&A 38, 239
                fdamp=2.5
    else:
        ## for molecules:
        ## Mackle et al. 1975 A&A 38, 239
        fdamp=2.5
    return fdamp

def ignore_hydrogen(data):
    newdata = []
    for block in data:
        saveblock = True ## Save the block by default
        spec_ion = block['spec_ion']
        spec, ion = spec_ion.replace("'", "").split()
        isAtom = is_atom(spec)
        if isAtom:
            converter = get_species_codes()
            iel = converter[spec]
            if iel==1:
                saveblock=False ## If hydrogen we do not save the block
        if saveblock:
            newdata.append(block)
    return newdata


import numpy as np
def compute_einstein(wvl, j1, j2, gf):

    ## Assume angstroms
    c = 3*1e8 # m/s
    c_angstrom = c*1e10
    freq = c_angstrom/wvl
    ## electron charge
    e = 1.6*1e19 # Coulomb
    ## electron mass
    me = 9.109*1e-31 # kG

    ## g1
    g1 = 2*j1+1
    ## g2
    g2 = 2*j2+1

    ## what is f12? g1*f12 = gf
    f12 = gf/g1 # !! Negligeable

    constant = 8*np.pi**2*e**2*freq**2/(me*c**3)
    coeff = constant*g1/g2
    A12 = coeff*f12
    return A12

import numpy as np
def compute_gf(wvl, j1, j2, A12):

    ## Assume angstroms
    c = 3*1e8 # m/s
    c_angstrom = c*1e10
    freq = c_angstrom/wvl
    ## electron charge
    e = 1.6*1e19 # Coulomb
    ## electron mass
    me = 9.109*1e-31 # kG

    ## g1
    g1 = 2*j1+1
    ## g2
    g2 = 2*j2+1

    constant = 8*np.pi**2*e**2*freq**2/(me*c**3)
    coeff = constant*g1/g2
    ## by definition:
    ## A12 = coeff*f12
    ## so f12 = A12 / coeff
    f12 = A12 / coeff
    gf = g1*f12
    return gf
    

import re
def read_bsyn_line(line):
    # line = "6173.3433  2.223 -3.080    0.000     1.0 0.000000  0.000 's' 'p' 0 1 'Fe I LS:3d7.(4P).4s a5P LS:3d7.(4F).4p y5D*'    1.000    0.000    2.500    0.000"
    # Regular expression to split on spaces but keep quoted substrings intact
    pattern = r"'[^']*'|\S+"
    # Find all matches
    matches = re.findall(pattern, line)
    # Remove quotes from the quoted substrings
    parsed = [match.strip("'") if match.startswith("'") and match.endswith("'") else match for match in matches]
    # print(parsed)
    for i in range(7):
        parsed[i] = float(parsed[i])
    for i in range(12, 16):
        parsed[i] = float(parsed[i])
    return parsed


def read_bsyn(file_path):
    blocks = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            # Read the first header to get the block size
            sl = lines[i].strip().replace("'", "").split()
            first_header = lines[i].strip()
            elcodestr = sl[0]  # First header line
            ion = sl[1]  # First header line
            num_data_lines = int(sl[-1])  # First header line
            i += 1
            # Skip the second header line
            second_header = lines[i].strip()  # Optional: store if needed
            i += 1
            
            # Collect the data block
            data_block = []
            lggf = [] ## Extract the oscillator strength
            for _ in range(num_data_lines):
                if i < len(lines):
                    data_block.append(lines[i].strip())
                    lggf.append(lines[i].strip().split()[2])
                    i += 1
            
            # Append the data block to the list of blocks
            blocks.append({
                "first_header": first_header,
                "elcodestr": elcodestr,
                "ion": ion,
                "num_lines": num_data_lines,
                "second_header": second_header,
                "data": data_block,
                "lggf": lggf
            })
    
    return blocks

def write_bsyn(blocks):
    ostr = ""
    for block in blocks:
        ## Update the first_header if needed:
        fh = block['first_header'].strip().split()
        if int(float(fh[-1]))!=block['num_lines']:
            fh[-1] = str(block['num_lines'])
            block['first_header'] = " ".join(fh)+'\n'
        ostr+=block['first_header']+'\n'
        ostr+=block['second_header']+'\n'
        for dataline in block['data']:
            ostr+=dataline+'\n'
    return ostr


# # Example usage
# file_path = '/Users/pcristofari/CfA/zeeturbo-syntheses/zeeturbo-spirou-runs/linelists/SPIRou_edit/3500_5.0_9000_15000_atom.bsyn_edit'
# parsed_blocks = read_bsyn(file_path)

# # Print or process the parsed blocks
# for idx, block in enumerate(parsed_blocks):
#     print(f"Block {idx + 1}:")
#     print(f"Number of lines: {block['num_lines']}")
#     print(f"Second header: {block['second_header']}")
#     print("Data:")
#     for line in block['data']:
#         print(f"  {line}")


# # # Usage
# # file_path = '/Users/pcristofari/CfA/zeeturbo-syntheses/zeeturbo-spirou-runs/linelists/SPIRou_edit/3500_5.0_9000_15000_atom.bsyn_edit'
# # blocks = parse_blocks(file_path)

# # # Example: Print headers and number of lines for each block
# # for idx, block in enumerate(blocks):
# #     print(f"Block {idx + 1} Header: {block['header'].strip()}")
# #     print(f"Number of lines in block: {len(block['data'])}")
