import re
from astroquery.simbad import Simbad
import numpy as np

patterns = [
    r'HD[_ ]?\d+',
    r'HIP[_ ]?\d+',
    r'BD[+\-]?\d+[_ ]?\d*',
    r'TYC[_ ]?\d+-\d+-\d+',
    r'GJ[_ ]?\d+',
    r'Gl[_ ]?\d+',
    r'2MASS[_ ]?J[\d+\-]+',
]

def guess_star_name(s):
    s = s.replace('output', '')
    if s[0]=='_': s = s[1:]
    candidates = []

    for p in patterns:
        for m in re.finditer(p, s, re.IGNORECASE):
            candidates.append(m.group())
    candidates.append(s)
    sl = s.split('-')
    candidates.append(sl[0])
    return candidates

if __name__=="__main__":
	import sys
	inputstr = sys.argv[1]
	outputstr = guess_star_name(inputstr)
	print(outputstr)
    
def query_simbad(name):
    simbad = Simbad()
    simbad.add_votable_fields('mesFe_h', 
                            #'pmra', 'pmdec', 'distance', 'rv_value', 
                            #'plx', 'plx_error', 
                            #'ra', 'dec',
                            )    
    ## Query magnitudes from SIMBAD
    result_ids = simbad.query_object(name)
    try:teff = np.nanmedian(np.array(result_ids['mesfe_h.teff']))
    except: teff = np.nan
    try:logg = np.nanmedian(np.array(result_ids['mesfe_h.log_g']))
    except: logg = np.nan
    try:mh = np.nanmedian(np.array(result_ids['mesfe_h.fe_h']))
    except: mh = np.nan
    try:vsini = np.nanmedian(np.array(result_ids['mesrot.vsini']))
    except: vsini = np.nan
    return teff, logg, mh, vsini
