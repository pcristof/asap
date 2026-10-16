import re
from astroquery.simbad import Simbad
import numpy as np
import time
from importlib.resources import files


patterns = [
    r'HD[_ ]?\d+',
    r'HIP[_ ]?\d+',
    r'BD[+\-]?\d+[_ ]?\d*',
    r'TYC[_ ]?\d+-\d+-\d+',
    r'GJ[_ ]?\d+',
    r'Gl[_ ]?\d+',
    r'2MASS[_ ]?J[\d+\-]+',
]

import sqlite3
import msgpack
import zlib
import os

# === 1. Initialize Database ===
def init_db(database_fname):
    conn = sqlite3.connect(database_fname)
    cursor = conn.cursor()
    
    # Main stars table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stars (
            main_key TEXT PRIMARY KEY,
            data BLOB NOT NULL,
            timestamp INTEGER NOT NULL
        )
    ''')
    
    # Aliases table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS aliases (
            alias TEXT NOT NULL,
            main_key TEXT NOT NULL,
            FOREIGN KEY (main_key) REFERENCES stars(main_key)
        )
    ''')
    
    # Index for fast alias lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_main_key ON aliases(main_key)")
    
    conn.commit()
    conn.close()

# === 2. Save a Star with Aliases ===
def save_star(database_fname, main_key, data, aliases):
    conn = sqlite3.connect(database_fname)
    cursor = conn.cursor()
    
    # Serialize and compress data
    packed = msgpack.packb(data, use_bin_type=True)
    compressed = zlib.compress(packed)
    
    # Insert or replace main star
    cursor.execute(
        "INSERT OR REPLACE INTO stars (main_key, data, timestamp) VALUES (?, ?, ?)",
        (main_key, compressed, int(time.time()))
    )
    
    # Insert aliases
    for alias in aliases:
        cursor.execute(
            "INSERT OR REPLACE INTO aliases (alias, main_key) VALUES (?, ?)",
            (alias.lower().strip(), main_key)
        )
    
    conn.commit()
    conn.close()

# === 3. Resolve Star from Alias (Partial or Full) ===
def resolve_star(database_fname, alias):
    ## What if we're looking for a planet?
    alias = alias.replace('_', ' ',).replace(' ', '',).strip()
    alias = alias[:-1].upper().replace('GL', 'GJ',)+alias[-1]
    conn = sqlite3.connect(database_fname)
    cursor = conn.cursor()
    
    # Search for any alias matching the input (partial match)
    cursor.execute("""
        SELECT main_key
        FROM aliases
        WHERE UPPER(REPLACE(SUBSTR(alias, 1, LENGTH(alias)-1), ' ', ''))
            || SUBSTR(alias, -1) LIKE ?
    """, (f"%{alias}%",))
    main_id = cursor.fetchone()
    if main_id is not None: main_id = main_id[0]
    conn.close()
    return main_id

def read_database(database_fname, main_key):
    # Load the full data
    conn = sqlite3.connect(database_fname)
    cursor = conn.cursor()

    cursor.execute("SELECT data FROM stars WHERE main_key = ?", (main_key,))
    row = cursor.fetchone()
    if not row:
        return None
    compressed = row[0]
    packed = zlib.decompress(compressed)
    data = msgpack.unpackb(packed, raw=False)
    conn.close()

    return data

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

def query_simbad(name, return_raw=False):
    database_fname = files("asap.support_data").joinpath('stars.db')
    if os.path.isfile(database_fname):
        main_id = resolve_star(database_fname, name)
        if main_id is not None:
            data = read_database(database_fname, main_id)
            teff, logg, mh, vsini = data['teff'], data['logg'], data['mh'], data['vsini']
            return teff, logg, mh, vsini
    ## If we couldn't find it we continue
    simbad = Simbad()
    simbad.add_votable_fields('mesFe_h', 'mesRot', 'ids',
                            #'pmra', 'pmdec', 'distance', 'rv_value', 
                            #'plx', 'plx_error', 
                            #'ra', 'dec',
                            )    
    ## Query magnitudes from SIMBAD
    result_ids = simbad.query_object(name)
    ## Possible ids
    ids_match=np.all([result_ids['ids'][i]==result_ids['ids'][0] for i in range(len(result_ids['ids']))])
    if not ids_match: print('Warning - Simbad ids list don\'t match.')
    if len(result_ids)==0:
        return np.nan, np.nan, np.nan, np.nan
    main_id = result_ids['main_id'][0]
    ## Data
    success = True
    try:teff = np.nanmedian(np.array(result_ids['mesfe_h.teff']))
    except: teff = np.nan; success=False
    try:logg = np.nanmedian(np.array(result_ids['mesfe_h.log_g']))
    except: logg = np.nan; success=False
    try:mh = np.nanmedian(np.array(result_ids['mesfe_h.fe_h']))
    except: mh = np.nan; success=False
    try:vsini = np.nanmedian(np.array(result_ids['mesrot.vsini']))
    except: vsini = np.nan; success=False

    ## if success, save data to database:
    if success:
        data={'teff':float(teff), 'logg':float(logg), 'mh':float(mh), 'vsini':float(vsini)}
        aliases = result_ids['ids'][0].split('|')
        
        database_fname = files("asap.support_data").joinpath('stars.db')
        if not os.path.isfile(database_fname):
            init_db(database_fname)
        save_star(database_fname, main_id, data, aliases)
    if return_raw:
        return result_ids
    else:
        return teff, logg, mh, vsini


def main():
    import sys
    inputstr = sys.argv[1]
    print(f'Collecting simbad data for {inputstr}')
    outputstr = guess_star_name(inputstr)
    a = query_simbad(outputstr[0])    
    print('Done.')

if __name__=="__main__":
    main()

# import sqlite3
# import msgpack
# import zlib
# import os

# # === 1. Initialize Database ===
# DB_PATH = "stars.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Main stars table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS stars (
#             main_key TEXT PRIMARY KEY,
#             data BLOB NOT NULL,
#             timestamp INTEGER NOT NULL
#         )
#     ''')
    
#     # Aliases table
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS aliases (
#             alias TEXT NOT NULL,
#             main_key TEXT NOT NULL,
#             FOREIGN KEY (main_key) REFERENCES stars(main_key)
#         )
#     ''')
    
#     # Index for fast alias lookup
#     cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias)")
#     cursor.execute("CREATE INDEX IF NOT EXISTS idx_aliases_main_key ON aliases(main_key)")
    
#     conn.commit()
#     conn.close()

# # === 2. Save a Star with Aliases ===
# def save_star(main_key, data, aliases):
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Serialize and compress data
#     packed = msgpack.packb(data, use_bin_type=True)
#     compressed = zlib.compress(packed)
    
#     # Insert or replace main star
#     cursor.execute(
#         "INSERT OR REPLACE INTO stars (main_key, data, timestamp) VALUES (?, ?, ?)",
#         (main_key, compressed, int(time.time()))
#     )
    
#     # Insert aliases
#     for alias in aliases:
#         cursor.execute(
#             "INSERT OR REPLACE INTO aliases (alias, main_key) VALUES (?, ?)",
#             (alias.lower().strip(), main_key)
#         )
    
#     conn.commit()
#     conn.close()

# # === 3. Resolve Star from Alias (Partial or Full) ===
# def resolve_star(alias):
#     alias = alias.strip().lower()
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     # Search for any alias matching the input (partial match)
#     cursor.execute("""
#         SELECT DISTINCT s.main_key 
#         FROM stars s
#         JOIN aliases a ON s.main_key = a.main_key
#         WHERE a.alias LIKE ?
#     """, (f"%{alias}%",))
    
#     result = cursor.fetchone()
#     conn.close()
    
#     if not result:
#         return None
#     main_key = result[0]
    
#     # Load the full data
#     cursor.execute("SELECT data FROM stars WHERE main_key = ?", (main_key,))
#     row = cursor.fetchone()
#     if not row:
#         return None
    
#     compressed = row[0]
#     packed = zlib.decompress(compressed)
#     data = msgpack.unpackb(packed, raw=False)
    
#     return {"main_key": main_key, "data": data}

# # === 4. Example Usage ===
# if __name__ == "__main__":
#     # Initialize
#     init_db()
    
#     # Save Sirius
#     save_star(
#         main_key="Sirius",
#         data={
#             "ra": 101.287,
#             "dec": -16.716,
#             "magnitude": -1.46,
#             "type": "A1V"
#         },
#         aliases=[
#             "sir",
#             "sirius",
#             "alpha canis majoris",
#             "α cm a",
#             "canis majoris",
#             "dog star"
#         ]
#     )
    
#     # Resolve
#     result = resolve_star("sir")
#     print(result)
#     # Output: {'main_key': 'Sirius', 'data': {'ra': 101.287, 'dec': -16.716, ...}}