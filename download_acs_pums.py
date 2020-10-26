'''
auto download ACS PUMS data from Census

chris zhang 10/26/2020
'''
import requests
from random import randint
import os
import shutil
import zipfile
import pandas as pd
from collections import OrderedDict
import time

## Set dir to save data
dir_save = './data/census_download/'

## Make st/st_code list
fips = pd.read_excel('./data/fips_state_code.xlsx')
fips['st'] = [x.lower() for x in fips['st']]
fips['st_code'] = [str(x) for x in fips['st_code']]
fips['st_code'] = ['0' + x if len(x)==1 else x for x in fips['st_code']]
dct_fips = OrderedDict(zip(fips['st'], fips['st_code']))

# a function to download and extract CSV
def download_url_and_extract(dir_save, file_type, yr, st, st_code, chunk_size=128):
    # get url
    url = 'https://www2.census.gov/programs-surveys/acs/data/pums/20%s/5-Year/csv_%s%s.zip' % (yr, file_type, st)
    # set up fps
    fp_zip_save = dir_save + 'ss%s%s%s.zip' % (yr, file_type, st)
    # download URL of ZIP
    r = requests.get(url, stream=True)
    with open(fp_zip_save, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    # extract CSV from ZIP
    archive = zipfile.ZipFile(fp_zip_save)
    for file in archive.namelist():
        if file.startswith('psam'):
            archive.extract(file, dir_save)
    # rename CSV
    os.rename(dir_save + 'psam_%s%s.csv' % (file_type, st_code), dir_save +'ss%s%s%s.csv' % (yr, file_type, st))
    return None

## Download and Extract CSV
file_type = 'p' # 'p' for person file, 'h' for household file
for st, st_code in dct_fips.items():
    # a random pause to let server rest
    time.sleep(randint(5, 10))
    # set download state list
    sts = ['ak', 'al', 'ar'] # test
    #sts = dct_fips.keys()
    # download
    if st in sts:
        print('---------- file type = %s, now working on state %s ... -----------' % (file_type, st.upper()))
        yr = '18'
        fp_zip_save = dir_save + 'ss%sh%s.zip' % (yr, st)
        download_url_and_extract(dir_save, file_type, yr, st, st_code)
        print('---------- State %s CSV file successfully extracted -----------' % st.upper())
