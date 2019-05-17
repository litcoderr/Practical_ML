# -*- coding: utf-8 -*-
#%%
import urllib
import re

def get_quote(symbol):
    base_url = 'http://finance.google.com/finance?q='
    content = urllib.urlopen(base_url + symbol).read()
    m = re.search('id="ref_694653_l".*?>(.*?)<', content)
    if m:
        quote = m.group(1)
    else:
        quote = 'no quote available for: ' + symbol
    return quote

#%%
if __name__ == "__main__":
    print(get_quote("APPL"))