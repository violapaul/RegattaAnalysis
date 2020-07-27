import requests
import urllib.request
import urllib.parse
import time
from bs4 import BeautifulSoup

from global_variables import G
from utils import DictClass


def data_url(params):
    query_string = urllib.parse.urlencode(params)
    return "https://tidesandcurrents.noaa.gov/cdata/DataPlot?" + query_string

def base_url(current_station):
    return data_url(dict(id = current_station))

def find_redirect_and_parse(soup):
    for meta in soup.find_all('meta'):
        if meta.has_attr('http-equiv') and meta.has_attr('content'):
            content = meta.get('content')
            parsed = urllib.parse.parse_qs(content)
            parsed.pop('url')
            stripped = {k : v[0] for k, v in parsed.items()}
            return DictClass(**stripped)
    return None

def wget(url):
    G.logger.info(f"Fetching {url}.")
    return requests.get(url).text

def noaa_data_page(current_station):
    current_station = "PUG1515"
    response = wget(base_url(current_station))
    soup = BeautifulSoup(response, "html.parser")
    params = find_redirect_and_parse(soup)
    G.logger.info(f"Found {params}.")
    params.id = current_station
    next_url = data_url(params)
    response2 = wget(next_url)
    soup2 = BeautifulSoup(response2, "html.parser")
    soup2.find_all('small')

https://tidesandcurrents.noaa.gov/cdata/DataPlot?bin=1&bdate=20150806&edate=20150807&unit=1&timeZone=UTC&id=PUG1515
    
https://tidesandcurrents.noaa.gov/cdata/DataPlot?id=PUG1515&bin=1&bdate=20150806&edate=20150807&unit=1&timeZone=UTC

https://tidesandcurrents.noaa.gov/cdata/DataPlot?id=PUG1515&bin=1&bdate=20150806&edate=20150807&unit=1&timeZone=UTC&view=csv        
