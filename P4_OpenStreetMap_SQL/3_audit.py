#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import required libraries

import xml.etree.cElementTree as ET
from collections import defaultdict
import pprint
import re
import os


os.chdir("/Users/sarauenoyama/AnacondaProjects/Udacity_DAND/P4Project_Resources")

OSMFILE = "singapore.osm"
OSM_PATH = "singapore.osm"

# OSMFILE = "create_sample_singapore.osm"
# OSM_PATH = "create_sample_singapore.osm"

osm_file = open(OSMFILE, "r")


# audit street names
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

# expected street names
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Jalan", "Lorong", "Close", "Crescent","Lengkong"]
# "Jalan" and "Lorong" mean street in Malay
# "Lengkong" means curve in Malay 

# update this "mapping" variable

# initial variable of mapping
'''
mapping = {'St': 'Street',
           'St.': 'Street',
           'Ave': 'Avenue',
            'Rd.': 'Road'}
'''
# updated variable of mapping
mapping = {'St': 'Street',
           'St.': 'Street',
           'Ave': 'Avenue',
           'Rd.': 'Road',
           'Rd' : 'Road',
           'rd' : 'Road',
           'jln' : 'Jalan',
           'Jln' : 'Jalan',
           'Lor' : 'Lorong : path',
           'Blk' : 'Block',
           'blk' : 'Block',
           'Cl' : 'Close',
           'Dr' : 'Drive',
           'Bkt' : 'Bukit',
           'Bt' : 'Bukit',
           'Upp' : 'Upper'}

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile):
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osmfile, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                    
    return street_types

def update_name(name, mapping):
    # YOUR CODE HERE

    return street_type_re.sub(lambda x: mapping[x.group()], name)

def test():
    st_types = audit(OSMFILE)
    #assert len(st_types) == 3
    pprint.pprint(dict(st_types))
    

st_types = audit(OSMFILE)

pprint.pprint(st_types)