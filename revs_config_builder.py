# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:44:14 2022

@author: rm5nz
"""

from configbuilder.validator import Validator
from configbuilder.parser import create_parser
import pprint

class MyValidator(Validator):
    def validate_protocol(self, value):
        """ Add validator method for type "Protocol"
        "Protocol" is a sub type of string.
        """
        supporting_protocol = ['ssh', 'ftp'] 
        return self._validate_choiceses(value, supporting_protocol)

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    parser = create_parser('./config_template.yaml',
                          validator=MyValidator(), 
                          casesensitivekey=False)
    cfg = parser.parse_configs('./revs_config.yaml')
    print (parser.get_keys())
    print (cfg.get('datadir'))
    print (cfg.get('nodelist/node1/ip'))
    node1cfg = cfg.get('nodelist/node1')
    print (node1cfg.get('ip'))
    print (cfg.get('tags'))
    pp.pprint(cfg.dump())