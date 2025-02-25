#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
from preparation import validate

with open("./config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    set_name = list(config["dataset_path"].keys())[1]
    validate(set_name)

