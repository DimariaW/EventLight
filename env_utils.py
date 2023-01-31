#!/usr/bin/env python
# coding=utf8
# File: env_utils.py
import json
import os

def number_of_cars(scenario):
  with open(scenario) as t_file:
    scenario = json.load(t_file)
  flow_file = os.path.join(scenario['dir'], scenario['flowFile'])
  with open(flow_file) as t_file:
    car_flow = json.load(t_file)
  return (len(car_flow))
