"""
All rights reserved to cnvrg.io

     http://www.cnvrg.io

flow_test.py
==============================================================================
"""
from cnvrg import Flow

flow_obj = Flow.create(file="/Users/omerliberman/Desktop/new_ailibs/_tests/yml_files/try.yml")
flow_version_obj = flow_obj.run()

ret = flow_version_obj.info()

print(ret)




