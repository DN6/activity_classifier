import os
import numpy as np 
import pandas as pd

java -jar target/converter-executable-1.1-SNAPSHOT.jar
--pkl-mapper-input mapper.pkl --pkl-estimator-input boosted_tree.pkl --pmml-output mapper-boosted_tree.pmml