import pandas as pd
import numpy as np

from config import RawTrainConfig, RawTestConfig

class Features(object):

	def __init__(self, df_x, df_y, df_z):
		self.acc_data = []
		self.acc_data.extend((df_x, df_y, df_z))


	def get_mi(self):
		self.mi_array = np.empty(self.acc_data[0].shape)
			
		for df in self.acc_data:
			self.df = df.apply(np.square, axis=1)			
			self.mi_array = np.add(self.mi_array, self.df)
			
		self.df_mi = pd.DataFrame(np.sqrt(self.mi_array))

		return self.df_mi


	def get_ai(self):
		self.df_mi = self.get_mi()
		self.ai = self.df_mi.apply(np.mean, axis=1)
		
		return pd.DataFrame(self.ai.reshape(self.ai.shape[0],1))


	def get_vi(self):
		self.df_mi = self.get_mi()
		self.vi =  self.df_mi.apply(np.var, axis=1)
		
		return pd.DataFrame(self.vi.reshape(self.vi.shape[0],1))



def set_features():

	configs = [ RawTrainConfig(), RawTestConfig() ]

	for config in configs:
		har = config.DF		
		df_x = config.X_DF
		df_y = config.Y_DF
		df_z = config.Z_DF
		
		ft = Features(df_x, df_y, df_z)

		ai = ft.get_ai()
		vi = ft.get_vi()

		har['tBody_AI'] = ai
		har['tBody_VI'] = vi

		har.to_csv(config.HAR_PATH)


if __name__ == '__main__':
	set_features()
