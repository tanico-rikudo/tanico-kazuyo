# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from chainer import Link, Chain, ChainList, cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import chainer.optimizers
import chainer

# Setting
batchsize = 2		# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
n_epoch   = 20		# 学習の繰り返し回数
n_units   = 1000	# 中間層の数
N = 80	#学習データ数


def read_data_pd():
	df = pd.read_csv('iris.csv',
					names = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species'],
					header=None,
					dtype = {'Sepal.Length':'float32','Sepal.Width':'float32','Petal.Length':'float32','Petal.Width':'float32','Species':'string'}
					)	#データの読み込み
	#### ダミー変数 ####
	df["Species"] = df["Species"].map( {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2} ).astype(int) #setosa・versicolor・virginica
	df[["Species"]]=df[["Species"]].astype(np.int32) #int32に変換
	df = (df.reindex(np.random.permutation(df.index))).reset_index(drop=True) #シャッフル
	x_train =  np.array(df.iloc[0:N,0:4].values)
	y_train = np.array(df.iloc[0:N,4].values)
	x_test = np.array(df.iloc[N:,0:4].values)
	y_test = np.array(df.iloc[N:,4].values)
	test_size = y_test.size

	return x_train,y_train,x_test,y_test,test_size

def read_data_np():
	data = np.genfromtxt('iris.csv',dtype = None,delimiter = ",") #CSV iris 読み込み
	x_train = [[line[0],line[1],line[2],line[3]] for line in data[:N]]
	y_train = [[line[4]] for line in data[:N]]
	x_test = [[line[0],line[1],line[2],line[3]]  for line in data[N:]]
	y_test = [[line[4]] for line in data[N:]]
	test_size = y_test.size

	return x_train,y_train,x_test,y_test,test_size

class chain_model(Chain):

	def __init__(self):
		# Prepare multi-layer perceptron model
		# 多層パーセプトロンモデルの設定
		input_dim  = 4 # 入力 4次元 
		output_dim = 3 # 出力 2次元
		super(chain_model,self).__init__(
							l1=F.Linear(input_dim, n_units),
							l2=F.Linear(n_units, n_units),
							l3=F.Linear(n_units, output_dim)
							)

	def forward_propagation(self, x_data, y_data, train=True):
		# Neural net architecture
		# ニューラルネットの構造
		x, t = Variable(x_data), Variable(y_data)			#それぞれをvariable型のオブジェクトに変換
		h1 = F.dropout(F.relu(self.l1(x)),  train=train)	#l1 活性化関数としてRELU 
		h2 = F.dropout(F.relu(self.l2(h1)), train=train)	#l2
		#dropout関数
		#隠れ層をランダムに消す。過学習防止
		# ratio: 0を出力する確率
		# train: Falseの場合はxをそのまま返却する
		# return: ratioの確率で0を、1−ratioの確率で,x*(1/(1-ratio))の値を返す

		y  = self.l3(h2) #l3 出力


		# 多クラス分類なので誤差関数としてソフトマックス関数の
		# 交差エントロピー関数を用いて、誤差を導出
		#print x
		#print t
		return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def training ():
	x_train,y_train,x_test,y_test,test_size =  read_data_pd()

	model = chain_model()
	optimizer = optimizers.Adam()
	optimizer.setup(model) #optimizer.setup(model.collect_parameters()) を切り替え（非推奨）

	train_loss = []
	train_acc  = []

	print "----- Learing LOOP -----"
	for epoch in xrange(1, n_epoch+1):
		perm = np.random.permutation(N) # array([3,2,4,5,1,0])みたいに入れ替え
		sum_accuracy = 0
		sum_loss = 0

		# 0〜Nまでのデータをバッチサイズごとに使って学習
		for i in xrange(0, N, batchsize):
			x_batch = x_train[perm[i:i+batchsize]]
			y_batch = y_train[perm[i:i+batchsize]]
			optimizer.zero_grads() # 勾配を初期化

			loss, acc = model.forward_propagation(x_batch, y_batch) # 順伝搬
			loss.backward() # 誤差逆伝播で勾配を計算
			optimizer.update() # 重みの更新

			train_loss.append(loss.data)
			train_acc.append(acc.data)
			sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize #GPU不使用なので
			sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

		print 'epoch = {}\t mean loss={}\t accuracy={}'.format(epoch ,sum_loss / N, sum_accuracy / N)


	return model

def testing(model):
	print "----- Testing LOOP -----"
	#流れは同じ
	x_train,y_train,x_test,y_test,test_size =  read_data_pd()
	sum_accuracy = 0
	sum_loss     = 0
	test_loss = []
	test_acc  = []

	for i in xrange(0, test_size, batchsize):

		x_batch = x_test[i:i+batchsize]
		y_batch = y_test[i:i+batchsize]
		loss, acc = model.forward_propagation(x_batch, y_batch, train=False) #テストなのでTRUE
		test_loss.append(loss.data)
		test_acc.append(acc.data)
		sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
		sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

	print 'test  mean loss={}, accuracy={}'.format(sum_loss / test_size, sum_accuracy / test_size)
	drow_graph("test",test_acc)
	return
def model_optional(model):
	# 学習したパラメーターを保存

	l1_W = []
	l2_W = []
	l3_W = []
	l1_W.append(model.l1.W)
	l2_W.append(model.l2.W)
	l3_W.append(model.l3.W)

	#将来的に保存処理？

def drow_graph(type,acc):
	# 精度と誤差をグラフ描画
	plt.figure()
	plt.plot(range(len(acc)), acc)
	st = "acc : " + type
	plt.legend([st],loc=4)
	plt.title("Accuracy")
	plt.plot()
	plt.show()


if __name__ == '__main__':

	model =  training()
	testing(model)
	model_optional(model)