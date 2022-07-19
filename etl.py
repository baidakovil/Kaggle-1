import numpy as np
import pandas as pd
import sklearn 


np.set_printoptions(precision=3)
pd.options.display.precision = 2
# загружаем файл
df_incoming = pd.read_csv('incoming/data.csv')
df = df_incoming.copy()

# убираем индексную колонку
df = df.drop('id', axis=1, inplace=True)

# меняем местами колонки по признаку линейной комбинации
df = df_incoming[['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 
    'f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06',
    'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 
    'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28']].copy()

# переименовываем для удобства пользования
df.columns = ['r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 
           'g0', 'g1', 'g', 'g3', 'g4', 'g5', 'g6','g14', 'g15', 'g16', 'g17', 'g18', 'g19', 'g20','g21', 'g22', 'g23',
           'b24', 'b25', 'b26', 'b27', 'b28']

# записываем новые названия колонок в 'pandas.core.indexes.base.Index'
all_col = df.columns

red_col = df.columns[0:6]
gre_col = df.columns[7:22]
blu_col = df.columns[23:29]

yel_col = df.columns[7:29]

# инфо
print(df_incoming.shape, type(df_incoming))
print(len(df), len(all_col), len(red_col), len(gre_col), len(blu_col), len(yel_col), type(df), type(all_col), type(red_col))
print(df.columns)
print(df.shape)
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(df)
print(type(x), x.shape)
x = pd.DataFrame(x,)
print(type(x), x.shape)
x.columns = df.columns
print(x.columns)
x.head(5)
x.describe()
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
x_pca.head()
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
x_pca.head()
pca.explained_variance_ratio_



